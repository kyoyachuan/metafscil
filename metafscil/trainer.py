import copy

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .models import IncrementalLinear
from .utils import disable_grad, compute_accuracy


class Pretrain:
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = args.pretrain_lr
        self.device = args.device

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                         momentum=0.9, weight_decay=0.0005, nesterov=True)
        self.schelduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 70], gamma=0.1)
        self.writer = SummaryWriter(f"{args.log_dir}/{args.model}")

    def train(self, epochs: int = 100):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            self.evaluate(epoch)
            self.schelduler.step()

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        train_accuracy = 0
        with tqdm(self.train_loader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch}')
            for data, target in tepoch:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                accuracy = compute_accuracy(output, target)
                loss.backward()
                self.optimizer.step()
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)

                train_loss += loss.item()
                train_accuracy += accuracy

        train_loss /= len(self.train_loader)
        train_accuracy /= len(self.train_loader)

        self.writer.add_scalar('Pretrain/train_loss', train_loss, epoch)
        self.writer.add_scalar('Pretrain/train_accuracy', train_accuracy, epoch)

    def evaluate(self, epoch):
        self.model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss(output, target).item()
                val_accuracy += compute_accuracy(output, target)
        val_loss /= len(self.val_loader)
        val_accuracy /= len(self.val_loader)

        self.writer.add_scalar('Pretrain/val_loss', val_loss, epoch)
        self.writer.add_scalar('Pretrain/val_accuracy', val_accuracy, epoch)


class MetaFSCIL:
    def __init__(self, model, task_sampler, args):
        self.model = model.to(args.device)
        self.task_sampler = task_sampler
        self.n_sessions = 9
        self.lr = args.meta_lr
        self.device = args.device
        self.in_features = model.classifier.in_features

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.writer = SummaryWriter(f"{args.log_dir}/{args.model}")

    def init_classifier(self, base_classes: int = 20):
        self.model.classifier = IncrementalLinear(self.in_features, base_classes, self.device)

    def accumulate_classifier(self, num_classes):
        self.fast_model.classifier.add_layer(num_classes)

    def warm_up(self, session, n_steps=20, scale=None):
        disable_grad(self.fast_model, ['feature_extractor', 'modulation'])
        optimizer = torch.optim.SGD(self.fast_model.parameters(), lr=0.1)
        self.fast_model.train(False)
        for _ in range(n_steps):
            data, target = self.task_sampler.sample_support()
            if scale:
                target = target % scale
            data, target = data.to(self.device), target.to(self.device)
            output = self.fast_model(data, layer=session)
            loss = self.loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def fast_adapt(self, n_steps=5):
        disable_grad(self.fast_model, ['modulation'])
        optimizer = torch.optim.SGD(self.fast_model.parameters(), lr=self.lr)
        self.fast_model.train(False)
        for _ in range(n_steps):
            data, target = self.task_sampler.sample_support()
            data, target = data.to(self.device), target.to(self.device)
            output = self.fast_model(data)
            loss = self.loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def meta_train(self, epoch):
        total_loss = 0
        total_acc = 0
        self.init_classifier(self.task_sampler.n_base_task)
        self.task_sampler.new_sequence()
        self.model.train(False)
        # self.fast_model = copy.deepcopy(self.model)
        with tqdm(range(self.n_sessions), unit='session') as tepoch:
            tepoch.set_description(f'Epoch {epoch}')
            for session in tepoch:
                self.task_sampler.new_session()
                self.fast_model = copy.deepcopy(self.model)
                if session > 0:
                    self.accumulate_classifier(self.task_sampler.n_way)
                self.warm_up(
                    session,
                    scale=None if session == 0 else self.task_sampler.n_way
                )

                self.fast_adapt()
                self.model.classifier = copy.deepcopy(self.fast_model.classifier)
                fast_params = list(self.fast_model.feature_extractor.parameters())
                del self.fast_model
                loss, acc = self.meta_update(fast_params, session)
                total_loss += loss
                total_acc += acc

                tepoch.set_postfix(loss=loss, accuracy=acc)

        #     self.model.classifier = copy.deepcopy(self.fast_model.classifier)
        #     fast_params = list(self.fast_model.feature_extractor.parameters())
        #     loss, acc = self.meta_update(fast_params, session)

        #     tepoch.set_postfix(loss=loss, accuracy=acc)

        # self.writer.add_scalar('Meta/train_loss', loss, epoch)
        # self.writer.add_scalar('Meta/train_accuracy', acc, epoch)

        self.writer.add_scalar('Meta/train_loss', total_loss / self.n_sessions, epoch)
        self.writer.add_scalar('Meta/train_accuracy', total_acc / self.n_sessions, epoch)

    def meta_update(self, params, session):
        disable_grad(self.model, ['classifier'])
        self.model.train(False)
        data, target = self.task_sampler.sample_query()
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data, params)
        loss = self.loss(output, target)
        # loss = loss / self.task_sampler.session_n_class
        with torch.no_grad():
            accuracy = compute_accuracy(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), accuracy

    def train(self, epochs):
        for epoch in range(epochs):
            self.meta_train(epoch)

    def meta_test(self, n_episodes=9):
        acc_dict = dict()
        self.init_classifier(self.task_sampler.n_base_task)
        self.model.train(False)
        with tqdm(range(n_episodes), unit='session') as tepoch:
            tepoch.set_description(f'Evaluation')
            for session in tepoch:
                self.model.train()
                self.task_sampler.new_session()
                self.fast_model = copy.deepcopy(self.model)
                if session > 0:
                    self.accumulate_classifier()
                self.warm_up(
                    session,
                    scale=None if session == 0 else self.task_sampler.n_way
                )

                self.fast_adapt()
                self.model = copy.deepcopy(self.fast_model)

                self.model.eval()
                with torch.no_grad():
                    acc = 0
                    for img, target in self.task_sampler.query_loader:
                        img, target = img.to(self.device), target.to(self.device)
                        output = self.model(img)
                        acc += compute_accuracy(output, target)
                    acc /= len(self.task_sampler.query_loader)

                tepoch.set_postfix(accuracy=acc)
                acc_dict[session] = acc

                self.writer.add_scalar('Meta/eval_accuracy', acc, session)

        print(f"Average accuracy: {sum(acc_dict.values()) / len(acc_dict)}")
