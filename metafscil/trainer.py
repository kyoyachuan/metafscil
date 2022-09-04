import copy

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import disable_grad, compute_accuracy


class Pretrain:
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = args.pretrain_lr
        self.device = args.device

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.schelduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 70], gamma=0.1)
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

        self.loss = nn.CrossEntropyLoss()

        self.writer = SummaryWriter(f"{args.log_dir}/{args.model}")

    def init_classifier(self, base_classes: int = 20):
        self.previous_classifier = None
        self.model.classifier = nn.Linear(self.model.classifier.in_features, base_classes).to(self.device)

    def backup_classifier(self):
        self.previous_classifier = copy.deepcopy(self.model.classifier)

    def accumulate_classifier(self):
        current_classifier = copy.deepcopy(self.fast_model.classifier)
        out_features = self.previous_classifier.out_features + current_classifier.out_features
        self.fast_model.classifier = nn.Linear(current_classifier.in_features, out_features).to(self.device)
        self.fast_model.classifier.weight.data = torch.concat(
            (self.previous_classifier.weight.data, current_classifier.weight.data), dim=0)

    def warm_up(self, n_steps=20, scale=None):
        disable_grad(self.fast_model, ['feature_extractor', 'modulation'])
        optimizer = torch.optim.SGD(self.fast_model.parameters(), lr=0.1)
        for _ in range(n_steps):
            data, target = self.task_sampler.sample_support()
            if scale:
                target = target % scale
            data, target = data.to(self.device), target.to(self.device)
            output = self.fast_model(data)
            loss = self.loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def fast_adapt(self, n_steps=5):
        disable_grad(self.fast_model, ['modulation'])
        optimizer = torch.optim.SGD(self.fast_model.parameters(), lr=self.lr)
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
        self.model.train()
        with tqdm(range(self.n_sessions), unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch}')
            for session in tepoch:
                self.task_sampler.new_session()
                if session > 0:
                    self.backup_classifier()
                    self.model.classifier = nn.Linear(
                        self.model.classifier.in_features, self.task_sampler.n_way).to(
                        self.device)
                self.fast_model = copy.deepcopy(self.model)
                self.warm_up(
                    scale=None if session == 0 else self.task_sampler.n_way
                )
                if session > 0:
                    self.accumulate_classifier()

                self.fast_adapt()
                self.model.classifier = copy.deepcopy(self.fast_model.classifier)
                fast_params = list(self.fast_model.feature_extractor.parameters())
                del self.fast_model
                loss, acc = self.meta_update(fast_params)
                total_loss += loss
                total_acc += acc

                tepoch.set_postfix(loss=loss, accuracy=acc)

        self.writer.add_scalar('Meta/train_loss', total_loss / self.n_sessions, epoch)
        self.writer.add_scalar('Meta/train_accuracy', total_acc / self.n_sessions, epoch)

    def meta_update(self, params):
        disable_grad(self.model, ['classifier'])
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        data, target = self.task_sampler.sample_query()
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data, params)
        loss = self.loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), compute_accuracy(output, target)

    def train(self, epochs):
        for epoch in range(epochs):
            self.meta_train(epoch)

    def meta_test(self, n_episodes=9):
        acc_dict = dict()
        self.init_classifier(self.task_sampler.n_base_task)
        with tqdm(range(n_episodes), unit='session') as tepoch:
            tepoch.set_description(f'Evaluation')
            for session in tepoch:
                self.model.train()
                self.task_sampler.new_session()
                if session > 0:
                    self.backup_classifier()
                    self.model.classifier = nn.Linear(
                        self.model.classifier.in_features, self.task_sampler.n_way).to(
                        self.device)
                self.fast_model = copy.deepcopy(self.model)
                self.warm_up(
                    scale=None if session == 0 else self.task_sampler.n_way
                )
                if session > 0:
                    self.accumulate_classifier()

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
