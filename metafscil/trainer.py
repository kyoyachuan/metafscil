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
    def __init__():
        pass

    def init_classifier():
        pass

    def accumulate_classifier():
        pass

    def warm_up():
        pass

    def fast_adapt():
        pass

    def meta_train():
        pass

    def meta_test():
        pass
