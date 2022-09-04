import os
import random
import argparse

import torch

from metafscil.models import get_model
from metafscil.dataset import get_pretrain_dataloader, SequentialTaskSampler
from metafscil.trainer import Pretrain, MetaFSCIL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model', type=str, default='sfm')

    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_lr', type=float, default=0.1)
    parser.add_argument('--pretrain_epoch', type=int, default=100)
    parser.add_argument('--pretrain_batch_size', type=int, default=128)

    parser.add_argument('--meta_epoch', type=int, default=200)
    parser.add_argument('--meta_lr', type=float, default=0.001)

    parser.add_argument('--eval_only', action='store_true')

    return parser.parse_args()


def pretrain(args):
    print('Pretraining...')

    os.makedirs(f"models/{args.model}", exist_ok=True)

    train_loader = get_pretrain_dataloader(
        'metadata/mini_imagenet/session_1.txt', args.pretrain_batch_size, transform=True, shuffle=True)
    val_loader = get_pretrain_dataloader('metadata/mini_imagenet/test_1.txt', args.pretrain_batch_size,
                                         transform=False, shuffle=False)
    model = get_model(args.model, 60).to(args.device)

    trainer = Pretrain(model, train_loader, val_loader, args)
    trainer.train(args.pretrain_epoch)

    torch.save(model.state_dict(), f'models/{args.model}/pretrain.pth')


def metatrain(args):
    print('Metatrain...')

    sampler = SequentialTaskSampler('metadata/mini_imagenet/session_1.txt')

    model = get_model(args.model, 60).to(args.device)
    model.load_state_dict(torch.load(f'models/{args.model}/pretrain.pth'))

    trainer = MetaFSCIL(model, sampler, args)
    trainer.train(args.meta_epoch)

    torch.save(model.state_dict(), f'models/{args.model}/metatrain.pth')


def evaluate(args):
    pass


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not args.eval_only:
        if args.pretrain:
            pretrain(args)
        metatrain(args)
    # evaluate(args)
