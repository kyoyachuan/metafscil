from email.policy import strict
import os
import random
import argparse

import torch

from metafscil.models import get_model
from metafscil.dataset import get_pretrain_dataloader, SequentialTaskSampler, EpisodeSampler
from metafscil.trainer import Pretrain, MetaFSCIL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--model_type', type=str, default='sfm')
    parser.add_argument('--model_name', type=str, default='sfm')
    parser.add_argument('--cos_cls', action='store_true')
    parser.add_argument('--temperature', type=int, default=16)

    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_lr', type=float, default=0.1)
    parser.add_argument('--pretrain_epoch', type=int, default=100)
    parser.add_argument('--pretrain_batch_size', type=int, default=128)

    parser.add_argument('--meta_epoch', type=int, default=200)
    parser.add_argument('--meta_lr', type=float, default=0.001)

    parser.add_argument('--eval_only', action='store_true')

    return parser.parse_args()


def pretrain(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print('Pretraining...')

    os.makedirs(f"models/{args.model_name}", exist_ok=True)

    train_loader = get_pretrain_dataloader(
        'metadata/mini_imagenet/session_1.txt', args.pretrain_batch_size, transform=True, shuffle=True)
    val_loader = get_pretrain_dataloader('metadata/mini_imagenet/test_1.txt', args.pretrain_batch_size,
                                         transform=False, shuffle=False)
    model = get_model(args.model_type, 60, cos_cls=args.cos_cls, temperature=args.temperature).to(args.device)

    trainer = Pretrain(model, train_loader, val_loader, args)
    trainer.train(args.pretrain_epoch)

    # torch.save(model.state_dict(), f'models/{args.model_name}/pretrain.pth')


def metatrain(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print('Metatrain...')

    sampler = SequentialTaskSampler('metadata/mini_imagenet/session_1.txt')

    model = get_model(args.model_type, 60, cos_cls=args.cos_cls, temperature=args.temperature).to(args.device)
    model.load_state_dict(torch.load(f'models/{args.model_name}/pretrain.pth'))

    trainer = MetaFSCIL(model, sampler, args)
    trainer.train(args.meta_epoch)

    torch.save(trainer.model.state_dict(), f'models/{args.model_name}/metatrain.pth')


def evaluate(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print('Evaluate...')

    sampler = EpisodeSampler()

    model = get_model(args.model_type, 60, cos_cls=args.cos_cls, temperature=args.temperature).to(args.device)
    model.load_state_dict(torch.load(f'models/{args.model_name}/metatrain.pth'), strict=False)

    trainer = MetaFSCIL(model, sampler, args)
    trainer.meta_test()


if __name__ == '__main__':
    args = parse_args()

    if not args.eval_only:
        if args.pretrain:
            pretrain(args)
        metatrain(args)
    evaluate(args)
