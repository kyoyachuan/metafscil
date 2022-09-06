import torch
from torch import nn


def enable_grad(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def disable_grad(model: nn.Module, patterns: list):
    for p in model.named_parameters():
        if any([pat in p[0] for pat in patterns]):
            p[1].requires_grad = False
        else:
            p[1].requires_grad = True


def compute_accuracy(output: torch.Tensor, target: torch.Tensor):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / len(target)


def cosine_classify(x: torch.Tensor, y: torch.Tensor, temperature: int, normalize: bool = True):
    if normalize:
        x = nn.functional.normalize(x, p=2, dim=-1)
        y = nn.functional.normalize(y, p=2, dim=-1)
    return nn.functional.linear(x, y, bias=None) * temperature
