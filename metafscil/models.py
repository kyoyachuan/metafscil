import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.query = nn.Conv1d(in_channel, in_channel // 4, 1)
        self.key = nn.Conv1d(in_channel, in_channel // 4, 1)
        self.value = nn.Conv1d(in_channel, in_channel, 1)

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = torch.sigmoid(attn) * input

        return out


class SelfModulation(nn.Module):
    def __init__(self, num_classes):
        super(SelfModulation, self).__init__()
        self.feature_extractor = models.resnet18()
        self.modulation_1 = SelfAttention(64)
        self.modulation_2 = SelfAttention(64)
        self.modulation_3 = SelfAttention(128)
        self.modulation_4 = SelfAttention(256)
        self.modulation_5 = SelfAttention(512)
        self.classifier = nn.Linear(self.feature_extractor.fc.in_features, num_classes)

        del self.feature_extractor.fc

    def forward(self, x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.modulation_1(x)
        x = self.feature_extractor.maxpool(x)

        x = self.feature_extractor.layer1(x)
        x = self.modulation_2(x)
        x = self.feature_extractor.layer2(x)
        x = self.modulation_3(x)
        x = self.feature_extractor.layer3(x)
        x = self.modulation_4(x)
        x = self.feature_extractor.layer4(x)
        x = self.modulation_5(x)

        x = self.feature_extractor.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def get_model(model_str):
    if model_str == 'selfmodulation':
        return SelfModulation
    else:
        raise ValueError('Unknown model: {}'.format(model_str))
