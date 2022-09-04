import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Linear(in_channel, in_channel // 2, 1)
        self.fc_2 = nn.Linear(in_channel // 2, in_channel, 1)

    def forward(self, input):
        flatten = torch.flatten(self.avg_pool(input), 1)
        out = torch.relu(self.fc_1(flatten))
        out = torch.sigmoid(self.fc_2(out))
        out = out.unsqueeze(-1).unsqueeze(-1)
        out = out * input

        return out


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
        out = attn * input

        return out


class SelfModulation(nn.Module):
    def __init__(self, att_mode, num_classes):
        super(SelfModulation, self).__init__()
        self.att_mode = att_mode
        self.feature_extractor = models.resnet18()
        self.modulation_1 = SelfAttention(64) if att_mode == 'sfm' else ChannelAttention(64)
        self.modulation_2 = SelfAttention(128) if att_mode == 'sfm' else ChannelAttention(128)
        self.modulation_3 = SelfAttention(256) if att_mode == 'sfm' else ChannelAttention(256)
        self.modulation_4 = ChannelAttention(512)
        self.modulation_5 = ChannelAttention(512)
        self.classifier = nn.Linear(self.feature_extractor.fc.in_features, num_classes)

        del self.feature_extractor.fc

    def forward(self, x):
        # pre layer
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.modulation_1(x)

        # layer 1
        x = self.feature_extractor.maxpool(x)
        x = self.feature_extractor.layer1(x)

        # layer 2
        x_skip_1 = x
        x = self.feature_extractor.layer2[0].conv1(x)
        x = self.feature_extractor.layer2[0].bn1(x)
        x = self.feature_extractor.layer2[0].relu(x)
        x = self.modulation_2(x)
        x = self.feature_extractor.layer2[0].conv2(x)
        x = self.feature_extractor.layer2[0].bn2(x)
        x_skip_1 = self.feature_extractor.layer2[0].downsample(x_skip_1)
        x = x + x_skip_1
        x = self.feature_extractor.layer2[1](x)

        # layer 3
        x_skip_2 = x
        x = self.feature_extractor.layer3[0].conv1(x)
        x = self.feature_extractor.layer3[0].bn1(x)
        x = self.feature_extractor.layer3[0].relu(x)
        x = self.modulation_3(x)
        x = self.feature_extractor.layer3[0].conv2(x)
        x = self.feature_extractor.layer3[0].bn2(x)
        x_skip_2 = self.feature_extractor.layer3[0].downsample(x_skip_2)
        x = x + x_skip_2
        x = self.feature_extractor.layer3[1](x)

        # layer 4
        x_skip_3 = x
        x = self.feature_extractor.layer4[0].conv1(x)
        x = self.feature_extractor.layer4[0].bn1(x)
        x = self.feature_extractor.layer4[0].relu(x)
        x = self.modulation_4(x)
        x = self.feature_extractor.layer4[0].conv2(x)
        x = self.feature_extractor.layer4[0].bn2(x)
        x_skip_3 = self.feature_extractor.layer4[0].downsample(x_skip_3)
        x = x + x_skip_3

        x = self.feature_extractor.layer4[1](x)

        x = self.feature_extractor.avgpool(x)
        x = self.modulation_5(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class BGM(nn.Module):
    def __init__(self, num_classes):
        super(BGM, self).__init__()
        self.feature_extractor = models.resnet18()
        self.modulation = models.resnet18()
        self.modulation_fc_1 = nn.Sequential(
            nn.Linear(64*3, 64),
            nn.ReLU(),
            nn.Linear(64, 64*3)
        )
        self.modulation_fc_2 = nn.Sequential(
            nn.Linear(128*64, 128),
            nn.ReLU(),
            nn.Linear(128, 128*64)
        )
        self.modulation_fc_3 = nn.Sequential(
            nn.Linear(256*128, 256),
            nn.ReLU(),
            nn.Linear(256, 256*128)
        )
        self.modulation_fc_4 = nn.Sequential(
            nn.Linear(512*256, 512),
            nn.ReLU(),
            nn.Linear(512, 512*256)
        )
        self.modulation_fc_5 = nn.Sequential(
            nn.Linear(512*512, 512),
            nn.ReLU(),
            nn.Linear(512, 512*512)
        )
        self.classifier = nn.Linear(self.feature_extractor.fc.in_features, num_classes)

        del self.feature_extractor.fc
        del self.modulation.fc

    def _translate_weight(self, input, value, translator):
        out = F.adaptive_avg_pool2d(input, (1, 1))
        out = out.view(1, -1)
        out = torch.sigmoid(translator(out))
        out = out.view(input.shape[0], input.shape[1])
        out = out.unsqueeze(-1).unsqueeze(-1)
        out = out * value
        return out

    def forward(self, input):
        # pre layer
        x = self.feature_extractor.conv1(input)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        ref_w_1 = self._translate_weight(self.feature_extractor.conv1.weight.data,
                                         self.modulation.conv1.weight.data, self.modulation_fc_1)
        x_ref = self.modulation.conv1._conv_forward(input, ref_w_1, None)
        x_ref = self.modulation.bn1(x_ref)
        x_ref = torch.sigmoid(x_ref)
        x = x * x_ref

        # layer 1
        x = self.feature_extractor.maxpool(x)
        x = self.feature_extractor.layer1(x)
        x_ref = self.modulation.maxpool(x_ref)
        x_ref = self.modulation.layer1(x_ref)

        # layer 2
        x_skip_1 = x
        x_skip_ref_1 = x_ref
        x = self.feature_extractor.layer2[0].conv1(x)
        x = self.feature_extractor.layer2[0].bn1(x)
        x = self.feature_extractor.layer2[0].relu(x)
        ref_w_2 = self._translate_weight(
            self.feature_extractor.layer2[0].conv1.weight.data,
            self.modulation.layer2[0].conv1.weight.data,
            self.modulation_fc_2
        )
        x_ref = self.modulation.layer2[0].conv1._conv_forward(x_ref, ref_w_2, None)
        x_ref = self.modulation.layer2[0].bn1(x_ref)
        x_ref = self.modulation.layer2[0].relu(x_ref)
        x_ref = torch.sigmoid(x_ref)
        x = x * x_ref
        x = self.feature_extractor.layer2[0].conv2(x)
        x = self.feature_extractor.layer2[0].bn2(x)
        x_skip_1 = self.feature_extractor.layer2[0].downsample(x_skip_1)
        x = x + x_skip_1
        x_ref = self.modulation.layer2[0].conv2(x_ref)
        x_ref = self.modulation.layer2[0].bn2(x_ref)
        x_skip_ref_1 = self.modulation.layer2[0].downsample(x_skip_ref_1)
        x_ref = x_ref + x_skip_ref_1
        x = self.feature_extractor.layer2[1](x)
        x_ref = self.modulation.layer2[1](x_ref)

        # layer 3
        x_skip_2 = x
        x_skip_ref_2 = x_ref
        x = self.feature_extractor.layer3[0].conv1(x)
        x = self.feature_extractor.layer3[0].bn1(x)
        x = self.feature_extractor.layer3[0].relu(x)
        ref_w_3 = self._translate_weight(
            self.feature_extractor.layer3[0].conv1.weight.data,
            self.modulation.layer3[0].conv1.weight.data,
            self.modulation_fc_3
        )
        x_ref = self.modulation.layer3[0].conv1._conv_forward(x_ref, ref_w_3, None)
        x_ref = self.modulation.layer3[0].bn1(x_ref)
        x_ref = self.modulation.layer3[0].relu(x_ref)
        x_ref = torch.sigmoid(x_ref)
        x = x * x_ref
        x = self.feature_extractor.layer3[0].conv2(x)
        x = self.feature_extractor.layer3[0].bn2(x)
        x_skip_2 = self.feature_extractor.layer3[0].downsample(x_skip_2)
        x = x + x_skip_2
        x_ref = self.modulation.layer3[0].conv2(x_ref)
        x_ref = self.modulation.layer3[0].bn2(x_ref)
        x_skip_ref_2 = self.modulation.layer3[0].downsample(x_skip_ref_2)
        x_ref = x_ref + x_skip_ref_2
        x = self.feature_extractor.layer3[1](x)
        x_ref = self.modulation.layer3[1](x_ref)

        # layer 4
        x_skip_3 = x
        x_skip_ref_3 = x_ref
        x = self.feature_extractor.layer4[0].conv1(x)
        x = self.feature_extractor.layer4[0].bn1(x)
        x = self.feature_extractor.layer4[0].relu(x)
        ref_w_4 = self._translate_weight(
            self.feature_extractor.layer4[0].conv1.weight.data,
            self.modulation.layer4[0].conv1.weight.data,
            self.modulation_fc_4
        )
        x_ref = self.modulation.layer4[0].conv1._conv_forward(x_ref, ref_w_4, None)
        x_ref = self.modulation.layer4[0].bn1(x_ref)
        x_ref = self.modulation.layer4[0].relu(x_ref)
        x_ref = torch.sigmoid(x_ref)
        x = x * x_ref
        x = self.feature_extractor.layer4[0].conv2(x)
        x = self.feature_extractor.layer4[0].bn2(x)
        x_skip_3 = self.feature_extractor.layer4[0].downsample(x_skip_3)
        x = x + x_skip_3
        x_ref = self.modulation.layer4[0].conv2(x_ref)
        x_ref = self.modulation.layer4[0].bn2(x_ref)
        x_skip_ref_3 = self.modulation.layer4[0].downsample(x_skip_ref_3)
        x_ref = x_ref + x_skip_ref_3
        x = self.feature_extractor.layer4[1](x)
        x_ref = self.modulation.layer4[1](x_ref)

        x = self.feature_extractor.layer4[1](x)
        x_skip_ref_4 = x_ref
        x_ref = self.modulation.layer4[1].conv1(x_ref)
        x_ref = self.modulation.layer4[1].bn1(x_ref)
        x_ref = self.modulation.layer4[1].relu(x_ref)
        ref_w_5 = self._translate_weight(
            self.feature_extractor.layer4[1].conv2.weight.data,
            self.modulation.layer4[1].conv2.weight.data,
            self.modulation_fc_5
        )
        x_ref = self.modulation.layer4[1].conv2._conv_forward(x_ref, ref_w_5, None)
        x_ref = self.modulation.layer4[1].bn2(x_ref)
        x_ref = x_ref + x_skip_ref_4
        x = self.feature_extractor.avgpool(x)
        x_ref = self.modulation.avgpool(x_ref)
        x = x * torch.sigmoid(x_ref)

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def get_model(model_str, num_classes):
    if model_str == 'sfm':
        return SelfModulation('sfm', num_classes)
    elif model_str == 'scm':
        return SelfModulation('scm', num_classes)
    elif model_str == 'bgm':
        return BGM(num_classes)
    else:
        raise ValueError('Unknown model: {}'.format(model_str))
