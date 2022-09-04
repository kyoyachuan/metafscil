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
        self.modulation_2 = SelfAttention(64) if att_mode == 'sfm' else ChannelAttention(64)
        self.modulation_3 = SelfAttention(128) if att_mode == 'sfm' else ChannelAttention(128)
        self.modulation_4 = SelfAttention(256) if att_mode == 'sfm' else ChannelAttention(256)
        self.modulation_5 = ChannelAttention(512)
        self.classifier = nn.Linear(self.feature_extractor.fc.in_features, num_classes)

        del self.feature_extractor.fc

    def forward(self, x, feat_params=None):
        if feat_params is None:
            feat_params = list(self.feature_extractor.parameters())

        fe = self.feature_extractor

        # pre layer
        x = fe.conv1._conv_forward(x, feat_params[0], None)
        x = F.batch_norm(x, fe.bn1.running_mean, fe.bn1.running_var,
                         feat_params[1], feat_params[2], fe.bn1.training or not fe.bn1.track_running_stats)
        x = fe.relu(x)
        x = self.modulation_1(x)

        # layer 1
        x = fe.maxpool(x)
        x_skip_0_1 = x
        x = fe.layer1[0].conv1._conv_forward(x, feat_params[3], None)
        x = F.batch_norm(
            x, fe.layer1[0].bn1.running_mean, fe.layer1[0].bn1.running_var, feat_params[4],
            feat_params[5],
            fe.layer1[0].bn1.training or not fe.layer1[0].bn1.track_running_stats)
        x = fe.layer1[0].relu(x)
        x = fe.layer1[0].conv2._conv_forward(x, feat_params[6], None)
        x = F.batch_norm(
            x, fe.layer1[0].bn2.running_mean, fe.layer1[0].bn2.running_var, feat_params[7],
            feat_params[8],
            fe.layer1[0].bn2.training or not fe.layer1[0].bn2.track_running_stats)
        x = x + x_skip_0_1

        x_skip_0_2 = x
        x = fe.layer1[1].conv1._conv_forward(x, feat_params[9], None)
        x = F.batch_norm(
            x, fe.layer1[1].bn1.running_mean, fe.layer1[1].bn1.running_var, feat_params[10],
            feat_params[11],
            fe.layer1[1].bn1.training or not fe.layer1[1].bn1.track_running_stats)
        x = fe.layer1[1].relu(x)
        x = fe.layer1[1].conv2._conv_forward(x, feat_params[12], None)
        x = F.batch_norm(
            x, fe.layer1[1].bn2.running_mean, fe.layer1[1].bn2.running_var, feat_params[13],
            feat_params[14],
            fe.layer1[1].bn2.training or not fe.layer1[1].bn2.track_running_stats)
        x = x + x_skip_0_2
        x = self.modulation_2(x)

        # layer 2
        x_skip_1_1 = x
        x = fe.layer2[0].conv1._conv_forward(x, feat_params[15], None)
        x = F.batch_norm(
            x, fe.layer2[0].bn1.running_mean, fe.layer2[0].bn1.running_var, feat_params[16],
            feat_params[17],
            fe.layer2[0].bn1.training or not fe.layer2[0].bn1.track_running_stats)
        x = fe.layer2[0].relu(x)
        x = fe.layer2[0].conv2._conv_forward(x, feat_params[18], None)
        x = F.batch_norm(
            x, fe.layer2[0].bn2.running_mean, fe.layer2[0].bn2.running_var, feat_params[19],
            feat_params[20],
            fe.layer2[0].bn2.training or not fe.layer2[0].bn2.track_running_stats)
        x_skip_1_1 = fe.layer2[0].downsample[0]._conv_forward(x_skip_1_1, feat_params[21], None)
        x_skip_1_1 = F.batch_norm(
            x_skip_1_1, fe.layer2[0].downsample[1].running_mean, fe.layer2[0].downsample[1].running_var,
            feat_params[22],
            feat_params[23],
            fe.layer2[0].downsample[1].training or not fe.layer2[0].downsample[1].track_running_stats)
        x = x + x_skip_1_1

        x_skip_1_2 = x
        x = fe.layer2[1].conv1._conv_forward(x, feat_params[24], None)
        x = F.batch_norm(
            x, fe.layer2[1].bn1.running_mean, fe.layer2[1].bn1.running_var, feat_params[25],
            feat_params[26],
            fe.layer2[1].bn1.training or not fe.layer2[1].bn1.track_running_stats)
        x = fe.layer2[1].relu(x)
        x = fe.layer2[1].conv2._conv_forward(x, feat_params[27], None)
        x = F.batch_norm(
            x, fe.layer2[1].bn2.running_mean, fe.layer2[1].bn2.running_var, feat_params[28],
            feat_params[29],
            fe.layer2[1].bn2.training or not fe.layer2[1].bn2.track_running_stats)
        x = x + x_skip_1_2
        x = self.modulation_3(x)

        # layer 3
        x_skip_2_1 = x
        x = fe.layer3[0].conv1._conv_forward(x, feat_params[30], None)
        x = F.batch_norm(
            x, fe.layer3[0].bn1.running_mean, fe.layer3[0].bn1.running_var, feat_params[31],
            feat_params[32],
            fe.layer3[0].bn1.training or not fe.layer3[0].bn1.track_running_stats)
        x = fe.layer3[0].relu(x)
        x = fe.layer3[0].conv2._conv_forward(x, feat_params[33], None)
        x = F.batch_norm(
            x, fe.layer3[0].bn2.running_mean, fe.layer3[0].bn2.running_var, feat_params[34],
            feat_params[35],
            fe.layer3[0].bn2.training or not fe.layer3[0].bn2.track_running_stats)
        x_skip_2_1 = fe.layer3[0].downsample[0]._conv_forward(x_skip_2_1, feat_params[36], None)
        x_skip_2_1 = F.batch_norm(
            x_skip_2_1, fe.layer3[0].downsample[1].running_mean, fe.layer3[0].downsample[1].running_var,
            feat_params[37],
            feat_params[38],
            fe.layer3[0].downsample[1].training or not fe.layer3[0].downsample[1].track_running_stats)
        x = x + x_skip_2_1

        x_skip_2_2 = x
        x = fe.layer3[1].conv1._conv_forward(x, feat_params[39], None)
        x = F.batch_norm(
            x, fe.layer3[1].bn1.running_mean, fe.layer3[1].bn1.running_var, feat_params[40],
            feat_params[41],
            fe.layer3[1].bn1.training or not fe.layer3[1].bn1.track_running_stats)
        x = fe.layer3[1].relu(x)
        x = fe.layer3[1].conv2._conv_forward(x, feat_params[42], None)
        x = F.batch_norm(
            x, fe.layer3[1].bn2.running_mean, fe.layer3[1].bn2.running_var, feat_params[43],
            feat_params[44],
            fe.layer3[1].bn2.training or not fe.layer3[1].bn2.track_running_stats)
        x = x + x_skip_2_2
        x = self.modulation_4(x)

        # layer 4
        x_skip_3_1 = x
        x = fe.layer4[0].conv1._conv_forward(x, feat_params[45], None)
        x = F.batch_norm(
            x, fe.layer4[0].bn1.running_mean, fe.layer4[0].bn1.running_var, feat_params[46],
            feat_params[47],
            fe.layer4[0].bn1.training or not fe.layer4[0].bn1.track_running_stats)
        x = fe.layer4[0].relu(x)
        x = fe.layer4[0].conv2._conv_forward(x, feat_params[48], None)
        x = F.batch_norm(
            x, fe.layer4[0].bn2.running_mean, fe.layer4[0].bn2.running_var, feat_params[49],
            feat_params[50],
            fe.layer4[0].bn2.training or not fe.layer4[0].bn2.track_running_stats)
        x_skip_3_1 = fe.layer4[0].downsample[0]._conv_forward(x_skip_3_1, feat_params[51], None)
        x_skip_3_1 = F.batch_norm(
            x_skip_3_1, fe.layer4[0].downsample[1].running_mean, fe.layer4[0].downsample[1].running_var,
            feat_params[52],
            feat_params[53],
            fe.layer4[0].downsample[1].training or not fe.layer4[0].downsample[1].track_running_stats)
        x = x + x_skip_3_1

        x_skip_3_2 = x
        x = fe.layer4[1].conv1._conv_forward(x, feat_params[54], None)
        x = F.batch_norm(
            x, fe.layer4[1].bn1.running_mean, fe.layer4[1].bn1.running_var, feat_params[55],
            feat_params[56],
            fe.layer4[1].bn1.training or not fe.layer4[1].bn1.track_running_stats)
        x = fe.layer4[1].relu(x)
        x = fe.layer4[1].conv2._conv_forward(x, feat_params[57], None)
        x = F.batch_norm(
            x, fe.layer4[1].bn2.running_mean, fe.layer4[1].bn2.running_var, feat_params[58],
            feat_params[59],
            fe.layer4[1].bn2.training or not fe.layer4[1].bn2.track_running_stats)
        x = x + x_skip_3_2

        # output
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

    def forward(self, input, feat_params=None):
        if feat_params is None:
            feat_params = list(self.feature_extractor.parameters())

        fe = self.feature_extractor

        # pre layer
        x = fe.conv1._conv_forward(input, feat_params[0], None)
        x = F.batch_norm(
            x, fe.bn1.running_mean, fe.bn1.running_var, feat_params[1],
            feat_params[2],
            fe.bn1.training or not fe.bn1.track_running_stats)
        x = fe.relu(x)
        ref_w_1 = self._translate_weight(feat_params[0].data,
                                         self.modulation.conv1.weight.data, self.modulation_fc_1)
        x_ref = self.modulation.conv1._conv_forward(input, ref_w_1, None)
        x_ref = self.modulation.bn1(x_ref)
        x_ref = torch.sigmoid(x_ref)
        x = x * x_ref

        # layer 1
        x_ref = self.modulation.maxpool(x_ref)
        x_ref = self.modulation.layer1(x_ref)
        x = fe.maxpool(x)

        x_skip_0_1 = x
        x = fe.layer1[0].conv1._conv_forward(x, feat_params[3], None)
        x = F.batch_norm(
            x, fe.layer1[0].bn1.running_mean, fe.layer1[0].bn1.running_var, feat_params[4],
            feat_params[5],
            fe.layer1[0].bn1.training or not fe.layer1[0].bn1.track_running_stats)
        x = fe.layer1[0].relu(x)
        x = fe.layer1[0].conv2._conv_forward(x, feat_params[6], None)
        x = F.batch_norm(
            x, fe.layer1[0].bn2.running_mean, fe.layer1[0].bn2.running_var, feat_params[7],
            feat_params[8],
            fe.layer1[0].bn2.training or not fe.layer1[0].bn2.track_running_stats)
        x = x + x_skip_0_1

        x_skip_0_2 = x
        x = fe.layer1[1].conv1._conv_forward(x, feat_params[9], None)
        x = F.batch_norm(
            x, fe.layer1[1].bn1.running_mean, fe.layer1[1].bn1.running_var, feat_params[10],
            feat_params[11],
            fe.layer1[1].bn1.training or not fe.layer1[1].bn1.track_running_stats)
        x = fe.layer1[1].relu(x)
        x = fe.layer1[1].conv2._conv_forward(x, feat_params[12], None)
        x = F.batch_norm(
            x, fe.layer1[1].bn2.running_mean, fe.layer1[1].bn2.running_var, feat_params[13],
            feat_params[14],
            fe.layer1[1].bn2.training or not fe.layer1[1].bn2.track_running_stats)
        x = x + x_skip_0_2

        # layer 2
        x_skip_1_1 = x
        x_skip_ref_1 = x_ref
        x = fe.layer2[0].conv1._conv_forward(x, feat_params[15], None)
        x = F.batch_norm(
            x, fe.layer2[0].bn1.running_mean, fe.layer2[0].bn1.running_var, feat_params[16],
            feat_params[17],
            fe.layer2[0].bn1.training or not fe.layer2[0].bn1.track_running_stats)
        x = fe.layer2[0].relu(x)
        ref_w_2 = self._translate_weight(
            feat_params[15].data,
            self.modulation.layer2[0].conv1.weight.data,
            self.modulation_fc_2
        )
        x_ref = self.modulation.layer2[0].conv1._conv_forward(x_ref, ref_w_2, None)
        x_ref = self.modulation.layer2[0].bn1(x_ref)
        x_ref = torch.sigmoid(x_ref)
        x = x * x_ref
        x_ref = self.modulation.layer2[0].conv2(x_ref)
        x_ref = self.modulation.layer2[0].bn2(x_ref)
        x_skip_ref_1 = self.modulation.layer2[0].downsample(x_skip_ref_1)
        x_ref = x_ref + x_skip_ref_1
        x = fe.layer2[0].conv2._conv_forward(x, feat_params[18], None)
        x = F.batch_norm(
            x, fe.layer2[0].bn2.running_mean, fe.layer2[0].bn2.running_var, feat_params[19],
            feat_params[20],
            fe.layer2[0].bn2.training or not fe.layer2[0].bn2.track_running_stats)
        x_skip_1_1 = fe.layer2[0].downsample[0]._conv_forward(x_skip_1_1, feat_params[21], None)
        x_skip_1_1 = F.batch_norm(
            x_skip_1_1, fe.layer2[0].downsample[1].running_mean,
            fe.layer2[0].downsample[1].running_var, feat_params[22], feat_params[23],
            fe.layer2[0].downsample[1].training or not fe.layer2[0].downsample[1].track_running_stats)
        x = x + x_skip_1_1

        x_skip_2_1 = x
        x = fe.layer2[1].conv1._conv_forward(x, feat_params[24], None)
        x = F.batch_norm(
            x, fe.layer2[1].bn1.running_mean, fe.layer2[1].bn1.running_var, feat_params[25],
            feat_params[26],
            fe.layer2[1].bn1.training or not fe.layer2[1].bn1.track_running_stats)
        x = fe.layer2[1].relu(x)
        x = fe.layer2[1].conv2._conv_forward(x, feat_params[27], None)
        x = F.batch_norm(
            x, fe.layer2[1].bn2.running_mean, fe.layer2[1].bn2.running_var, feat_params[28],
            feat_params[29],
            fe.layer2[1].bn2.training or not fe.layer2[1].bn2.track_running_stats)
        x = x + x_skip_2_1
        x_ref = self.modulation.layer2[1](x_ref)

        # layer 3
        x_skip_2_1 = x
        x_skip_ref_2 = x_ref
        x = fe.layer3[0].conv1._conv_forward(x, feat_params[30], None)
        x = F.batch_norm(
            x, fe.layer3[0].bn1.running_mean, fe.layer3[0].bn1.running_var, feat_params[31],
            feat_params[32],
            fe.layer3[0].bn1.training or not fe.layer3[0].bn1.track_running_stats)
        x = fe.layer3[0].relu(x)
        ref_w_3 = self._translate_weight(
            feat_params[30].data,
            self.modulation.layer3[0].conv1.weight.data,
            self.modulation_fc_3
        )
        x_ref = self.modulation.layer3[0].conv1._conv_forward(x_ref, ref_w_3, None)
        x_ref = self.modulation.layer3[0].bn1(x_ref)
        x_ref = torch.sigmoid(x_ref)
        x = x * x_ref
        x_ref = self.modulation.layer3[0].conv2(x_ref)
        x_ref = self.modulation.layer3[0].bn2(x_ref)
        x_skip_ref_2 = self.modulation.layer3[0].downsample(x_skip_ref_2)
        x_ref = x_ref + x_skip_ref_2
        x = fe.layer3[0].conv2._conv_forward(x, feat_params[33], None)
        x = F.batch_norm(
            x, fe.layer3[0].bn2.running_mean, fe.layer3[0].bn2.running_var, feat_params[34],
            feat_params[35],
            fe.layer3[0].bn2.training or not fe.layer3[0].bn2.track_running_stats)
        x_skip_2_1 = fe.layer3[0].downsample[0]._conv_forward(x_skip_2_1, feat_params[36], None)
        x_skip_2_1 = F.batch_norm(
            x_skip_2_1, fe.layer3[0].downsample[1].running_mean,
            fe.layer3[0].downsample[1].running_var, feat_params[37], feat_params[38],
            fe.layer3[0].downsample[1].training or not fe.layer3[0].downsample[1].track_running_stats)
        x = x + x_skip_2_1

        x_skip_3_1 = x
        x = fe.layer3[1].conv1._conv_forward(x, feat_params[39], None)
        x = F.batch_norm(
            x, fe.layer3[1].bn1.running_mean, fe.layer3[1].bn1.running_var, feat_params[40],
            feat_params[41],
            fe.layer3[1].bn1.training or not fe.layer3[1].bn1.track_running_stats)
        x = fe.layer3[1].relu(x)
        x = fe.layer3[1].conv2._conv_forward(x, feat_params[42], None)
        x = F.batch_norm(
            x, fe.layer3[1].bn2.running_mean, fe.layer3[1].bn2.running_var, feat_params[43],
            feat_params[44],
            fe.layer3[1].bn2.training or not fe.layer3[1].bn2.track_running_stats)
        x = x + x_skip_3_1
        x_ref = self.modulation.layer3[1](x_ref)

        # layer 4
        x_skip_3_1 = x
        x_skip_ref_3 = x_ref
        x = fe.layer4[0].conv1._conv_forward(x, feat_params[45], None)
        x = F.batch_norm(
            x, fe.layer4[0].bn1.running_mean, fe.layer4[0].bn1.running_var, feat_params[46],
            feat_params[47],
            fe.layer4[0].bn1.training or not fe.layer4[0].bn1.track_running_stats)
        x = fe.layer4[0].relu(x)
        ref_w_4 = self._translate_weight(
            feat_params[45].data,
            self.modulation.layer4[0].conv1.weight.data,
            self.modulation_fc_4
        )
        x_ref = self.modulation.layer4[0].conv1._conv_forward(x_ref, ref_w_4, None)
        x_ref = self.modulation.layer4[0].bn1(x_ref)
        x_ref = torch.sigmoid(x_ref)
        x = x * x_ref
        x_ref = self.modulation.layer4[0].conv2(x_ref)
        x_ref = self.modulation.layer4[0].bn2(x_ref)
        x_skip_ref_3 = self.modulation.layer4[0].downsample(x_skip_ref_3)
        x_ref = x_ref + x_skip_ref_3
        x = fe.layer4[0].conv2._conv_forward(x, feat_params[48], None)
        x = F.batch_norm(
            x, fe.layer4[0].bn2.running_mean, fe.layer4[0].bn2.running_var, feat_params[49],
            feat_params[50],
            fe.layer4[0].bn2.training or not fe.layer4[0].bn2.track_running_stats)
        x_skip_3_1 = fe.layer4[0].downsample[0]._conv_forward(x_skip_3_1, feat_params[51], None)
        x_skip_3_1 = F.batch_norm(
            x_skip_3_1, fe.layer4[0].downsample[1].running_mean,
            fe.layer4[0].downsample[1].running_var, feat_params[52], feat_params[53],
            fe.layer4[0].downsample[1].training or not fe.layer4[0].downsample[1].track_running_stats)
        x = x + x_skip_3_1

        x_skip_4_1 = x
        x = fe.layer4[1].conv1._conv_forward(x, feat_params[54], None)
        x = F.batch_norm(
            x, fe.layer4[1].bn1.running_mean, fe.layer4[1].bn1.running_var, feat_params[55],
            feat_params[56],
            fe.layer4[1].bn1.training or not fe.layer4[1].bn1.track_running_stats)
        x = fe.layer4[1].relu(x)
        x = fe.layer4[1].conv2._conv_forward(x, feat_params[57], None)
        x = F.batch_norm(
            x, fe.layer4[1].bn2.running_mean, fe.layer4[1].bn2.running_var, feat_params[58],
            feat_params[59],
            fe.layer4[1].bn2.training or not fe.layer4[1].bn2.track_running_stats)
        x = x + x_skip_4_1
        x_skip_ref_4 = x_ref
        ref_w_5 = self._translate_weight(
            feat_params[54].data,
            self.modulation.layer4[1].conv2.weight.data,
            self.modulation_fc_5
        )
        x_ref = self.modulation.layer4[1].conv1._conv_forward(x_ref, ref_w_5, None)
        x_ref = self.modulation.layer4[1].bn1(x_ref)
        x_ref = self.modulation.layer4[1].relu(x_ref)
        x_ref = self.modulation.layer4[1].conv2(x_ref)
        x_ref = self.modulation.layer4[1].bn2(x_ref)
        x_ref = x_ref + x_skip_ref_4

        # output
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
