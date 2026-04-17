import math
import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=18, output_dim=1024):
        super(FeatureExtractor, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # 保留空间维度：pool 到 pool_size 而非 1，避免信息瓶颈
        # layer4 输出 [B, 512, 45]，pool 后 [B, 512, pool_size]
        # flatten 后 512 * pool_size 维，每维对应不同通道×不同时间段
        backbone_channels = 512 * BasicBlock1D.expansion
        self.pool_size = math.ceil(output_dim / backbone_channels)
        self.avgpool = nn.AdaptiveAvgPool1d(self.pool_size)

        flat_dim = backbone_channels * self.pool_size
        # 仅在 flat_dim != output_dim 时需要投射层
        self.fc = nn.Linear(flat_dim, output_dim) if flat_dim != output_dim else None

        # Output activation to ensure (-1, 1)
        self.output_act = nn.Tanh()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * BasicBlock1D.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * BasicBlock1D.expansion),
            )

        layers = []
        layers.append(BasicBlock1D(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [Batch * SeqLen, 18, 1424]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)       # [B, 512, pool_size]
        x = torch.flatten(x, 1)   # [B, 512 * pool_size]
        if self.fc is not None:
            x = self.fc(x)
        x = self.output_act(x)

        return x
