import math
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """可逆实例归一化（Reversible Instance Normalization）。
    对每个样本独立归一化，解决不同市场环境（牛市/熊市）的分布漂移问题。
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        # x: [B, C, L]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm


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

        # RevIN: 替代全局z-score，按样本自适应归一化
        self.revin = RevIN(input_channels)

        self.inplanes = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers: 每stage 1个block（精简版，原来是2个）
        self.layer1 = self._make_layer(64, 1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)

        backbone_channels = 512 * BasicBlock1D.expansion
        self.pool_size = math.ceil(output_dim / backbone_channels)
        self.avgpool = nn.AdaptiveAvgPool1d(self.pool_size)

        flat_dim = backbone_channels * self.pool_size
        self.fc = nn.Linear(flat_dim, output_dim) if flat_dim != output_dim else None

        # BatchNorm 替代 Tanh：梯度不饱和，分布可学习
        self.output_norm = nn.BatchNorm1d(output_dim)

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
        x = self.revin(x)

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
        x = self.output_norm(x)

        return x
