import torch
import torch.nn as nn

# 建構block
class ResidualBlock(nn.Module):
  def __init__(self, in_channel, out_channel, strides=1):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channel)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channel)

    self.downsample = None
    if strides != 1 or in_channel != out_channel:
        self.downsample = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides),
        nn.BatchNorm2d(out_channel)
    )

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

#建構Resnet
class ResNet18(nn.Module):
  def __init__(self, ResidualBlock, num_class=100):
    super(ResNet18, self).__init__()

    # 第一層
    self.channel = 64
    self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.MaxPool2d(kernel_size=3, stride=2))
    # 第2~5層
    self.layer1 = self.make_layer(ResidualBlock, 64, 2, 1)
    self.layer2 = self.make_layer(ResidualBlock, 128, 2, 2)
    self.layer3 = self.make_layer(ResidualBlock, 256, 2, 2)
    self.layer4 = self.make_layer(ResidualBlock, 512, 2, 2)
    self.avgpool = nn.AdaptiveAvgPool2d((1,1)) #平均池化層
    self.fcl = nn.Linear(512, num_class) #全連通層

  # 建立每一層
  def make_layer(self, block, outchannel, num_block, stride):
    strides = [stride] + [1] * (num_block - 1) #只有第一層stride會不一定是，剩下都是1
    layer = []

    for i in strides:
      layer.append(block(self.channel, outchannel, i))
      self.channel = outchannel

    return nn.Sequential(*layer)

  def forward(self, x):
    out = self.conv1(x)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)

    out = torch.flatten(out, 1)
    out = self.fcl(out)   

    return out