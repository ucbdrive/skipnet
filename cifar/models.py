""" This file contains the model definitions for both original ResNet (6n+2
layers) and SkipNets.
"""

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


########################################
# Original ResNet                      #
########################################


class ResNet(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# For CIFAR-10
# ResNet-38
def cifar10_resnet_38(pretrained=False, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], **kwargs)
    return model


# ResNet-74
def cifar10_resnet_74(pretrained=False, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], **kwargs)
    return model


# ResNet-110
def cifar10_resnet_110(pretrained=False, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


# ResNet-152
def cifar10_resnet_152(pretrained=False, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], **kwargs)
    return model


# For CIFAR-100
# ResNet-38
def cifar100_resnet_38(pretrained=False, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], num_classes=100)
    return model


# ResNet-74
def cifar100_resnet_74(pretrained=False, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], num_classes=100)
    return model


# ResNet-110
def cifar100_resnet_110(pretrained=False, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], num_classes=100)
    return model


# ResNet-152
def cifar100_resnet_152(pretrained=False, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], num_classes=100)
    return model


########################################
# SkipNet+SP with Feedforward Gate     #
########################################


# Feedforward-Gate (FFGate-I)
class FeedforwardGateI(nn.Module):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2"""
    def __init__(self, pool_size=5, channel=10):
        super(FeedforwardGateI, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = conv3x3(channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        # adding another conv layer
        self.conv2 = conv3x3(channel, channel, stride=2)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2)  # for max pooling
        pool_size = math.floor(pool_size/2 + 0.5)  # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)

        # discretize output in forward pass.
        # use softmax gradients in backward pass
        x = (softmax[:, 1] > 0.5).float().detach() - \
            softmax[:, 1].detach() + softmax[:, 1]

        x = x.view(x.size(0), 1, 1, 1)
        return x, logprob


# soft gate v3 (matching FFGate-I)
class SoftGateI(nn.Module):
    """This module has the same structure as FFGate-I.
    In training, adopt continuous gate output. In inference phase,
    use discrete gate outputs"""
    def __init__(self, pool_size=5, channel=10):
        super(SoftGateI, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = conv3x3(channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        # adding another conv layer
        self.conv2 = conv3x3(channel, channel, stride=2)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2)  # for max pooling
        pool_size = math.floor(pool_size/2 + 0.5)  # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)

        x = softmax[:, 1].contiguous()
        x = x.view(x.size(0), 1, 1, 1)

        if not self.training:
            x = (x > 0.5).float()
        return x, logprob


# FFGate-II
class FeedforwardGateII(nn.Module):
    """ use single conv (stride=2) layer only"""
    def __init__(self, pool_size=5, channel=10):
        super(FeedforwardGateII, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.conv1 = conv3x3(channel, channel, stride=2)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2 + 0.5) # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)

        # discretize
        x = (softmax[:, 1] > 0.5).float().detach() - \
            softmax[:, 1].detach() + softmax[:, 1]

        x = x.view(x.size(0), 1, 1, 1)
        return x, logprob


class SoftGateII(nn.Module):
    """ Soft gating version of FFGate-II"""
    def __init__(self, pool_size=5, channel=10):
        super(SoftGateII, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.conv1 = conv3x3(channel, channel, stride=2)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size / 2 + 0.5)  # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)

        x = softmax[:, 1].contiguous()
        x = x.view(x.size(0), 1, 1, 1)
        if not self.training:
            x = (x > 0.5).float()
        return x, logprob


class ResNetFeedForwardSP(nn.Module):
    """ SkipNets with Feed-forward Gates for Supervised Pre-training stage.
    Adding one routing module after each basic block."""

    def __init__(self, block, layers, num_classes=10,
                 gate_type='fisher', **kwargs):
        self.inplanes = 16
        super(ResNetFeedForwardSP, self).__init__()

        self.num_layers = layers
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # going to have 3 groups of layers. For the easiness of skipping,
        # We are going to break the sequential of layers into a list of layers.

        self.gate_type = gate_type
        self._make_group(block, 16, layers[0], group_id=1,
                         gate_type=gate_type, pool_size=32)
        self._make_group(block, 32, layers[1], group_id=2,
                         gate_type=gate_type, pool_size=16)
        self._make_group(block, 64, layers[2], group_id=3,
                         gate_type=gate_type, pool_size=8)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_group(self, block, planes, layers, group_id=1,
                    gate_type='fisher', pool_size=16):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       gate_type=gate_type,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, block, planes, stride=1,
                       gate_type='fisher', pool_size=16):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),

            )
        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        if gate_type == 'ffgate1':
            gate_layer = FeedforwardGateI(pool_size=pool_size,
                                          channel=planes*block.expansion)
        elif gate_type == 'ffgate2':
            gate_layer = FeedforwardGateII(pool_size=pool_size,
                                           channel=planes*block.expansion)
        elif gate_type == 'softgate1':
            gate_layer = SoftGateI(pool_size=pool_size,
                                   channel=planes*block.expansion)
        elif gate_type == 'softgate2':
            gate_layer = SoftGateII(pool_size=pool_size,
                                    channel=planes*block.expansion)
        else:
            gate_layer = None

        if downsample:
            return downsample, layer, gate_layer
        else:
            return None, layer, gate_layer

    def forward(self, x):
        """Return output logits, masks(gate ouputs) and probabilities
        associated to each gate."""

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        masks = []
        gprobs = []
        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer

        mask, gprob = getattr(self, 'group1_gate0')(x)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x  # input of next layer

        for g in range(3):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                prev = x = mask.expand_as(x) * x \
                           + (1 - mask).expand_as(prev) * prev
                mask, gprob = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                gprobs.append(gprob)
                masks.append(mask.squeeze())

        del masks[-1]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, masks, gprobs


# FeeforwardGate-I
# For CIFAR-10
def cifar10_feedforward_38(pretrained=False, **kwargs):
    """SkipNet-38 with FFGate-I"""
    model = ResNetFeedForwardSP(BasicBlock, [6, 6, 6], gate_type='ffgate1')
    return model


def cifar10_feedforward_74(pretrained=False, **kwargs):
    """SkipNet-74 with FFGate-I"""
    model = ResNetFeedForwardSP(BasicBlock, [12, 12, 12], gate_type='ffgate1')
    return model


def cifar10_feedforward_110(pretrained=False, **kwargs):
    """SkipNet-110 with FFGate-II"""
    model = ResNetFeedForwardSP(BasicBlock, [18, 18, 18], gate_type='ffgate2')
    return model


# For CIFAR-100
def cifar100_feeforward_38(pretrained=False, **kwargs):
    """SkipNet-38 with FFGate-I"""
    model = ResNetFeedForwardSP(BasicBlock, [6, 6, 6], num_classes=100,
                                gate_type='ffgate1')
    return model


def cifar100_feedforward_74(pretrained=False, **kwargs):
    """SkipNet-74 with FFGate-I"""
    model = ResNetFeedForwardSP(BasicBlock, [12, 12, 12], num_classes=100,
                                gate_type='ffgate1')
    return model


def cifar100_feedforward_110(pretrained=False, **kwargs):
    """SkipNet-110 with FFGate-II"""
    model = ResNetFeedForwardSP(BasicBlock, [18, 18, 18], num_classes=100,
                                gate_type='ffgate2')
    return model


########################################
# SkipNet+SP with Recurrent Gate       #
########################################


# For Recurrent Gate
def repackage_hidden(h):
    """ to reduce memory usage"""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNGate(nn.Module):
    """Recurrent Gate definition.
    Input is already passed through average pooling and embedding."""
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm'):
        super(RNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        # reduce dim
        self.proj = nn.Linear(hidden_dim, 1)
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        # Take the convolution output of each step
        batch_size = x.size(0)
        self.rnn.flatten_parameters()
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)

        proj = self.proj(out.squeeze())
        prob = self.prob(proj)

        disc_prob = (prob > 0.5).float().detach() - \
                    prob.detach() + prob

        disc_prob = disc_prob.view(batch_size, 1, 1, 1)
        return disc_prob, prob


class SoftRNNGate(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm'):
        super(SoftRNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        # reduce dim
        self.proj = nn.Linear(hidden_dim, 1)
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        # Take the convolution output of each step
        batch_size = x.size(0)
        self.rnn.flatten_parameters()
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)

        proj = self.proj(out.squeeze())
        prob = self.prob(proj)

        x = prob.view(batch_size, 1, 1, 1)
        if not self.training:
            x = (x > 0.5).float()
        return x, prob


class ResNetRecurrentGateSP(nn.Module):
    """SkipNet with Recurrent Gate Model"""
    def __init__(self, block, layers, num_classes=10, embed_dim=10,
                 hidden_dim=10, gate_type='rnn'):
        self.inplanes = 16
        super(ResNetRecurrentGateSP, self).__init__()

        self.num_layers = layers
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self._make_group(block, 16, layers[0], group_id=1, pool_size=32)
        self._make_group(block, 32, layers[1], group_id=2, pool_size=16)
        self._make_group(block, 64, layers[2], group_id=3, pool_size=8)

        # define recurrent gating module
        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_group(self, block, planes, layers, group_id=1, pool_size=16):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, block, planes, stride=1, pool_size=16):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),

            )
        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        gate_layer = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Conv2d(in_channels=planes * block.expansion,
                      out_channels=self.embed_dim,
                      kernel_size=1,
                      stride=1))
        if downsample:
            return downsample, layer, gate_layer
        else:
            return None, layer, gate_layer

    def forward(self, x):

        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # reinitialize hidden units
        self.control.hidden = self.control.init_hidden(batch_size)

        masks = []
        gprobs = []
        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer

        gate_feature = getattr(self, 'group1_gate0')(x)
        mask, gprob = self.control(gate_feature)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x  # input of next layer

        for g in range(3):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                # new mask is taking the current output
                prev = x = mask.expand_as(x) * x \
                           + (1 - mask).expand_as(prev) * prev
                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                mask, grob = self.control(gate_feature)
                gprobs.append(gprob)
                masks.append(mask.squeeze())

        # last block doesn't have gate module
        del masks[-1]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, masks, gprobs


# For CIFAR-10
def cifar10_rnn_gate_38(pretrained=False, **kwargs):
    """SkipNet-38 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [6, 6, 6], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar10_rnn_gate_74(pretrained=False, **kwargs):
    """SkipNet-74 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [12, 12, 12], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar10_rnn_gate_110(pretrained=False,  **kwargs):
    """SkipNet-110 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [18, 18, 18], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar10_rnn_gate_152(pretrained=False,  **kwargs):
    """SkipNet-152 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [25, 25, 25], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


# For CIFAR-100
def cifar100_rnn_gate_38(pretrained=False, **kwargs):
    """SkipNet-38 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [6, 6, 6], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar100_rnn_gate_74(pretrained=False, **kwargs):
    """SkipNet-74 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [12, 12, 12], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar100_rnn_gate_110(pretrained=False, **kwargs):
    """SkipNet-110 with Recurrent Gate """
    model = ResNetRecurrentGateSP(BasicBlock, [18, 18, 18], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar100_rnn_gate_152(pretrained=False, **kwargs):
    """SkipNet-152 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [25, 25, 25], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


########################################
# SkipNet+RL with Feedforward Gate     #
########################################

class RLFeedforwardGateI(nn.Module):
    """ FFGate-I with sampling. Use Pytorch 2.0"""
    def __init__(self, pool_size=5, channel=10):
        super(RLFeedforwardGateI, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = conv3x3(channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        # adding another conv layer
        self.conv2 = conv3x3(channel, channel, stride=2)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2)  # for max pooling
        pool_size = math.floor(pool_size/2 + 0.5)  # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()

        # saved actions and rewards
        self.saved_action = []
        self.rewards = []

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)

        if self.training:
            action = softmax.multinomial()
            self.saved_action = action
        else:
            action = (softmax[:, 1] > 0.5).float()
            self.saved_action = action

        action = action.view(action.size(0), 1, 1, 1).float()
        return action, softmax


class RLFeedforwardGateII(nn.Module):
    def __init__(self, pool_size=5, channel=10):
        super(RLFeedforwardGateII, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.conv1 = conv3x3(channel, channel, stride=2)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size/2 + 0.5)  # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()

        # saved actions and rewards
        self.saved_action = None
        self.rewards = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)

        if self.training:
            action = softmax.multinomial()
            self.saved_action = action
        else:
            action = (softmax[:, 1] > 0.5).float()
            self.saved_action = action

        action = action.view(action.size(0), 1, 1, 1).float()
        return action, softmax


class ResNetFeedForwardRL(nn.Module):
    """Adding gating module on every basic block"""

    def __init__(self, block, layers, num_classes=10,
                 gate_type='ffgate1', **kwargs):
        self.inplanes = 16
        super(ResNetFeedForwardRL, self).__init__()

        self.num_layers = layers
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.gate_instances = []
        self.gate_type = gate_type
        self._make_group(block, 16, layers[0], group_id=1,
                         gate_type=gate_type, pool_size=32)
        self._make_group(block, 32, layers[1], group_id=2,
                         gate_type=gate_type, pool_size=16)
        self._make_group(block, 64, layers[2], group_id=3,
                         gate_type=gate_type, pool_size=8)

        # remove the last gate instance, (not optimized)
        del self.gate_instances[-1]

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.softmax = nn.Softmax()
        self.saved_actions = []
        self.rewards = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_group(self, block, planes, layers, group_id=1,
                    gate_type='fisher', pool_size=16):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       gate_type=gate_type,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

            # add into gate instance collection
            self.gate_instances.append(meta[2])

    def _make_layer_v2(self, block, planes, stride=1,
                       gate_type='fisher', pool_size=16):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),

            )
        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        if gate_type == 'ffgate1':
            gate_layer = RLFeedforwardGateI(pool_size=pool_size,
                                            channel=planes*block.expansion)
        elif gate_type == 'ffgate2':
            gate_layer = RLFeedforwardGateII(pool_size=pool_size,
                                             channel=planes*block.expansion)
        else:
            gate_layer = None

        if downsample:
            return downsample, layer, gate_layer
        else:
            return None, layer, gate_layer

    def repackage_vars(self):
        self.saved_actions = repackage_hidden(self.saved_actions)

    def forward(self, x, reinforce=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        masks = []
        gprobs = []
        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer
        mask, gprob = getattr(self, 'group1_gate0')(x)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x  # input of next layer

        for g in range(3):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                # new mask is taking the current output
                prev = x = mask.expand_as(x) * x \
                           + (1 - mask).expand_as(prev) * prev
                mask, gprob = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                gprobs.append(gprob)
                masks.append(mask.squeeze())

        del masks[-1]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # collect all actions
        for inst in self.gate_instances:
            self.saved_actions.append(inst.saved_action)

        if reinforce:  # for pure RL
            softmax = self.softmax(x)
            action = softmax.multinomial()
            self.saved_actions.append(action)

        return x, masks, gprobs


# FFGate-I
# For CIFAR-10
def cifar10_feedfoward_rl_38(pretrained=False, **kwargs):
    """SkipNet-38 + RL with FFGate-I"""
    model = ResNetFeedForwardRL(BasicBlock, [6, 6, 6],
                                num_classes=10, gate_type='ffgate1')
    return model


def cifar10_feedforward_rl_74(pretrained=False, **kwargs):
    """SkipNet-74 + RL with FFGate-I"""
    model = ResNetFeedForwardRL(BasicBlock, [12, 12, 12],
                                num_classes=10, gate_type='ffgate1')
    return model


def cifar10_feedforward_rl_110(pretrained=False, **kwargs):
    """SkipNet-110 + RL with FFGate-II"""
    model = ResNetFeedForwardRL(BasicBlock, [18, 18, 18],
                                num_classes=10, gate_type='ffgate2')
    return model


# For CIFAR-100
def cifar100_feedford_rl_38(pretrained=False, **kwargs):
    """SkipNet-38 + RL with FFGate-I"""
    model = ResNetFeedForwardRL(BasicBlock, [6, 6, 6],
                                num_classes=100, gate_type='ffgate1')
    return model


def cifar100_feedforward_rl_74(pretrained=False, **kwargs):
    """SkipNet-74 + RL with FFGate-I"""
    model = ResNetFeedForwardRL(BasicBlock, [12, 12, 12],
                                num_classes=100, gate_type='ffgate1')
    return model


def cifar100_feedforward_rl_110(pretrained=False, **kwargs):
    """SkipNet-110 + RL with FFGate-II"""
    model = ResNetFeedForwardRL(BasicBlock, [18, 18, 18],
                                num_classes=100, gate_type='ffgate2')
    return model


########################################
# SkipNet+RL with Recurrent Gate       #
########################################

class RNNGatePolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm'):
        super(RNNGatePolicy, self).__init__()

        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        # reduce dim. use softmax here for two actions.
        self.proj = nn.Linear(hidden_dim, 1)
        self.prob = nn.Sigmoid()

        # saved actions and rewards
        self.saved_actions = []
        self.rewards = []

    def hotter(self, t):
        self.proj.weight.data /= t
        self.proj.bias.data /= t

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        batch_size = x.size(0)
        self.rnn.flatten_parameters()
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)

        # do action selection in the forward pass
        if self.training:
            proj = self.proj(out.squeeze())
            prob = self.prob(proj)
            bi_prob = torch.cat([1 - prob, prob], dim=1)
            action = bi_prob.multinomial()
            self.saved_actions.append(action)
        else:
            proj = self.proj(out.squeeze())
            prob = self.prob(proj)
            bi_prob = torch.cat([1 - prob, prob], dim=1)
            action = (prob > 0.5).float()
            self.saved_actions.append(action)
        action = action.view(action.size(0), 1, 1, 1).float()
        return action, bi_prob


class ResNetRecurrentGateRL(nn.Module):
    """Adding gating module on every basic block"""

    def __init__(self, block, layers, num_classes=10,
                 embed_dim=64, hidden_dim=64):
        self.inplanes = 16
        super(ResNetRecurrentGateRL, self).__init__()

        self.num_layers = layers
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self._make_group(block, 16, layers[0], group_id=1, pool_size=32)
        self._make_group(block, 32, layers[1], group_id=2, pool_size=16)
        self._make_group(block, 64, layers[2], group_id=3, pool_size=8)

        self.control = RNNGatePolicy(embed_dim, hidden_dim)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.softmax = nn.Softmax()

        self.saved_actions = []
        self.rewards = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def _make_group(self, block, planes, layers, group_id=1, pool_size=16):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])

    def _make_layer_v2(self, block, planes, stride=1, pool_size=16):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),

            )
        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        gate_layer = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Conv2d(in_channels=planes * block.expansion,
                      out_channels=self.embed_dim,
                      kernel_size=1,
                      stride=1))

        return downsample, layer, gate_layer

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # reinitialize hidden units
        self.control.hidden = self.control.init_hidden(batch_size)

        masks = []
        gprobs = []
        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer
        gate_feature = getattr(self, 'group1_gate0')(x)

        mask, gprob = self.control(gate_feature)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x

        for g in range(3):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                prev = x = mask.expand_as(x) * x + \
                           (1 - mask).expand_as(prev)*prev
                if not (g == 2 and (i == self.num_layers[g] -1)):
                    gate_feature = getattr(self,
                                'group{}_gate{}'.format(g+1, i))(x)
                    mask, gprob = self.control(gate_feature)
                    gprobs.append(gprob)
                    masks.append(mask.squeeze())

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.training:
            x = self.fc(x)
            softmax = self.softmax(x)
            pred = softmax.multinomial()
        else:
            x = self.fc(x)
            pred = x.max(1)[1]
        self.saved_actions.append(pred)

        return x, masks, gprobs


# for CIFAR-10
def cifar10_rnn_gate_rl_38(pretrained=False, **kwargs):
    """SkipNet-38 + RL with Recurrent Gate"""
    model = ResNetRecurrentGateRL(BasicBlock, [6, 6, 6], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar10_rnn_gate_rl_74(pretrained=False, **kwargs):
    """SkipNet-74 + RL with Recurrent Gate"""
    model = ResNetRecurrentGateRL(BasicBlock, [12, 12, 12], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar10_rnn_gate_rl_110(pretrained=False, **kwargs):
    """SkipNet-110 + RL with Recurrent Gate"""
    model = ResNetRecurrentGateRL(BasicBlock, [18, 18, 18], num_classes=10,
                                  embed_dim=10, hidden_dim=10)
    return model


# for CIFAR-100
def cifar100_rnn_gate_rl_38(pretrained=False, **kwargs):
    """SkipNet-38 + RL with Recurrent Gate"""
    model = ResNetRecurrentGateRL(BasicBlock, [6, 6, 6], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar100_rnn_gate_rl_74(pretrained=False, **kwargs):
    """SkipNet-74 + RL with Recurrent Gate"""
    model = ResNetRecurrentGateRL(BasicBlock, [12, 12, 12], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


def cifar100_rnn_gate_rl_110(pretrained=False, **kwargs):
    """SkipNet-110 + RL with Recurrent Gate"""
    model = ResNetRecurrentGateRL(BasicBlock, [18, 18, 18], num_classes=100,
                                  embed_dim=10, hidden_dim=10)
    return model


