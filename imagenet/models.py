#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.autograd as autograd
from torch.autograd.variable import Variable
from threading import Lock
from torch.distributions import Categorical

global_lock = Lock()


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ==============================
# Original Model without Gating
# ==============================

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet74(pretrained=False, **kwargs):
    """ ResNet-74"""
    model = ResNet(Bottleneck, [3, 4, 14, 3], **kwargs)
    return model


def resnet101(pretrained=False,  **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False,  **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


# ======================
# Recurrent Gate  Design
# ======================

def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNGate(nn.Module):
    """given the fixed input size, return a single layer lstm """
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
        self.proj = nn.Conv2d(in_channels=hidden_dim, out_channels=1,
                              kernel_size=1, stride=1)
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
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

        out = out.squeeze()
        proj = self.proj(out.view(out.size(0), out.size(1), 1, 1,)).squeeze()
        prob = self.prob(proj)

        disc_prob = (prob > 0.5).float().detach() - prob.detach() + prob
        disc_prob = disc_prob.view(batch_size, 1, 1, 1)
        return disc_prob, prob


# =======================
# Recurrent Gate Model
# =======================
class RecurrentGatedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, embed_dim=10,
                 hidden_dim=10, gate_type='rnn', **kwargs):
        self.inplanes = 64
        super(RecurrentGatedResNet, self).__init__()

        self.num_layers = layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # going to have 4 groups of layers. For the easiness of skipping,
        # We are going to break the sequential of layers into a list of layers.
        self._make_group(block, 64, layers[0], group_id=1, pool_size=56)
        self._make_group(block, 128, layers[1], group_id=2, pool_size=28)
        self._make_group(block, 256, layers[2], group_id=3, pool_size=14)
        self._make_group(block, 512, layers[3], group_id=4, pool_size=7)

        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_group(self, block, planes, layers, group_id=1, pool_size=56):
        """ Create the whole group """
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

    def _make_layer_v2(self, block, planes, stride=1, pool_size=56):
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

        # this is for having the same input dimension to rnn gate.
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

    def repackage_hidden(self):
        self.control.hidden = repackage_hidden(self.control.hidden)

    def forward(self, x):
        """mask_values is for the test random gates"""
        # pdb.set_trace()

        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

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

        for g in range(4):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                prev = x = mask.expand_as(x)*x + (1-mask).expand_as(prev)*prev
                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                mask, gprob = self.control(gate_feature)
                if not (g == 3 and i == (self.num_layers[3]-1)):
                    # not add the last mask to masks
                    gprobs.append(gprob)
                    masks.append(mask.squeeze())

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, masks, gprobs, self.control.hidden


def imagenet_rnn_gate_18(pretrained=False, **kwargs):
    """ Construct SkipNet-18 + SP """
    model = RecurrentGatedResNet(BasicBlock, [2, 2, 2, 2],
                                 embed_dim=10, hidden_dim=10, gate_type='rnn')
    return model


def imagenet_rnn_gate_34(pretrained=False, **kwargs):
    """ Construct SkipNet-34 + SP """
    model = RecurrentGatedResNet(BasicBlock, [3, 4, 6, 3],
                                 embed_dim=10, hidden_dim=10, gate_type='rnn')
    return model


def imagenet_rnn_gate_50(pretrained=False, **kwargs):
    """ Construct SkipNet-50 + SP """
    model = RecurrentGatedResNet(Bottleneck, [3, 4, 6, 3],
                                 embed_dim=10, hidden_dim=10, gate_type='rnn')
    return model


def imagenet_rnn_gate_101(pretrained=False,  **kwargs):
    """ Constructs SkipNet-101 + SP """
    model = RecurrentGatedResNet(Bottleneck, [3, 4, 23, 3],
                                 embed_dim=10, hidden_dim=10, gate_type='rnn')
    return model


def imagenet_rnn_gate_152(pretrained=False,  **kwargs):
    """Constructs SkipNet-152 + SP """
    model = RecurrentGatedResNet(Bottleneck, [3, 8, 36, 3],
                                 embed_dim=10, hidden_dim=10, gate_type='rnn')
    return model


# =============================
# Recurrent Gate Model with RL
# =============================

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

        self.proj = nn.Conv2d(in_channels=hidden_dim, out_channels=1,
                              kernel_size=1, stride=1)
        self.prob = nn.Sigmoid()

    def hotter(self, t):
        self.proj.weight.data /= t
        self.proj.bias.data /= t

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
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

        out = out.squeeze()
        out = out.view(out.size(0), out.size(1), 1, 1)
        proj = self.proj(out).squeeze()
        prob = self.prob(proj)
        bi_prob = torch.stack([1-prob, prob]).t()

        # do action selection in the forward pass
        if self.training:
            # action = bi_prob.multinomial()
            dist = Categorical(bi_prob)
            action = dist.sample()
        else:
            dist = None
            action = (prob > 0.5).float()
        action_reshape = action.view(action.size(0), 1, 1, 1).float()
        return action_reshape, prob, action, dist


# ================================
# Recurrent Gate Model with RL
# ================================
class RecurrentGatedRLResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, embed_dim=10,
                 hidden_dim=10, **kwargs):
        self.inplanes = 64
        super(RecurrentGatedRLResNet, self).__init__()

        self.num_layers = layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # going to have 4 groups of layers. For the easiness of skipping,
        # We are going to break the sequential of layers into a list of layers.
        self._make_group(block, 64, layers[0], group_id=1, pool_size=56)
        self._make_group(block, 128, layers[1], group_id=2, pool_size=28)
        self._make_group(block, 256, layers[2], group_id=3, pool_size=14)
        self._make_group(block, 512, layers[3], group_id=4, pool_size=7)

        self.control = RNNGatePolicy(embed_dim, hidden_dim, rnn_type='lstm')

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.softmax = nn.Softmax()

        # save everything
        self.saved_actions = {}
        self.saved_dists = {}
        self.saved_outputs = {}
        self.saved_targets = {}

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

    def _make_group(self, block, planes, layers, group_id=1, pool_size=56):
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

    def _make_layer_v2(self, block, planes, stride=1, pool_size=56):
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

    def forward(self, x, target_var, reinforce=False):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # reinitialize hidden units
        self.control.hidden = self.control.init_hidden(batch_size)

        masks = []
        gprobs = []
        actions = []
        dists = []

        # must pass through the first layer in first group
        x = getattr(self, 'group1_layer0')(x)
        # gate takes the output of the current layer
        gate_feature = getattr(self, 'group1_gate0')(x)

        mask, gprob, action, dist = self.control(gate_feature)
        gprobs.append(gprob)
        masks.append(mask.squeeze())
        prev = x  # input of next layer

        current_device = torch.cuda.current_device()
        actions.append(action)
        dists.append(dist)

        for g in range(4):
            for i in range(0 + int(g == 0), self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev)
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x)
                prev = x = mask.expand_as(x)*x + (1-mask).expand_as(prev)*prev
                if not (g == 3 and (i == self.num_layers[g] - 1)):
                    gate_feature = getattr(self,
                                           'group{}_gate{}'.format(g+1, i))(x)
                    mask, gprob, action, dist = self.control(gate_feature)
                    gprobs.append(gprob)
                    masks.append(mask.squeeze())
                    actions.append(action)
                    dists.append(dist)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if reinforce:
            softmax = self.softmax(x)
            # action = softmax.multinomial()
            dist = Categorical(softmax)
            action = dist.sample()
            actions.append(action)
            dists.append(dist)

        with global_lock:
            self.saved_actions[current_device] = actions
            self.saved_outputs[current_device] = x
            self.saved_targets[current_device] = target_var
            self.saved_dists[current_device] = dists

        return x, masks, gprobs, self.control.hidden


def imagenet_rnn_gate_rl_18(pretrained=False, **kwargs):
    """ Construct SkipNet-18 + HRL.
    has the same architecture as SkipNet-18+SP """
    model = RecurrentGatedRLResNet(BasicBlock, [2, 2, 2, 2], embed_dim=10,
                                   hidden_dim=10, gate_type='rnn')
    return model


def imagenet_rnn_gate_rl_34(pretrained=False, **kwargs):
    """ Construct SkipNet-34 + HRL.
    has the same architecture as SkipNet-34+SP """
    model = RecurrentGatedRLResNet(BasicBlock, [3, 4, 6, 3], embed_dim=10,
                                   hidden_dim=10, gate_type='rnn')
    return model


def imagenet_rnn_gate_rl_50(pretrained=False, **kwargs):
    """ Construct SkipNet-50 + HRL.
    has the same architecture as SkipNet-50+SP """
    model = RecurrentGatedRLResNet(Bottleneck, [3, 4, 6, 3], embed_dim=10,
                                   hidden_dim=10, gate_type='rnn')
    return model


def imagenet_rnn_gate_rl_101(pretrained=False,  **kwargs):
    """ Construct SkipNet-101 + HRL.
    has the same architecture as SkipNet-101+SP """
    model = RecurrentGatedRLResNet(Bottleneck, [3, 4, 23, 3], embed_dim=10,
                                   hidden_dim=10, gate_type='rnn')
    return model


def imagenet_rnn_gate_rl_152(pretrained=False,  **kwargs):
    """ Construct SkipNet-152 + HRL.
    has the same architecture as SkipNet-152+SP """
    model = RecurrentGatedRLResNet(Bottleneck, [3, 8, 36, 3], embed_dim=10,
                                   hidden_dim=10, gate_type='rnn')
    return model


