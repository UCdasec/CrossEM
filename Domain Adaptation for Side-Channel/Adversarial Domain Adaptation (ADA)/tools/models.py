import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import math
import pdb


pad_size = 5


def init_weights_normal(m):
    inition_func = torch.nn.init.xavier_normal
    if isinstance(m, nn.Linear):
        inition_func(m.weight)
    if isinstance(m, nn.Conv1d):
        inition_func(m.weight)


def init_weights_uniform(m):
    inition_func = torch.nn.init.xavier_uniform
    if isinstance(m, nn.Linear):
        inition_func(m.weight)
        #m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        inition_func(m.weight)
        #m.bias.data.fill_(0.01)


class ReverseLayerF(Function):
    ''' Reverse layer functions '''
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RevGrad(nn.Module):
    ''' network '''
    def __init__(self, inp_shape, batch_size=100, num_classes=256):
        super(RevGrad, self).__init__()
        self.clsNum = num_classes
        self.inp_shape = inp_shape

        act_func = nn.ReLU

        # feature extractor block
        self.feature = nn.Sequential()
        # block 1
        self.feature.add_module('f_conv1', nn.Conv1d(1, 64, kernel_size=11, stride=1, padding=pad_size))
        self.feature.add_module('f_relu1', act_func())
        self.feature.add_module('f_bn1', nn.BatchNorm1d(64, track_running_stats=False))
        self.feature.add_module('f_pool1', nn.AvgPool1d(2, stride=2))
        #self.feature.add_module('f_drop1', nn.Dropout(p=0.3))
        # block 2
        self.feature.add_module('f_conv2', nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=pad_size))
        self.feature.add_module('f_relu2', act_func())
        self.feature.add_module('f_bn2', nn.BatchNorm1d(128, track_running_stats=False))
        self.feature.add_module('f_pool2', nn.AvgPool1d(2, stride=2))
        #self.feature.add_module('f_drop2', nn.Dropout(p=0.3))
        # block 3
        self.feature.add_module('f_conv3', nn.Conv1d(128, 256, kernel_size=11, stride=1, padding=pad_size))
        self.feature.add_module('f_relu3', act_func())
        self.feature.add_module('f_bn3', nn.BatchNorm1d(256, track_running_stats=False))
        self.feature.add_module('f_pool3', nn.AvgPool1d(2, stride=2))
        #self.feature.add_module('f_drop3', nn.Dropout(p=0.3))
        # block 4
        self.feature.add_module('f_conv4', nn.Conv1d(256, 512, kernel_size=11, stride=1, padding=pad_size))
        self.feature.add_module('f_relu4', act_func())
        self.feature.add_module('f_bn4', nn.BatchNorm1d(512, track_running_stats=False))
        self.feature.add_module('f_pool4', nn.AvgPool1d(2, stride=2))
        #self.feature.add_module('f_drop4', nn.Dropout(p=0.3))
        # block 5
        self.feature.add_module('f_conv5', nn.Conv1d(512, 512, kernel_size=11, stride=1, padding=pad_size))
        self.feature.add_module('f_relu5', act_func())
        self.feature.add_module('f_bn5', nn.BatchNorm1d(512, track_running_stats=False))
        self.feature.add_module('f_pool5', nn.AvgPool1d(2, stride=2))
        #self.feature.add_module('f_drop5', nn.Dropout(p=0.3))

        self.feature.add_module('f_flatten', nn.Flatten())

        nsize = self._get_flatten_output(inp_shape, batch_size)

        self.feature.add_module('f_dense1', nn.Linear(nsize, 4096))
        self.feature.add_module('f_d_relu1', act_func())
        self.feature.add_module('f_dense2', nn.Linear(4096, 4096))
        self.feature.add_module('f_d_relu2', act_func())

        # source domain classifier block
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(4096, 1024))
        self.class_classifier.add_module('c_relu', act_func())
        self.class_classifier.add_module('c_out', nn.Linear(1024, self.clsNum))

        # domain discriminator block
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(4096, 1024))
        self.domain_classifier.add_module('d_relu1', act_func())
        self.domain_classifier.add_module('d_fc2', nn.Linear(1024, 512))
        self.domain_classifier.add_module('d_relu2', act_func())
        self.domain_classifier.add_module('d_fc3', nn.Linear(512, 256))
        self.domain_classifier.add_module('d_relu3', act_func())
        self.domain_classifier.add_module('d_out', nn.Linear(256, 2))

        # initialize the weights
        self.feature.apply(init_weights_normal)
        self.class_classifier.apply(init_weights_normal)
        self.domain_classifier.apply(init_weights_normal)

    def _get_flatten_output(self, inp_shape, batch_size):
        input = Variable(torch.rand(batch_size, *inp_shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.feature(x)
        return x

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        #feature = feature.view(feature.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return feature, class_output, domain_output


class CNN_Best(nn.Module):
    ''' network '''
    def __init__(self, inp_shape, num_classes=256, batch_size=100):
        super(CNN_Best, self).__init__()
        self.clsNum = num_classes
        self.inp_shape = inp_shape
        self.batch_size = batch_size

        act_func = nn.ReLU

        # feature extractor block
        self.feature = nn.Sequential()
        # block 1
        self.feature.add_module('f_conv1', nn.Conv1d(1, 64, kernel_size=11, stride=1, padding=pad_size))
        self.feature.add_module('f_relu1', act_func())
        #self.feature.add_module('f_bn1', nn.BatchNorm1d(64, track_running_stats=False))
        self.feature.add_module('f_pool1', nn.AvgPool1d(2, stride=2))
        # block 2
        self.feature.add_module('f_conv2', nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=pad_size))
        self.feature.add_module('f_relu2', act_func())
        #self.feature.add_module('f_bn2', nn.BatchNorm1d(128, track_running_stats=False))
        self.feature.add_module('f_pool2', nn.AvgPool1d(2, stride=2))
        # block 3
        self.feature.add_module('f_conv3', nn.Conv1d(128, 256, kernel_size=11, stride=1, padding=pad_size))
        self.feature.add_module('f_relu3', act_func())
        #self.feature.add_module('f_bn3', nn.BatchNorm1d(256, track_running_stats=False))
        self.feature.add_module('f_pool3', nn.AvgPool1d(2, stride=2))
        # block 4
        self.feature.add_module('f_conv4', nn.Conv1d(256, 512, kernel_size=11, stride=1, padding=pad_size))
        self.feature.add_module('f_relu4', act_func())
        #self.feature.add_module('f_bn4', nn.BatchNorm1d(512, track_running_stats=False))
        self.feature.add_module('f_pool4', nn.AvgPool1d(2, stride=2))
        # block 5
        self.feature.add_module('f_conv5', nn.Conv1d(512, 512, kernel_size=11, stride=1, padding=pad_size))
        self.feature.add_module('f_relu5', act_func())
        #self.feature.add_module('f_bn5', nn.BatchNorm1d(512, track_running_stats=False))
        self.feature.add_module('f_pool5', nn.AvgPool1d(2, stride=2))

        self.feature.add_module('f_flatten', nn.Flatten())

        nsize = self._get_flatten_output(inp_shape, batch_size)

        self.feature.add_module('f_dense1', nn.Linear(nsize, 4096))
        self.feature.add_module('f_d_relu1', act_func())
        self.feature.add_module('f_dense2', nn.Linear(4096, 4096))
        self.feature.add_module('f_d_relu2', act_func())

        self.feature.add_module('c_out', nn.Linear(4096, self.clsNum))

        # initialize the weights
        self.feature.apply(init_weights_normal)

    def _get_flatten_output(self, inp_shape, batch_size):
        input = Variable(torch.rand(batch_size, *inp_shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.feature(x)
        return x

    def forward(self, input_data):
        feature = self.feature(input_data)
        #feature = feature.view(feature.size(0), -1)
        return feature
