'''
Copyright https://github.com/tkipf/pygcn
'''


import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .. import utils


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self, val=None):
        if val is None:
            fan = self.in_features +  self.out_features
            spread = math.sqrt(2.0) * math.sqrt( 2.0 / fan )
        else:
            spread = val
        self.weight.data.uniform_(-spread,spread)
        if self.bias is not None:
            self.bias.data.uniform_(-spread,spread)


    def forward(self, input, adj):
        support = torch.bmm(input, self.weight.expand(input.size(0),self.in_features,self.out_features))
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'