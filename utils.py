import torch.nn as nn
import math
from torch.nn import init
import torch
import pdb
import numpy as np

'''def init_weight(module):
    if isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight)
        #bais cannot be handle by kaiming
        if module.bias is not None:
            init.kaiming_normal_(module.bias)'''

def init_weight(linear, pad_idx=None):
    if isinstance(linear, nn.Conv2d):
        init.xavier_normal_(linear.weight)
        '''n = linear.kernel_size[0] * linear.kernel_size[1] * linear.out_channels
        linear.weight.data.normal_(0, math.sqrt(2. / n))'''
    if isinstance(linear, nn.Linear):
        init.xavier_normal_(linear.weight)
    if isinstance(linear, nn.Embedding):
        bias = np.sqrt(3.0 / linear.weight.size(1))
        nn.init.uniform_(linear.weight, -bias, bias)
        if pad_idx is not None:
            linear.weight.data[pad_idx] = 0

def init_gru_cell(input):

    weight = eval('input.weight_ih')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform_(weight, -bias, bias)
    weight = eval('input.weight_hh')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform_(weight, -bias, bias)

    if input.bias:
        weight = eval('input.bias_ih' )
        weight.data.zero_()
        weight.data[input.hidden_size: 2 * input.hidden_size] = 1
        weight = eval('input.bias_hh')
        weight.data.zero_()
        weight.data[input.hidden_size: 2 * input.hidden_size] = 1

def init_lstm(input_lstm):

    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l'+str(ind))
        nn.init.orthogonal(weight)
        weight = eval('input_lstm.weight_hh_l'+str(ind))
        nn.init.orthogonal(weight)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l'+str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_non_linearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_non_linearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def format_dict(d, s, p):
    rv = ""
    for (k,v) in d.items():
        if len(rv) > 0: rv += " , "
        rv+=p+str(k) + ":" + s.format(v*100)
    return rv

def cross_entropy_loss(pred, target, ignore_index=None):
    #-x[class] + log (sum_j exp(x[j]))
    if ignore_index is not None and target == ignore_index:
        #print('no loss')
        return 0
    _x_class = - pred[target]
    denom = torch.log(torch.sum(torch.exp(pred)))

    print('target :', target, _x_class, denom)

    loss = _x_class + denom
    print('single loss ', loss)
    return loss

def likelihood(pred, target, ignore_index=None):
    if target == ignore_index:
        #print('no loss')
        return 0
    likelihood = pred[target]

    return likelihood

