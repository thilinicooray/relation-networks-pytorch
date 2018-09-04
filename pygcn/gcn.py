'''
Copyright https://github.com/tkipf/pygcn
'''

import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid*2)
        self.dropout = dropout
        self.final = nn.Linear(nhid*2, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout)

        final_x = self.final(x)
        return final_x