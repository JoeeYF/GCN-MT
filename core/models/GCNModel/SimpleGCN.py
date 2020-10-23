import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, inchannel, outchannel, bias=False):
        super(GraphConv, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.linear = nn.Linear(inchannel, outchannel, bias=bias)

    def forward(self, adj, x):
        out = self.linear(torch.matmul(adj, x))
        return out


class SimpleGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0, alpha=0.2, nheads=2):
        super(SimpleGCN, self).__init__()

        self.GCN1 = GraphConv(nfeat, nhid)
        self.GCN2 = GraphConv(nhid, nclass)

    def forward(self, feature_s, feature_t):
        """
        input: Batch_size*Feature_num
        adj: Batch_size*Batch_size
        """
        adj = feature_s.mm(feature_t.t())
        adj = torch.softmax(adj, dim=1)
        x = feature_s
        x1 = self.GCN1(adj, x)
        x = self.GCN2(adj, x1)
        return x1,x
