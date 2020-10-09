import torch
import torch.nn as nn
import torch.nn.functional as F
from .GraphAttentionConv import GraphAttentionLayer


class MutilHeadGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0, alpha=0.2, nheads=2):
        """Dense version of GAT."""
        super(MutilHeadGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        """
        input: Batch_size*Feature_num
        adj: Batch_size*Batch_size
        """
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.att1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)

        self.out_att = GraphAttentionLayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.att1(x,adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x