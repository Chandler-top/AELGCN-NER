import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import copy


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, device, gcn_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()

        self.gcn_dim = gcn_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling

        self.W2 = nn.Linear(self.gcn_dim, self.gcn_dim).to(self.device)
        # EANJU W2
        self.W1 = nn.Linear(self.gcn_dim, self.gcn_dim).to(self.device)
        self.highway = Edgeupdate(gcn_dim, self.dep_embed_dim, dropout_ratio=0.5).to(self.device)

    # def forward(self, weight_adj, gcn_inputs):
    def forward(self, weight_adj, gcn_inputs, adj_matrix):
        """
        :param weight_adj: [batch, seq, seq, dim_e]
        :param gcn_inputs: [batch, seq, dim]
        :return:
        """
        # Edge-Aware Node Joint Update Module
        batch, seq, dim = gcn_inputs.shape
        weight_adj = weight_adj.permute(0, 3, 1, 2)  # [batch, dim_e, seq, seq]
        gcn_inputs2 = gcn_inputs
        gcn_inputs1 = gcn_inputs.unsqueeze(1).expand(batch, self.dep_embed_dim, seq, dim)
        Ax = torch.matmul(weight_adj, gcn_inputs1)  # [batch, dim_e, seq, dim]
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)

        # 将 Pool(Hl_1, Hl_2, ..., Hl_p) 与原始邻接矩阵相乘 论文公式（5）+的 后半部分, 线性变换和激活函数的操作,即消融实验中提到的vanilla GCN模块
        Ax2 = F.relu(self.W1(torch.matmul(adj_matrix, gcn_inputs2)))
        # 将原始邻接矩阵与线性变换后的结果相加，要保证Ax2 形状[batch, seq, dim]
        Ax = Ax + Ax2
        # Ax: [batch, seq, dim] 通过线性变化完成聚合即论文中的公式（6）
        gcn_outputs = self.W2(Ax)
        # EANU
        # weights_gcn_outputs = F.relu(gcn_outputs)
        # node_outputs = weights_gcn_outputs
        # EANJU
        node_outputs = gcn_outputs
        # Edge update weight_adj[batch, dim_e, seq, seq]
        # Node-Aware Edge Update Module
        weight_adj = weight_adj.permute(0, 2, 3, 1).contiguous()  # [batch, seq, seq, dim_e]
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)
        return node_outputs, edge_outputs


class Edgeupdate(nn.Module):
    def __init__(self, hidden_dim, dim_e, dropout_ratio=0.5):
        super(Edgeupdate, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.dim_e, self.dim_e)

    def forward(self, edge, node1, node2):
        """
        :param edge: [batch, seq, seq, dim_e]
        :param node: [batch, seq, seq, dim]
        :return:
        """

        node = torch.cat([node1, node2], dim=-1) # [batch, seq, seq, dim * 2]
        edge = self.W(torch.cat([edge, node], dim=-1))
        return edge  # [batch, seq, seq, dim_e]


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0, c0


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])