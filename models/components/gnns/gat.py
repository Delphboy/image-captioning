import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm


class Gat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gat, self).__init__()
        self.gat1 = GATConv(in_channels, out_channels)
        self.gat2 = GATConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)
        self.lin = nn.Linear(in_channels, out_channels)


    def forward(self, graph_batch, pool=False):
        x, edge_index, edge_attr, batch = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch
        edge_index = edge_index.to(torch.int64)
        edge_attr = edge_attr.to(torch.float32)

        if len(edge_index) > 0:
            x = self.gat1(x, edge_index, edge_attr)
            x = F.leaky_relu(x)
            x = self.gat2(x, edge_index, edge_attr)
            x = F.leaky_relu(x)
            x = self.bn(x)

        if pool:
            x = global_mean_pool(x, batch)
            x = F.dropout(x, p=0.1, training=self.training)
            x = self.lin(x)
            return x

        graph_batch.x = x

        # unbatch the graphs into a set
        graphs = graph_batch.to_data_list()
        X = [g.x for g in graphs]
        X = nn.utils.rnn.pad_sequence(X, batch_first=True)
        return X


