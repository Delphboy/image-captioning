import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GatMeanPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatMeanPool, self).__init__()
        self.gat1 = GATConv(in_channels, out_channels)
        self.gat2 = GATConv(in_channels, out_channels)
        self.gat3 = GATConv(in_channels, out_channels)
        self.lin = nn.Linear(in_channels, out_channels)


    def forward(self, graph_batch):
        x, edge_index, edge_attr, batch = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch
        edge_index = edge_index.to(torch.int64)

        if len(edge_index) > 0:
            x = self.gat1(x, edge_index, edge_attr)
            x = x.relu()
            x = self.gat2(x, edge_index, edge_attr)
            x = x.relu()
            x = self.gat3(x, edge_index, edge_attr)
        
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
                
        return x

