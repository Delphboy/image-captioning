import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class Gcn(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gcn, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GATConv(in_channels, out_channels)
        self.conv2 = GCNConv(in_channels, out_channels)
        self.conv3 = GCNConv(in_channels, out_channels)
        self.lin = Linear(in_channels, out_channels)


    def forward(self, graph_batch):
        x, edge_index, edge_attr, batch = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch
        edge_index = edge_index.to(torch.int64)

        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)
        
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
                
        return x
