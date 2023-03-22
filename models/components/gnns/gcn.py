import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling


class Gcn(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gcn, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(in_channels, out_channels)
        self.conv3 = GCNConv(in_channels, out_channels)
        self.lin = Linear(in_channels, out_channels)


    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)
        
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
                
        return x
