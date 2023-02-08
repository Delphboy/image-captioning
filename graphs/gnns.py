import torch
import torch.nn as nn
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops, degree


class Gcn(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gcn, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(in_channels, out_channels)
        self.conv3 = GCNConv(in_channels, out_channels)
        self.lin = Linear(in_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

# class Gcn(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr='add')
#         self.lin = nn.Linear(in_channels, out_channels, bias=False)
#         self.bias = Parameter(torch.Tensor(out_channels))
        
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         self.bias.data.zero_()
    
#     def forward(self, x, edge_index):
#         edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))
        
#         x = self.lin(x)
#         row, col = edge_index
        
#         deg = degree(col)#, x.size(), dtype=x.dtype)
        
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         out = self.propagate(edge_index, x=x, norm=norm)
#         out += self.bias
        
#         return out
    
#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j
    