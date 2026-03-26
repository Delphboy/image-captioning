import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class GnnLayer(nn.Module):
    def __init__(
        self, convolution: str, in_dim: int, out_dim: int, dropout_rate: float
    ):
        super().__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        if convolution.upper() == "GCN":
            self.mpnn = gnn.GCNConv(in_dim, out_dim)
        elif convolution.upper() == "GAT":
            # TODO: Investigate using multiple heads
            self.mpnn = gnn.GATv2Conv(in_dim, out_dim)
        elif convolution.upper() == "GIN":
            self.mpnn = gnn.GINConv(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim * 2, bias=False),
                    nn.LayerNorm(out_dim * 2),
                    nn.ReLU(),
                    nn.Linear(out_dim * 2, out_dim, bias=False),
                )
            )
        else:
            raise ValueError(f"The convolution type {convolution} is not supported")

        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    # Initialize weights with identity matrix
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.eye_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, gnn.GCNConv) or isinstance(m, gnn.GATv2Conv):
                torch.nn.init.eye_(m.lin.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, gnn.GINConv):
                torch.nn.init.eye_(m.nn[0].lin.weight)
                torch.nn.init.eye_(m.nn[3].lin.weight)
                if m.nn[0].bias is not None:
                    torch.nn.init.zeros_(m.nn[0].bias)
                if m.nn[3].bias is not None:
                    torch.nn.init.zeros_(m.nn[3].bias)

    def forward(self, nodes, edges):
        x = self.mpnn(nodes, edges)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class Gnn(nn.Module):
    """
    A GNN Wrapper Class
    Given a convolution and required dimensions, wrap around
    the PyG MPNN convolution and provide normalation/activations
    as required.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        convolution: str = "GCN",
        layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.convoltuion = convolution
        self.layers = layers
        self.activation = nn.ReLU()
        self.dropout = dropout
        self.network = self._build_gnn()

    def _build_layer(self, convolution: str, in_dim: int, out_dim: int) -> nn.Module:
        return GnnLayer(convolution, in_dim, out_dim, self.dropout)

    def _build_gnn(self) -> nn.ModuleList:
        layers = []
        for layer_idx in range(1, self.layers + 1):
            if layer_idx == self.layers and self.layers == 1:
                layer = self._build_layer(
                    self.convoltuion, self.in_features, self.out_features
                )
            elif layer_idx == self.layers:
                layer = self._build_layer(
                    self.convoltuion, self.hidden_features, self.out_features
                )
            elif layer_idx == 1:
                layer = self._build_layer(
                    self.convoltuion, self.in_features, self.hidden_features
                )
            else:
                layer = self._build_layer(
                    self.convoltuion, self.hidden_features, self.hidden_features
                )
            layers.append(layer)
        modules = nn.ModuleList(layers)
        return modules

    def forward(self, graph):
        x = graph.x

        for layer in self.network:
            x = layer(x, graph.edge_index)

        graph.x = x
        return graph
