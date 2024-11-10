import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv

class BaselineModels(nn.Module):
    """
    A class to encapsulate various baseline models for cryptocurrency price forecasting.
    
    Args:
        model_type (str): Type of the model to instantiate (e.g., 'A3TGCN', 'GCN', 'GAT', 'TGCN').
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.
        hidden_size (int): Number of hidden units in each model layer.
        layers_nb (int): Number of layers in the model (default is 2).
        output_activation (nn.Module, optional): Activation function for the output layer (default is None).
        use_gat (bool): Whether to use Graph Attention mechanism for A3T-GCN and TGCN (default is True).
    """
    
    def __init__(self, model_type: str, in_channels: int, out_channels: int, hidden_size: int, 
                 layers_nb: int = 2, output_activation: nn.Module = None, use_gat: bool = True):
        super(BaselineModels, self).__init__()

        # Model selection based on the model type
        if model_type == 'A3TGCN':
            self.model = A3TGCN(in_channels, out_channels, hidden_size, layers_nb, output_activation, use_gat)
        elif model_type == 'GCN':
            self.model = GCN(in_channels, [hidden_size] * layers_nb)  
        elif model_type == 'GAT':
            self.model = GAT(in_channels, [hidden_size] * layers_nb)  
        elif model_type == 'TGCN':
            self.model = TGCN(in_channels, out_channels, hidden_size, layers_nb, output_activation, use_gat)
        else:
            raise ValueError(f"Model type '{model_type}' is not recognized. Please choose from 'A3TGCN', 'GCN', 'GAT', or 'TGCN'.")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the selected model.

        Args:
            x (torch.Tensor): The feature matrix of the graph X_t (Nodes_nb, Features_nb, SeqLength).
            edge_index (torch.Tensor): The edge index of the graph A (2, Edges_nb).
            edge_weight (torch.Tensor): The edge weight of the graph (Edges_nb,).

        Returns:
            torch.Tensor: The output of the model (Nodes_nb, OutFeatures_nb).
        """
        return self.model(x, edge_index, edge_weight)


class GCN(nn.Module):
    """
    Simple two layers GCN model.
    """
    def __init__(self, in_channels: int, layer_sizes: list[int] = None, bias: bool = True, improved: bool = False):
        super(GCN, self).__init__()
        layer_sizes = layer_sizes or [32, 32]
        self.convs = nn.ModuleList([
           GCNConv(in_channels, layer_sizes[0], bias=bias, improved=improved),
        ] + [
           GCNConv(layer_sizes[i], layer_sizes[i + 1], bias=bias, improved=improved) for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor) -> torch.tensor:
        for conv in self.convs[:-1]:
            x = F.leaky_relu(conv(x, edge_index, edge_weight))
        return self.convs[-1](x, edge_index, edge_weight)

class GAT(nn.Module):
    """
    Simple two layers GAT model.
    """
    def __init__(self, in_channels: int, layer_sizes: list[int] = None, bias: bool = True):
        super(GAT, self).__init__()
        layer_sizes = layer_sizes or [32, 32]
        self.convs = nn.ModuleList([
           GATv2Conv(in_channels, layer_sizes[0], bias=bias, edge_dim=1),
        ] + [
           GATv2Conv(layer_sizes[i], layer_sizes[i + 1], bias=bias, edge_dim=1) for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor) -> torch.tensor:
        for conv in self.convs[:-1]:
            x = F.leaky_relu(conv(x, edge_index, edge_weight))
        return self.convs[-1](x, edge_index, edge_weight)

class A3TGCN(nn.Module):
    """
    A3T-GCN model adapted from https://arxiv.org/pdf/2006.11583.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_size: int, layers_nb: int = 2, output_activation: nn.Module = None, use_gat: bool = True):
        super(A3TGCN, self).__init__()
        self.hidden_size = hidden_size
        self.layers_nb = max(1, layers_nb)
        self.cells = nn.ModuleList(
            [TGCNCell(in_channels, hidden_size, use_gat=use_gat)] + 
            [TGCNCell(hidden_size, hidden_size, use_gat=use_gat) for _ in range(self.layers_nb - 1)]
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1),
        )
        nn.init.uniform_(self.attention[0].weight)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, out_channels),
            output_activation if output_activation is not None else nn.Identity(),
        )

    def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor) -> torch.tensor:
        """
        Performs a forward pass on the A3T-GCN model.
        :param x: The feature matrix of the graph X_t (Nodes_nb, Features_nb, SeqLength)
        :param edge_index: The edge index of the graph A (2, Edges_nb)
        :param edge_weight: The edge weight of the graph (Edges_nb,)
        :return: The output of the model (Nodes_nb, OutFeatures_nb)
        """
        h_prev = [
            torch.zeros(x.shape[0], self.hidden_size) for _ in range(self.layers_nb)
        ]
        h_final = torch.zeros(x.shape[0], x.shape[-1], self.hidden_size)
        for t in range(x.shape[-1]):
            h = x[:, :, t]  # h is the output of the previous GRU layer (the input features for the first layer)
            for i, cell in enumerate(self.cells):
                h = cell(h, edge_index, edge_weight, h_prev[i])
                h_prev[i] = h
            h_final[:, t, :] = h
        return self.out(F.leaky_relu(torch.sum(F.leaky_relu(h_final) * self.attention(h_final), dim=1)))
    
    
class TGCNCell(nn.Module):
    """
    T-GCN Cell for one timestep, combining GCN/GAT with GRU.
    """
    def __init__(self, in_channels: int, hidden_size: int, use_gat: bool = True):
        super(TGCNCell, self).__init__()
        self.graph_layer = GAT(in_channels, [hidden_size, hidden_size]) if use_gat else GCN(in_channels, [hidden_size, hidden_size])
        self.update_gate = nn.Linear(2 * hidden_size + in_channels, hidden_size)
        self.reset_gate = nn.Linear(2 * hidden_size + in_channels, hidden_size)
        self.candidate = nn.Linear(2 * hidden_size + in_channels, hidden_size)

    def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor, h_prev: torch.tensor) -> torch.tensor:
        graph_out = self.graph_layer(x, edge_index, edge_weight)  # Graph output
        combined = torch.cat([x, graph_out, h_prev], dim=-1)  # Combine inputs for GRU

        u = F.sigmoid(self.update_gate(combined))  # Update gate
        r = F.sigmoid(self.reset_gate(combined))    # Reset gate
        c = F.tanh(self.candidate(torch.cat([x, graph_out, r * h_prev], dim=-1)))  # Candidate state

        h_next = u * h_prev + (1 - u) * c  # Next hidden state
        return h_next

class TGCN(nn.Module):
    """
    Temporal Graph Convolutional Network (T-GCN) model.
    
    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.
        hidden_size (int): Number of hidden units in each TGCN cell.
        layers_nb (int): Number of TGCN cells (default is 2).
        output_activation (nn.Module, optional): Activation function for the output layer (default is None).
        use_gat (bool): Whether to use Graph Attention mechanism (default is True).
    """
    
    def __init__(self, in_channels: int, out_channels: int, hidden_size: int, 
                 layers_nb: int = 2, output_activation: nn.Module = None, use_gat: bool = True):
        super(TGCN, self).__init__()
        
        self.hidden_size = hidden_size
        self.layers_nb = max(1, layers_nb)
        
        # Create TGCN cells
        self.cells = nn.ModuleList(
            [TGCNCell(in_channels, hidden_size, use_gat)] + 
            [TGCNCell(hidden_size, hidden_size, use_gat) for _ in range(self.layers_nb - 1)]
        )
        
        # Output layer
        self.out = nn.Sequential(
            nn.Linear(hidden_size, out_channels),
            output_activation if output_activation is not None else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # Validate input dimensions
        assert x.dim() == 3, "Input x must be a 3D tensor."
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, "Edge index must be a 2D tensor with shape (2, Edges_nb)."
        assert edge_weight.dim() == 1, "Edge weights must be a 1D tensor."

        # Initialize hidden states for all layers
        h_prev = [torch.zeros(x.shape[0], self.hidden_size, device=x.device) for _ in range(self.layers_nb)]
        
        for t in range(x.shape[-1]):
            h = x[:, :, t]  # Get features at time step t
            for i, cell in enumerate(self.cells):
                h = cell(h, edge_index, edge_weight, h_prev[i])
                h_prev[i] = h
        
        return self.out(h_prev[-1])


