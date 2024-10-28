from typing import Optional

import torch
import torch.nn as nn
from .layer import *


class MTGNN(nn.Module):
    r"""An implementation of the Multivariate Time Series Forecasting Graph Neural Networks.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        gcn_true (bool): Whether to add graph convolution layer.
        build_adj (bool): Whether to construct adaptive adjacency matrix.
        gcn_depth (int): Graph convolution depth.
        num_nodes (int): Number of nodes in the graph.
        kernel_set (list of int): List of kernel sizes.
        kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
        dropout (float): Droupout rate.
        subgraph_size (int): Size of subgraph.
        node_dim (int): Dimension of nodes.
        dilation_exponential (int): Dilation exponential.
        conv_channels (int): Convolution channels.
        residual_channels (int): Residual channels.
        skip_channels (int): Skip channels.
        end_channels (int): End channels.
        seq_length (int): Length of input sequence.
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        layers (int): Number of layers.
        propalpha (float): Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1.
        tanhalpha (float): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate.
        layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
        xd (int, optional): Static feature dimension, default None.
    """

    def __init__(self, gcn_true: bool, build_adj: bool, gcn_depth: int, num_nodes: int, kernel_set: list, kernel_size: int,\
                dropout: float, subgraph_size: int, node_dim: int, dilation_exponential: int, conv_channels: int, residual_channels: int,\
                skip_channels: int, end_channels: int, seq_length: int, in_dim: int, out_dim: int, layers: int, propalpha: float, tanhalpha: float,\
                layer_norm_affline: bool, xd: Optional[int] = None):

        super(MTGNN, self).__init__()

        self._gcn_true = gcn_true
        self._build_adj_true = build_adj
        self._num_nodes = num_nodes
        self._dropout = dropout
        self._seq_length = seq_length
        self._layers = layers
        self._idx = torch.arange(self._num_nodes)

        self._mtgnn_layers = nn.ModuleList()

        self._graph_constructor = GraphConstructor(
            num_nodes, subgraph_size, node_dim, alpha = tanhalpha, xd = xd
        )

        self._set_receptive_field(dilation_exponential, kernel_size, layers)

        new_dilation = 1
        for j in range(1, layers + 1):
            self._mtgnn_layers.append(
                MTGNNLayer(
                    dilation_exponential = dilation_exponential,
                    rf_size_i = 1,
                    kernel_size = kernel_size,
                    j = j,
                    residual_channels = residual_channels,
                    conv_channels = conv_channels,
                    skip_channels = skip_channels,
                    kernel_set = kernel_set,
                    new_dilation = new_dilation,
                    layer_norm_affline = layer_norm_affline,
                    gcn_true = gcn_true,
                    seq_length = seq_length,
                    receptive_field = self._receptive_field,
                    dropout = dropout,
                    gcn_depth = gcn_depth,
                    num_nodes = num_nodes,
                    propalpha = propalpha,
                )
            )

            new_dilation *= dilation_exponential

        self._setup_conv(in_dim, skip_channels, end_channels, residual_channels, out_dim)

        self._reset_parameters()

    def _setup_conv(self, in_dim, skip_channels, end_channels, residual_channels, out_dim):

        self._start_conv = nn.Conv2d(in_channels = in_dim, out_channels = residual_channels, kernel_size = (1, 1))

        if self._seq_length > self._receptive_field:

            self._skip_conv_0 = nn.Conv2d(
                in_channels = in_dim,
                out_channels = skip_channels,
                kernel_size = (1, self._seq_length),
                bias = True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels = residual_channels,
                out_channels = skip_channels,
                kernel_size = (1, self._seq_length - self._receptive_field + 1),
                bias = True,
            )

        else:
            self._skip_conv_0 = nn.Conv2d(
                in_channels = in_dim,
                out_channels = skip_channels,
                kernel_size = (1, self._receptive_field),
                bias = True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels = residual_channels,
                out_channels = skip_channels,
                kernel_size = (1, 1),
                bias = True,
            )

        self._end_conv_1 = nn.Conv2d(
            in_channels = skip_channels,
            out_channels = end_channels,
            kernel_size = (1, 1),
            bias = True,
        )

        self._end_conv_2 = nn.Conv2d(
            in_channels = end_channels,
            out_channels = out_dim,
            kernel_size = (1, 1),
            bias = True,
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _set_receptive_field(self, dilation_exponential, kernel_size, layers):
        if dilation_exponential > 1:
            self._receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1

    def forward(self, X_in: torch.FloatTensor, A_tilde: Optional[torch.FloatTensor] = None, idx: Optional[torch.LongTensor] = None, FE: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN.

        Arg types:
            * **X_in** (PyTorch FloatTensor) - Input sequence, with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor, optional) - Predefined adjacency matrix, default None.
            * **idx** (Pytorch LongTensor, optional) - Input indices, a permutation of the num_nodes, default None (no permutation).
            * **FE** (Pytorch FloatTensor, optional) - Static feature, default None.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence for prediction, with shape (batch_size, seq_len, num_nodes, 1).
        """
        seq_len = X_in.size(3)
        assert (seq_len == self._seq_length), "Input sequence length not equal to preset sequence length."

        if self._seq_length < self._receptive_field:
            X_in = nn.functional.pad(X_in, (self._receptive_field - self._seq_length, 0, 0, 0))

        if self._gcn_true:
            if self._build_adj_true:
                if idx is None:
                    A_tilde = self._graph_constructor(self._idx.to(X_in.device), FE = FE)
                else:
                    A_tilde = self._graph_constructor(idx, FE = FE)

        X = self._start_conv(X_in)
        X_skip = self._skip_conv_0(F.dropout(X_in, self._dropout, training = self.training))
        if idx is None:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(X, X_skip, A_tilde, self._idx.to(X_in.device), self.training)
        else:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(X, X_skip, A_tilde, idx, self.training)

        X_skip = self._skip_conv_E(X) + X_skip
        X = F.relu(X_skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        return X