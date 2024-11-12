from old_layer import *
from typing import Optional, List

class gtnet(nn.Module):
    def __init__(self, gcn_true: bool, buildA_true: bool, gcn_depth: int, num_nodes: int, kernel_size: int, kernel_set: List[int], device: str, dropout: float, subgraph_size: int, node_dim: int, dilation_exponential: int,
                 conv_channels: int, residual_channels: int, skip_channels: int, end_channels: int, seq_length: int, in_dim: int, out_dim: int, layers: int, propalpha: float, tanhalpha: float, attention_layer: bool,
                 layer_norm_affline = True, predefined_A: torch.FloatTensor = None, static_feat: Optional[int] = None):

        r"""An implementation of the Multivariate Time Series Forecasting Graph Neural Networks.
        For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
        <https://arxiv.org/pdf/2005.11650.pdf>`_

        Args:
            gcn_true (bool): Whether to add graph convolution layer.
            build_adj (bool): Whether to construct adaptive adjacency matrix.
            gcn_depth (int): Graph convolution depth.
            num_nodes (int): Number of nodes in the graph.
            kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
            kernel_set (list of int): List of kernel sizes.
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
            predefined_adj (FloatTensor): Predefined adjacency matrix
            static_feat (int, optional): Static feature dimension, default None.
        """

        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels = in_dim, out_channels = residual_channels, kernel_size = (1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha = tanhalpha, static_feat = static_feat)

        self.seq_length = seq_length
        kernel_size = 7

        if dilation_exponential > 1:
            self._receptive_field = int(1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1


        rf_size_i = 1
        new_dilation = 1

        for j in range(1, layers + 1):
            if dilation_exponential > 1:
                rf_size_j = int(rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            self.filter_convs.append(dilated_inception(kernel_set, residual_channels, conv_channels, dilation_factor = new_dilation))
            self.gate_convs.append(dilated_inception(kernel_set, residual_channels, conv_channels, dilation_factor = new_dilation))
            self.residual_convs.append(nn.Conv2d(in_channels = conv_channels, out_channels = residual_channels, kernel_size = (1, 1)))
            if self.seq_length > self._receptive_field:
                self.skip_convs.append(nn.Conv2d(in_channels = conv_channels, out_channels = skip_channels, kernel_size = (1, self.seq_length - rf_size_j + 1)))
            else:
                self.skip_convs.append(nn.Conv2d(in_channels = conv_channels, out_channels = skip_channels, kernel_size = (1, self._receptive_field - rf_size_j + 1)))

            if self.gcn_true:
                self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

            if self.seq_length>self._receptive_field:
                self.norm.append(layernorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1), elementwise_affine = layer_norm_affline))
            else:
                self.norm.append(layernorm((residual_channels, num_nodes, self._receptive_field - rf_size_j + 1), elementwise_affine = layer_norm_affline))

            new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels = skip_channels, out_channels = end_channels, kernel_size = (1, 1), bias = True)
        self.end_conv_2 = nn.Conv2d(in_channels = end_channels, out_channels = out_dim, kernel_size = (1, 1), bias = True)
        if self.seq_length > self._receptive_field:
            self.skip0 = nn.Conv2d(in_channels = in_dim, out_channels = skip_channels, kernel_size = (1, self.seq_length), bias = True)
            self.skipE = nn.Conv2d(in_channels = residual_channels, out_channels = skip_channels, kernel_size = (1, self.seq_length - self._receptive_field + 1), bias = True)

        else:
            self.skip0 = nn.Conv2d(in_channels = in_dim, out_channels = skip_channels, kernel_size = (1, self._receptive_field), bias = True)
            self.skipE = nn.Conv2d(in_channels = residual_channels, out_channels = skip_channels, kernel_size = (1, 1), bias = True)

        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input: torch.FloatTensor, idx: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN.

        Arg types:
            * **input** (PyTorch FloatTensor) - Input sequence, with shape (batch_size, in_dim, num_nodes, seq_len).
            * **idx** (Pytorch LongTensor, optional) - Input indices, a permutation of the num_nodes, default None (no permutation).

        Return types:
            * **x** (PyTorch FloatTensor) - Output sequence for prediction, with shape (batch_size, seq_len, num_nodes, 1).
        """

        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self._receptive_field:
            input = nn.functional.pad(input, (self._receptive_field - self.seq_length, 0, 0, 0))


        if self.gcn_true:
            if self.buildA_true:
                adp = self.gc(self.idx) if idx is None else self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training = self.training))

        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training = self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip

            x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0)) if self.gcn_true else self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx) if idx is None else self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x
