from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from typing import Optional, List


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvwl->ncwl', (x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        r"""An implementation of the linear layer, conducting 2D convolution.

        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            bias (bool, optional): Whether to have bias. Default: True.
        """

        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size = (1, 1), padding = (0, 0), stride = (1, 1), bias = bias)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the linear layer.

        Arg types:
            * **x** (Pytorch Float Tensor) - Input tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **x** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, c_out, num_nodes, seq_len).
        """

        return self.mlp(x)

class mixprop(nn.Module):
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        r"""An implementation of the mix-hop propagation layer.

        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            gdep (int): Depth of graph convolution.
            dropout (float): Dropout rate.
            alpha (float): Ratio of retaining the root nodes's original states, a value between 0 and 1.
        """

        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x: torch.FloatTensor, adj: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of mix-hop propagation.

        Arg types:
            * **x** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).
            * **adj** (PyTorch Float Tensor) - Adjacency matrix, with shape (num_nodes, num_nodes).

        Return types:
            * **ho** (PyTorch Float Tensor) - Hidden representation for all nodes, with shape (batch_size, c_out, num_nodes, seq_len).
        """

        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)

        for i in range(self.gdep):
            h = self.alpha*x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)

        ho = torch.cat(out, dim = 1)
        ho = self.mlp(ho)
        return ho

class dilated_inception(nn.Module):
    def __init__(self, kernel_set: List[int], cin: int, cout: int, dilation_factor: Optional[int] = 2):
        r"""
        An implementation of the dilated inception layer.
        Args:
            kernel_set (list of int): List of kernel sizes.
            cin (int): Number of input channels.
            cout (int): Number of output channels.
            dilated_factor (int, optional): Dilation factor.
        """    

        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = kernel_set
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation = (1, dilation_factor)))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of dilated inception.

        Arg types:
            * **input** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, cin, num_nodes, seq_len).

        Return types:
            * **x** (PyTorch Float Tensor) - Hidden representation for all nodes,
            with shape (batch_size, c_out, num_nodes, seq_len - 6).
        """

        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim = 1)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes: int, k: int, dim: int, device: str, alpha: Optional[float] = 3, static_feat: Optional[int] = None):
        r"""An implementation of the graph learning layer to construct an adjacency matrix.

        Args:
            nnodes (int): Number of nodes in the graph.
            k (int): Number of largest values to consider in constructing the neighbourhood of a node (pick the "nearest" k nodes).
            device (str): Device used for model training
            dim (int): Dimension of the node embedding.
            alpha (float, optional): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate
            static_feat (int, optional): Static feature dimension, default None.
        """

        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, idx: torch.LongTensor) -> torch.FloatTensor:
        """
        Making a forward pass to construct an adjacency matrix from node embeddings.

        Arg types:
            * **idx** (Pytorch Long Tensor) - Input indices, a permutation of the number of nodes, default None (no permutation).
        Return types:
            * **adj** (PyTorch Float Tensor) - Adjacency matrix constructed from node embeddings.
        """

        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx: torch.LongTensor) -> torch.FloatTensor:
        '''
        Construct full adjacency matrix from node embeddings.

        Arg types:
            * **idx** (Pytorch Long Tensor) - Input indices, a permutation of the number of nodes, default None (no permutation).
        Return types:
            * **adj** (PyTorch Float Tensor) - Adjacency matrix constructed from node embeddings.

        '''

        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj

class layernorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape: int, eps: Optional[float] = 1e-5, elementwise_affine: bool = True):
        r"""An implementation of the layer normalization layer.

        Args:
            normalized_shape (int): Input shape from an expected input of size.
            eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5.
            elementwise_affine (bool, optional): Whether to conduct elementwise affine transformation or not. Default: True.
        """

        super(layernorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: torch.FloatTensor, idx: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of layer normalization.

        Arg types:
            * **input** (Pytorch Float Tensor) - Input tensor,
                with shape (batch_size, feature_dim, num_nodes, seq_len).
            * **idx** (Pytorch Long Tensor) - Input indices.

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor,
                with shape (batch_size, feature_dim, num_nodes, seq_len).
        """

        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
