"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import geoopt.manifolds.poincare.math as pmath
from layers.layers import GCNConv, GATConv, HFusion, EFusion
from layers.layers import remove_self_loops, add_self_loops, softmax, MessagePassing, glorot, zeros


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HGNNLayer(nn.Module):
    """
    HGNN layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias):
        super(HGNNLayer, self).__init__()
        self.conv = GCNConv(in_features, out_features, False, False, dropout, use_bias, act)
        self.p = dropout

    def forward(self, input):
        x, adj = input
        h = pmath.logmap0(x)
        h, _ = self.conv((h, adj))
        h = F.dropout(h, p=self.p, training=self.training)
        h = pmath.project(pmath.expmap0(h))
        h = F.relu(h)
        output = h, adj
        return output


class HGCNLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias):
        super(HGCNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias, self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.dropout = dropout

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HGATLayer(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, act, args):
        super(HGATLayer, self).__init__()
        self.conv = HGATConv(manifold, in_features, out_features, args.n_heads, args.concat, args.alpha, args.dropout,
                             args.bias, act, dist=0)

    def forward(self, input):
        x = input[0]
        adj = input[1]
        "hyper forward"
        input_h = x, adj
        x = self.conv(input_h)
        return x, adj


class GILayer(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, act, args):
        super(GILayer, self).__init__()
        self.conv = HGATConv(manifold, in_features, out_features, args.n_heads, args.concat, args.alpha, args.dropout,
                             args.bias, act, atten=args.atten, dist=args.dist)
        self.conv_e = GATConv(in_features, out_features, args.n_heads, args.concat, args.alpha,
                              args.dropout, args.bias, act)

        '''feature fusion'''
        self.h_fusion = HFusion(c, args.drop_e)
        self.e_fusion = EFusion(c, args.drop_h)

    def forward(self, input):
        x, x_e = input[0]
        adj = input[1]
        "hyper forward"
        input_h = x, adj
        x = self.conv(input_h)

        "eucl forward"
        input_e = x_e, adj
        x_e, _ = self.conv_e(input_e)

        "feature fusion"
        x = self.h_fusion(x, x_e)
        x_e = self.e_fusion(x, x_e)

        return (x, x_e), adj


class HGATConv(MessagePassing):
    def __init__(self,
                 manifold,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 act=None,
                 atten=True,
                 dist=True):
        super(HGATConv, self).__init__('add')

        self.manifold = manifold
        self.c = 1.0
        self.concat = concat
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if concat:
            self.out_channels = out_channels // heads
        else:
            self.out_channels = out_channels

        self.in_channels = in_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.act = act
        self.dist = dist
        self.atten = atten

        self.hy_linear = HypLinear(manifold, in_channels, heads * self.out_channels, 1, dropout, bias)
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)
        self.hy_linear.reset_parameters()

    def forward(self, input):
        x, adj = input
        x = self.hy_linear.forward(x)

        edge_index = adj._indices()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        log_x = pmath.logmap0(x, c=1.0)  # Get log(x) as input to GCN
        log_x = log_x.view(-1, self.heads, self.out_channels)
        out = self.propagate(edge_index, x=log_x, num_nodes=x.size(0), original_x=x)
        out = self.manifold.proj_tan0(out, c=self.c)

        out = self.act(out)
        out = self.manifold.proj_tan0(out, c=self.c)

        return self.manifold.proj(self.manifold.expmap0(out, c=self.c), c=self.c)

    def message(self, edge_index_i, x_i, x_j, num_nodes, original_x_i, original_x_j):
        # Compute attention coefficients.
        if self.atten:
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            if self.dist:  # Compute distance
                dist = pmath.dist(original_x_i, original_x_j)
                dist = softmax(dist, edge_index_i, num_nodes).reshape(-1, 1)
                alpha = alpha * dist
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, edge_index_i, num_nodes)

            # Sample attention coefficients stochastically.
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            return x_j * alpha.view(-1, self.heads, 1)
        else:
            return x_j

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
