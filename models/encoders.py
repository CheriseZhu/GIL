"""Graph encoders."""
import manifolds
import layers.hyp_layers as hyp_layers
import numpy as np
import torch
import torch.nn as nn
from layers.layers import Linear, get_dim_act, GCNConv, GATConv, SGConv, SAGEConv


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)

        return output


'''============Shallow=========='''


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)


'''=============NN=============='''


class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                hyp_layers.HNNLayer(
                    self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)


'''===========GNN================'''


class GCN(Encoder):
    """
    Graph Convolutional Neural Networks
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GCNConv(in_dim, out_dim, False, False, args.dropout, args.bias, act))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True


class GAT(Encoder):
    """
    Graph Attention Networks
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            # out_dim = dims[i + 1] // args.n_heads
            out_dim = dims[i + 1]
            concat = args.concat
            gat_layers.append(
                GATConv(in_dim, out_dim, args.n_heads, concat, args.alpha, args.dropout, args.bias, act))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True


class SGC(Encoder):
    """
    Simplifying graph convolutional networks
    """

    def __init__(self, c, args):
        super(SGC, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            out_dim = dims[i + 1]
            layers.append(
                SGConv(in_dim, out_dim, K=2, cached=False, dropout=args.dropout, bias=args.bias, act=act))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = True


class SAGE(Encoder):
    """
    Inductive Representation Learning on Large Graphs
    """

    def __init__(self, c, args):
        super(SAGE, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            out_dim = dims[i + 1]
            layers.append(
                SAGEConv(in_dim, out_dim, dropout=args.dropout, bias=args.bias, act=act))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = True


'''===========Hyperbolic=========='''


class HGCN(Encoder):
    """
    Hyperbolic GCN
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 0
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HGCNLayer(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.input_type = args.input_type

    def encode(self, x, adj):
        if self.input_type == 'eucl':
            x_hyp = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]),
                c=self.curvatures[0])
        else:
            x_hyp = x
        return super(HGCN, self).encode(x_hyp, adj)


class HGNN(Encoder):
    """
    Hyperbolic GNN
    """

    def __init__(self, c, args):
        super(HGNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 0
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HGNNLayer(
                    self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.input_type = args.input_type

    def encode(self, x, adj):
        if self.input_type == 'eucl':
            x_hyp = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]),
                c=self.curvatures[0])
        else:
            x_hyp = x
        return super(HGNN, self).encode(x_hyp, adj)


class HGAT(Encoder):
    """
    Hyperbolic GAT
    """

    def __init__(self, c, args):
        super(HGAT, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 0
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HGATLayer(self.manifold, in_dim, out_dim, c, act, args)
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.input_type = args.input_type

    def encode(self, x, adj):
        if self.input_type == 'eucl':
            x_hyp = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.curvatures[0]), c=self.curvatures[0]),
                c=self.curvatures[0])
        else:
            x_hyp = x
        return super(HGAT, self).encode(x_hyp, adj)


class GIL(Encoder):
    """
    Geometry Interaction Learning including Euclidean and Hyperbolic
    """

    def __init__(self, c, args):
        super(GIL, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 0
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.GILayer(
                    self.manifold, in_dim, out_dim, c, act, args
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.input_type = args.input_type

    def encode(self, x, adj):
        if self.input_type == 'eucl':
            x_hyp = self.manifold.proj(
                self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c),
                c=self.c)
        else:
            x_hyp = x
        return super(GIL, self).encode((x_hyp, x), adj)
