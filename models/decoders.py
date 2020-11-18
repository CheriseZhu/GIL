"""Graph decoders."""
import manifolds
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.hyp_layers import HGATConv
from layers.layers import Linear, GCNConv, GATConv, SAGEConv
import geoopt.manifolds.poincare.math as pmath
import geoopt


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decoder_name is not None:
            input = (x, adj)
            probs = self.forward(input)
        else:
            if self.decode_adj:
                input = (x, adj)
                probs, _ = self.cls.forward(input)
            else:
                probs = self.cls.forward(x)
        return probs

    def forward(self, probs):
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GCNConv(args.dim, args.n_classes, False, False, args.dropout, args.bias, act)
        self.decode_adj = True
        self.decoder_name = None


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        act = lambda x: x
        if args.dataset == 'pubmed':
            self.cls = GATConv(args.dim, args.n_classes, 8, False, args.alpha, args.dropout, True, act)
        else:
            self.cls = GATConv(args.dim, args.n_classes, 1, False, args.alpha, args.dropout, True, act)
        self.decode_adj = True
        self.decoder_name = None


class SAGEDecoder(Decoder):
    def __init__(self, c, args):
        super(SAGEDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = SAGEConv(args.dim, args.n_classes, args.dropout, act)
        self.decode_adj = True
        self.decoder_name = None


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False
        self.decoder_name = None

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
            self.input_dim, self.output_dim, self.bias, self.c
        )


class DualDecoder(Decoder):
    def __init__(self, c, args):
        super(DualDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.in_features = args.dim
        act = getattr(F, args.act)
        if args.dataset == 'pubmed':
            self.cls_e = GATConv(self.in_features, args.n_classes, 8, False, args.alpha, args.dropout, args.bias,
                                 lambda x: x)
            self.cls_h = HGATConv(self.manifold, self.in_features, args.dim, 8, False, args.alpha,
                                  args.dropout, args.bias, act, atten=args.atten, dist=args.dist)
        else:
            self.cls_e = GATConv(self.in_features, args.n_classes, 1, args.concat, args.alpha, args.dropout, args.bias,
                                 lambda x: x)
            self.cls_h = HGATConv(self.manifold, self.in_features, args.dim, 1, args.concat, args.alpha,
                                  args.dropout, args.bias, act, atten=args.atten, dist=args.dist)

        self.in_features = args.dim
        self.out_features = args.n_classes
        self.c = c
        self.ball = ball = geoopt.PoincareBall(c=c)
        self.sphere = sphere = geoopt.manifolds.Sphere()
        self.scale = nn.Parameter(torch.zeros(self.out_features))
        point = torch.randn(self.out_features, self.in_features) / 4
        point = pmath.expmap0(point.to(args.device), c=c)
        tangent = torch.randn(self.out_features, self.in_features)
        self.point = geoopt.ManifoldParameter(point, manifold=ball)
        with torch.no_grad():
            self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()
        self.decoder_name = 'DualDecoder'

        '''prob weight'''
        self.w_e = nn.Linear(args.n_classes, 1, bias=False)
        self.w_h = nn.Linear(args.dim, 1, bias=False)
        self.drop_e = args.drop_e
        self.drop_h = args.drop_h
        self.reset_param()

    def reset_param(self):
        self.w_e.reset_parameters()
        self.w_h.reset_parameters()

    def forward(self, input):
        x, x_e = input[0]
        adj = input[1]
        '''Euclidean probs'''
        probs_e, _ = self.cls_e((x_e, adj))

        '''Hyper probs'''
        x = self.cls_h((x, adj))
        x = x.unsqueeze(-2)
        distance = pmath.dist2plane(
            x=x, p=self.point, a=self.tangent, c=self.ball.c, signed=True
        )
        probs_h = distance * self.scale.exp()

        '''Prob. Assembling'''
        w_h = torch.sigmoid(self.w_h(self.manifold.logmap0(x.squeeze(), self.c)))
        w_h = F.dropout(w_h, p=self.drop_h, training=self.training)
        w_e = torch.sigmoid(self.w_e(probs_e))
        w_e = F.dropout(w_e, p=self.drop_e, training=self.training)

        w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)
        w = F.normalize(w, p=1, dim=-1)
        probs = w[-1, 0] * probs_h + w[-1, 1] * probs_e

        return super(DualDecoder, self).forward(probs)


model2decoder = {
    'Shallow': LinearDecoder,
    'MLP': LinearDecoder,
    'HNN': LinearDecoder,
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'SGC': LinearDecoder,
    'SAGE': LinearDecoder,
    'HGCN': LinearDecoder,
    'HGNN': LinearDecoder,
    'HGAT': LinearDecoder,
    'GIL': DualDecoder
}
