import manifolds
import models.encoders as encoders
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.decoders import model2decoder
from layers.layers import FermiDiracDecoder
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.eval_utils import acc_f1

'''Implementation based on HGCN '''


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        probs = F.log_softmax(output[idx], dim=1)
        return probs

    def compute_metrics(self, embeddings, data, split, args):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)

        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]
        # return m1["acc"] < m2["acc"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.w_e = nn.Linear(args.dim, 1, bias=False)
        self.w_h = nn.Linear(args.dim, 1, bias=False)
        self.drop_e = args.drop_e
        self.drop_h = args.drop_h
        self.data = args.dataset
        self.model = args.model
        self.reset_param()

    def reset_param(self):
        self.w_e.reset_parameters()
        self.w_h.reset_parameters()

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        if isinstance(h, tuple):
            emb_in = h[0][idx[:, 0], :]
            emb_out = h[0][idx[:, 1], :]
            "compute hyperbolic dist"
            emb_in = self.manifold.logmap0(emb_in, self.c)
            emb_out = self.manifold.logmap0(emb_out, self.c)
            sqdist_h = torch.sqrt((emb_in - emb_out).pow(2).sum(dim=-1) + 1e-15)
            probs_h = self.dc.forward(sqdist_h)

            "compute dist in Euclidean"
            emb_in_e = h[1][idx[:, 0], :]
            emb_out_e = h[1][idx[:, 1], :]
            sqdist_e = torch.sqrt((emb_in_e - emb_out_e).pow(2).sum(dim=-1) + 1e-15)
            probs_e = self.dc.forward(sqdist_e)

            # sub
            w_h = torch.sigmoid(self.w_h(emb_in - emb_out).view(-1))
            w_e = torch.sigmoid(self.w_e(emb_in_e - emb_out_e).view(-1))
            w = torch.cat([w_h.view(-1, 1), w_e.view(-1, 1)], dim=-1)
            if self.data == 'pubmed':
                w = F.normalize(w, p=1, dim=-1)
            probs = torch.sigmoid(w[-1, 0] * probs_h + w[-1, 1] * probs_e)

            assert torch.min(probs) >= 0
            assert torch.max(probs) <= 1
        else:
            emb_in = h[idx[:, 0], :]
            emb_out = h[idx[:, 1], :]
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            assert torch.max(sqdist) >= 0
            probs = self.dc.forward(sqdist)

        return probs

    def compute_metrics(self, embeddings, data, split, args):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)

        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()

        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])
