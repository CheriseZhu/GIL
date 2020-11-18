import scipy.sparse as sp
import networkx as nx
from utils.data_utils import load_synthetic_data

dataset = 'disease'
adj = sp.load_npz('../data/' + dataset + '_nc/' + dataset + '_adj.npz')
# adj, _, _ = load_synthetic_data(dataset_str=dataset, use_feats=True, data_path=data_path)
G = nx.from_scipy_sparse_matrix(adj, nx.Graph)

li = list(nx.connected_component_subgraphs(G))
connected_G = li[0]
connected_G.remove_edges_from(connected_G.selfloop_edges())

from sage.graphs.hyperbolicity import hyperbolicity, hyperbolicity_distribution

g = Graph(connected_G)
hyperbolicity_distribution(g, algorithm='sampling')
L, C, U = hyperbolicity(g, algorithm='BCCM')

sum = 0
for i in range(len(li)):
    g = li[i]
    g.remove_edges_from(g.selfloop_edges())
    g = Graph(g)
    L, _, _ = hyperbolicity(g, algorithm='BCCM')
    sum = sum + L

'''hyperbolicity_distribution'''
# airport: {0: 0.6376, 1/2: 0.3563, 1: 0.0061}
# disease: {0: 1.0000}
# cora: {0: 0.4474, 1/2: 0.4073, 1: 0.1248, 3/2: 0.0189, 2: 0.0016}
# citeseer: {0: 0.3659, 1/2: 0.3538, 1: 0.1699, 3/2: 0.0678, 2: 0.0288, 5/2: 0.0102, 3: 0.0030, 7/2: 0.0005, 4: 0.0001}
# pubmed: {0: 0.4239, 1/2: 0.4549, 1: 0.1094, 3/2: 0.0112, 2: 0.0006}