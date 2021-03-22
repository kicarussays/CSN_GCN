import networkx as nx
from scipy import sparse
import numpy as np
import torch
import time

path = '/home/kicarus/PycharmProjects/2021CSN/GCNencoding/sparse_1.npz'
support = sparse.load_npz(path)
start = time.time()
tmp = nx.from_scipy_sparse_matrix(support)

end = time.time() - start

nx.draw(tmp)


def edge_to_remove(graph):
  G_dict = nx.edge_betweenness_centrality(graph)
  edge = ()

  # extract the edge with highest edge betweenness centrality score
  for key, value in sorted(G_dict.items(), key=lambda item: item[1], reverse = True):
      edge = key
      break

  return edge


























