import numpy as np
from scipy import sparse


for i in range(0, 100):
    path = 'Laplacian/Laplacian_' + str(i) + '.npy'
    path = '/home/kicarus/PycharmProjects/2021CSN/GCNencoding/data/Laplacian_0.npy'
    x1 = np.load(path)

    spar = sparse.coo_matrix(x1)
    savepath = 'Sparse/sparse_' + str(i) + '.npz'
    sparse.save_npz(savepath, spar)