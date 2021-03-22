"""
    Fake data 멀티프로세싱 작업

"""


import louvain.community as community_louvain
import networkx as nx
from scipy import sparse
import numpy as np
import pandas as pd
import os
import multiprocessing

which = 'train'
filelist = os.listdir(which)
print('lets go~!')

def fake_making(f):
    start_pt = f*10
    for i in range(start_pt, start_pt + 10):
        if i >= len(filelist):
            break

        path = which + '/' + filelist[i]

        raw = sparse.load_npz(path)
        a1 = np.array(raw.todense())
        tmp = nx.convert_matrix.from_numpy_array(a1)

        # compute the best partition
        # print('compute the best partition')
        nodes = tmp.nodes
        edges = tmp.edges
        nx_part = community_louvain.best_partition(tmp)
        comm = set(list(nx_part.values()))

        # Exclude each community
        # print('Exclude each community...')
        cls_part = []
        for j in comm:
            cls_part.append(list({k: v for k, v in nx_part.items() if v != j}.keys()))


        graph_save = []
        # print('Adj making...')
        # print('# of Community: %d' %(len(cls_part)))
        for num, j in enumerate(cls_part):
            if len(cls_part) > 30:
                # print('Too many communities')
                break

            # New Adjacent Matrix: A
            gtmp = nx.Graph()
            putedge = [value for value in edges if value[0] in j and value[1] in j]
            gtmp.add_nodes_from(nodes)
            gtmp.add_edges_from(putedge)
            A = nx.convert_matrix.to_numpy_array(gtmp, dtype=float)

            # Renormalization Laplacian Trick
            A = A + np.eye(len(A))  # Adj Mat + Identity Mat
            # D'^(-1/2) A' D'^(-1/2)
            Degree = [sum(k) for k in A]

            A = np.transpose(A)
            for k in range(len(A)):
                A[k] = Degree[k] ** (-0.5) * A[k]
            A = np.transpose(A)
            for k in range(len(A)):
                A[k] = Degree[k] ** (-0.5) * A[k]
            A = A.astype(float)

            # Save as Sparse Matrix
            spar = sparse.coo_matrix(A)
            savepath = 'fake/' + which + '_' + str(i) + '_' + str(num) + '.npz'

            sparse.save_npz(savepath, spar)

            if num % 10 == 0:
                print('%5d / %5d     Complete' %(num + 1, len(cls_part)))


num_list = range(0, (len(filelist) // 10) + 1)
print('# of files: {}'.format(len(filelist)))

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=25)
    pool.map(fake_making, num_list)
    pool.close()
    pool.join()









