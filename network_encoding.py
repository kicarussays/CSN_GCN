import numpy as np
import pandas as pd


GCN_ENCODED_MATRIX = []
# GEM = pd.read_csv('/home/kicarus/PycharmProjects/2021CSN/GCNencoding/data/GEM_fs_0001.csv')
GEM = pd.read_csv('GEM_fs_0001.csv')
GEM = np.array(GEM)[:, 1:]

print("Let's get started..")
print("please..")

for i in range(0, 100):
    # path = '/home/kicarus/PycharmProjects/2021CSN/GCNencoding/data/9th_csn_BRCA.csv'
    path = str(i) + 'th_csn_BRCA.csv'
    A = pd.read_csv(path, header=None) # Adj matrix
    A = np.array(A)
    A = A + np.eye(len(A)) # Adj Mat + Identity Mat

    # D'^(-1/2) A' D'^(-1/2)
    Degree = []
    for j in A:
        Degree.append(sum(j))

    A = np.transpose(A)
    for j in range(len(A)):
        A[j] = Degree[j]**(-0.5) * A[j]

    A = np.transpose(A)
    for j in range(len(A)):
        A[j] = Degree[j]**(-0.5) * A[j]

    GCN_ENCODED_MATRIX.append(np.matmul(A, GEM[:, i]))
    if (i+1) % 10 == 0:
        print('%d / 981 completed' % (i+1))
        print('filename: %s' % path)

print('Job Complete')
GCN_ENCODED_MATRIX = np.array(GCN_ENCODED_MATRIX).astype(float)
np.save('GCN_ENCODED_MATRIX.npy', GCN_ENCODED_MATRIX)

print('Matrix Saved; File name : GCN_ENCODED_MATRIX.npy')


