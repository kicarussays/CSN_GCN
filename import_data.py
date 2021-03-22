"""
Written on October 29, 2020
This provide train, test data for learning.

- Data format
    -> clinical data:
        data: person_id, clinical information
        label: cancer type, subtype, etc (labels should be located at the last column)
    -> gene data: person_id, gene information
        data: person_id, gene information
        label: same as clinical data


Revised on November 9, 2020
    - Added survival data join function

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import sparse
import time
from tqdm import tqdm
import torch
import os
import sys


def CLINICAL_DATA_LOAD(clinical_filepath):
    tmp = np.loadtxt(clinical_filepath, dtype=str, delimiter=',')
    STATUS = pd.DataFrame({'SAMPLE': np.transpose(tmp)[0], 'class': np.transpose(tmp)[-1]})
    # NUM_CLASSES = len(set(np.transpose(tmp)[-1]))  # For output layer

    return STATUS


def NORMALIZATION(data, mode):
    if mode == 'standard':
        scaler = StandardScaler()
    if mode == 'minmax':
        scaler = MinMaxScaler()

    normalized = scaler.fit_transform(data)

    return normalized


def DATA_JOIN(status, gene_data, mode):
    """
    - This function join clinical data and gene_data on person id
    - Normalization mode is required
    """
    print('Gene data [{}] loading...'.format(gene_data))
    data = np.transpose(np.loadtxt(gene_data, dtype=str, delimiter=','))
    print('Size: %s' % str(data.shape))
    print('Done')
    data[0, 0] = 'SAMPLE'  # For joining between clinical data and gene data

    print('Data Merging...')
    data = pd.DataFrame(data=data[1:], columns=data[0])
    data = pd.merge(data, status, how='inner', on='SAMPLE')
    data = np.delete(np.array(data), 0, 1)
    data = np.array(data, dtype=float)
    print('Done')

    # print('Data discription...')
    # datadisc = [['Da']]

    data = np.transpose(data)
    puredata = np.transpose(data[:-1])
    labels = data[-1]

    normalized = np.transpose(NORMALIZATION(puredata, mode))

    concatenate = np.concatenate((normalized, [labels]))
    data = np.transpose(concatenate)
    num_classes = len(set(concatenate[-1]))

    clscnt = [0] * num_classes
    for i in labels:
        clscnt[int(i)] += 1

    print('========================= Data Discription ==========================')
    print('')
    print('         Joined Size: %10s' % str(data.shape))
    print('         # of classes: %8d' % num_classes)
    print('         # of variables: %9d' % (len(data[0])-1))
    for i in range(num_classes):
        print('         # of %d class: %10d' % (i, clscnt[i]))
    print('')
    print('=====================================================================')

    return data, num_classes


def COMBINE(data1, data2):
    labels = np.array(np.transpose(data1)[-1])
    labels = np.transpose(np.expand_dims(labels, axis=0))

    gem = np.delete(data1, -1, 1)
    ndm = np.delete(data2, -1, 1)
    union = np.concatenate((gem, ndm, labels), axis=1)

    return union


def KFOLD(DATA, KFOLD):
    NUM_CLASSES = len(set(np.transpose(DATA)[-1]))

    CLS_TMP = list()
    for i in range(NUM_CLASSES):
        clssp = []
        for j in DATA:
            if int(j[-1]) == i:
                clssp.append(j)
        clssp = np.array(clssp)
        CLS_TMP.append(clssp)

    def KFOLD_SEPARATION(sample):
        getsam = sample
        kfoldset = []
        for i in range(KFOLD):
            putnum = len(getsam) // (KFOLD - i)
            wtput = getsam[:putnum]
            kfoldset.append(wtput)
            getsam = getsam[putnum:]

        return kfoldset

    SEPARATED_SET = []
    for i in CLS_TMP:
        SEPARATED_SET.append(KFOLD_SEPARATION(i))
    FINAL_SET = []
    for i in np.transpose(SEPARATED_SET):
        tmp1 = i[0]
        for j in range(len(i) - 1):
            tmp1 = np.concatenate((tmp1, i[j + 1]))
        FINAL_SET.append(tmp1)

    return FINAL_SET


def KFOLD_GCN(DATA, NUM_CLASSES, KFOLD):
    CLS_TMP = list()
    for i in range(NUM_CLASSES):
        clssp = []
        for j in DATA:
            if int(j[0][-1]) == i:
                clssp.append(j)
        clssp = np.array(clssp)
        CLS_TMP.append(clssp)
    #
    # print('데이터 개수')
    # print(len(DATA))
    #
    # for i in range(NUM_CLASSES):
    #     print('%d 클래스별' % i)
    #     print(CLS_TMP[i])


    def KFOLD_SEPARATION(sample):
        getsam = sample
        kfoldset = []
        for i in range(KFOLD):
            putnum = len(getsam) // (KFOLD - i)
            wtput = getsam[:putnum]
            kfoldset.append(wtput)
            getsam = getsam[putnum:]

        return kfoldset

    SEPARATED_SET = []
    for i in CLS_TMP:
        SEPARATED_SET.append(KFOLD_SEPARATION(i))
    # print('seprate')
    # print(SEPARATED_SET)
    # print('tp')
    # print(np.transpose([SEPARATED_SET[0]]))

    FINAL_SET = []
    for i in np.transpose(SEPARATED_SET):
        FINAL_SET.append(np.vstack(i))

    return FINAL_SET


def SparseLoad(training):
    # 라플라시안 행렬 불러오고, 임의추출하기 위해서 GEM과 튜플로 묶어서 저장
    net_feature = []
    # cnt = 0
    # cls = 0

    start = time.time()
    for i in tqdm(range(len(training))):
        spend = 0.
        path = 'Sparse/sparse_' + str(i) + '.npz'
        support = sparse.load_npz(path)
        support = toSparse(support)

        net_feature.append([training[i], support])

    print('데이터 로드 소요시간: %.2f' % ((time.time() - start) / 60))
    return net_feature


def FakeLoad(training):
    net_feature = []
    fake_list = os.listdir('fake')

    start = time.time()
    for i in tqdm(range(len(training))):
        spend = 0.
        pathlist = list(filter(lambda x: 'fake' + str(i) + '_' in x, fake_list))
        if len(pathlist) == 0:
            # print('No fake data')
            continue
        for path in pathlist:
            # print(path)
            support = sparse.load_npz('fake/' + path)
            support = toSparse(support)
            net_feature.append([training[i], support])
        sys.stdout.flush()

        if i == 10:
            break

    print('Fake 데이터 로드 소요시간: %.2f' % ((time.time() - start) / 60))
    return net_feature


def toSparse(mat):
    values = mat.data
    indices = np.vstack((mat.row, mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mat.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def DataLoad(clinical_filepath, gene_data, mode):
    """
        clinical_filepath: 환자 id, 임상정보를 가지고 있는 데이터 경로
        gene_data: 환자 id, 유전자 정보를 가지고 있는 데이터 경로
        mode: 정규화 방식 (Standard or Minmax)

        Return:
            training: Gene, Subtype이 결합된 데이터
            num_vars: 변수의 개수
            num_classes: 클래스의 개수

    """
    # Loading clinical data
    status = CLINICAL_DATA_LOAD(clinical_filepath)

    # Loading X, Y data / # of classes / # of variables(genes)
    training, num_classes = DATA_JOIN(status, gene_data, mode)
    num_vars = len(training[0]) - 1

    return training, num_vars, num_classes












