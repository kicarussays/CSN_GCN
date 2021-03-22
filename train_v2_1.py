"""
Written on January 15, 2020 by Junmo Kim

    이 코드는 CrossEntropyLoss를 이용하여 암의 Subtype을 분류합니다.
    GCN을 사용하여 학습을 진행합니다.

"""

from class_DeepNetwork import DNN, DNN_decreasing, GCN
from import_data import CLINICAL_DATA_LOAD, DATA_JOIN, KFOLD, NORMALIZATION, KFOLD_GCN
from configuration import *
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

if torch.cuda.is_available():
    print("Let's go CUDA!!!!!")
cuda = torch.device('cuda')


np.random.seed(777)
torch.manual_seed(777)
batch_size = 2
kfold = 5


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


def toTensor(training, batch_size, shuffle):
    """
        Tensor 변환
        변환 대상: Feature, Laplacian Mat(support)

        Return:
            trainloader: 파이토치 모델에 넣기 위한 객체

    """

    support2 = np.array(np.transpose(training)[1])
    put = []
    for i in support2:
        put.append(i.astype(float))

    support = np.array(put)
    # print('support')
    # print(support)


    _training = np.vstack(np.transpose(training)[0])
    # print('training')
    # print(_training)

    # Convert numpy array to Tensor
    x = torch.Tensor(_training[:, :-1]).to(cuda)
    # x = torch.unsqueeze(x, -1)
    support = torch.Tensor(support).to(cuda)
    y = torch.LongTensor(_training[:, -1]).to(cuda)




    # Convert raw Tensor to DataLoader (This process is for minibatch)
    ds = TensorDataset(x, support, y)
    trainloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return trainloader


def Parameters(net):
    """
        신경망으로부터 파라미터 받아오는 함수

    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    return criterion, optimizer


# 여기는 주석처리 해야
clinical_filepath = 'data/Subtype_Info.csv'
gene_data = 'data/GEM_fs_0001.csv'

# 데이터 불러오기
training, num_vars, num_classes = DataLoad(clinical_filepath, gene_data, mode)

# 라플라시안 행렬 불러오고, 임의추출하기 위해서 GEM과 튜플로 묶어서 저장
net_feature = []
cnt = 0
cls = 0
for i in range(len(training)):

    ### 여기는 지워야됨
    j = np.random.randint(980)
    if training[j][-1] == cls:

        if cnt <= 5:
            cnt += 1

            path = 'Laplacian/Laplacian_' + str(j) + '.npy'
            support = np.load(path).astype(float)
            net_feature.append([training[j], support])
        else:
            print('클래스 변경')
            cls += 1
            cnt = 0

    if cls == 5:
        break




    #### 여기는 살려야함

    # path = 'Laplacian/Laplacian_' + str(i) + '.npy'
    # support = np.load(path).astype(float)
    # net_feature.append([training[i], torch.Tensor(support)])
    # if i % 20 == 0:
    #     print('%d / 981 Laplacian Loaded' % i)
    #
    # if i == 10:
    #     break

####

# Train / Test Set Split
np.random.shuffle(net_feature)
# print('random net feature')
# print(net_feature)
# print('transpose')
# print(np.transpose(net_feature))

_split_1 = KFOLD_GCN(net_feature, num_classes, kfold)

test = _split_1[0]

_train_tmp = []
for i in range(1, 5):
    _train_tmp.append(_split_1[i])
train = np.vstack(_train_tmp)



### 제거해야함
trainloader = toTensor(train, batch_size, True)
testloader = toTensor(test, batch_size, False)

net = GCN(num_vars, 1, num_classes, 8)
criterion, optimizer = Parameters(net)

net.to(cuda)
net.train()

for epoch in range(10):
    running_loss = 0.

    for i, data in enumerate(trainloader):
        inputs, support, labels = data
        inputs = inputs.unsqueeze(-1)

        inputs = inputs.to(cuda)
        support = support.to(cuda)
        labels = labels.to(cuda)

        optimizer.zero_grad()

        outputs = net((inputs, support))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('%d Epoch / Loss: %.3f' %(epoch, loss.item()))





###







