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
from scipy import sparse
import numpy as np
import time

if torch.cuda.is_available():
    print("Let's go CUDA!!!!!")
cuda = torch.device('cuda')


np.random.seed(777)
torch.manual_seed(777)
batch_size = 3
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

    support = np.transpose(training)[1]
    _training = np.vstack(np.transpose(training)[0])

    # Convert numpy array to Tensor
    x = torch.Tensor(_training[:, :-1])
    y = torch.LongTensor(_training[:, -1])

    slice_num = len(training) // batch_size
    trainloader = []
    for i in range(slice_num + 1):
        if slice_num == 0:
            trainloader.append((x, support, y))
            break

        if i != slice_num:
            j = i * batch_size
            trainloader.append((x[j:j+batch_size], support[j:j+batch_size], y[j:j+batch_size]))
        else:
            j = slice_num * batch_size
            trainloader.append((x[j:], support[j:], y[j:]))

    if shuffle:
        np.random.shuffle(trainloader)

    return trainloader


def toSparse(mat):
    values = mat.data
    indices = np.vstack((mat.row, mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = torch.Size(mat.shape)

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(cuda)


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

start = time.time()
for i in range(len(training)):
    j = np.random.randint(980)
    if training[j][-1] == cls:
        if cnt <= 5:
            cnt += 1
            path = 'Sparse/sparse_' + str(j) + '.npz'
            support = sparse.load_npz(path)
            support = toSparse(support)
            net_feature.append([training[j], support])
        else:
            print('class change')
            cls += 1
            cnt = 0

    if cls == 5:
        break



# print(net_feature)


print('데이터 로드 소요시간: %.2f' % ((time.time() - start) / 60))


# Train / Test Set Split
np.random.shuffle(net_feature)
# print('random net feature')
# print(net_feature)
# print('transpose')
# print(np.transpose(net_feature))

_split_1 = KFOLD_GCN(net_feature, num_classes, kfold)

test = _split_1[0]
print('test')
# print(test)


_train_tmp = []
for i in range(1, 5):
    _train_tmp.append(_split_1[i])
train = np.vstack(_train_tmp)
np.random.shuffle(train)
print('train')
# print(train)



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
        # support = support.to(cuda)
        labels = labels.to(cuda)

        optimizer.zero_grad()

        outputs = net((inputs, support))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('%d Epoch / Loss: %.3f' %(epoch, loss.item()))

net.eval()
net.to(cuda)

correct = 0
total = 0
with torch.no_grad():
    for k, data in enumerate(testloader):
        inputs, support, labels = data
        inputs = inputs.unsqueeze(-1)

        inputs = inputs.to(cuda)
        labels = labels.to(cuda)

        outputs = net((inputs, support))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Accuracy: %d %%' % (100 * correct / total))



class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

with torch.no_grad():
    for k, data in enumerate(testloader):
        inputs, support, labels = data
        inputs = inputs.unsqueeze(-1)
        inputs = inputs.to(cuda)
        labels = labels.to(cuda)

        outputs = net((inputs, support))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))

