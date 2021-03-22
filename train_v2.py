"""
Written on January 15, 2020 by Junmo Kim

    이 코드는 CrossEntropyLoss를 이용하여 암의 Subtype을 분류합니다.
    GCN을 사용하여 학습을 진행합니다.

"""

from class_DeepNetwork import DNN, DNN_decreasing, GCN, GCN_2
from import_data import DataLoad, KFOLD_GCN, SparseLoad
from configuration import *
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm


if torch.cuda.is_available():
    print("Let's go CUDA!!!!!")
cuda = torch.device('cuda')

torch.cuda.empty_cache()
np.random.seed(777)
torch.manual_seed(777)

batch_size = args.batch
kfold = args.kfold
filter = args.filter


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


def Parameters(net):
    """
        신경망으로부터 파라미터 받아오는 함수

    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    return criterion, optimizer


def Split(net_feature):
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
    np.random.shuffle(train)

    # Train / Validation Set Split
    split = KFOLD_GCN(train, num_classes, kfold)

    return train, test, split

# 데이터 불러오기
training, num_vars, num_classes = DataLoad(clinical_filepath, gene_data, mode)
net_feature = SparseLoad(training)

# Train / Test Set Split
train, test, split = Split(net_feature)


epoch_list = range(100, 2100, 100)   # 에폭은 100부터 2000까지 100단위로
epoch_list = [10, 20, 30, 40, 50]

acc_box = []
for ep in epoch_list:
    acc = 0
    for i in range(kfold):  # 5-fold validation
        torch.cuda.empty_cache()

        # Validation Set
        val_set = split[i]
        train_set = []

        for j in range(kfold):
            if j != i:
                train_set.append(split[j])

        # Training Set
        train_set = np.vstack(train_set)

        trainloader = toTensor(train_set, batch_size, True)
        testloader = toTensor(val_set, batch_size, False)

        net = GCN(num_vars, 1, num_classes, filter)
        criterion, optimizer = Parameters(net)

        net.to(cuda)
        net.train()

        for epoch in range(ep):
            torch.cuda.empty_cache()

            running_loss = 0.
            tcnt = 0

            for k, data in enumerate(trainloader):

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

                tcnt += len(data)
                running_loss += loss.item()

        print('Epoch: %5d / Final Loss: %5.5f' % (ep, running_loss / tcnt))

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

                # print('Accuracy: %d %%' % (100 * correct / total))

        acc += 100 * correct / total


    print('Epoch: %d' % ep)
    print('Accuracy: %.3f %%' % (acc / num_classes))
    acc_box.append(acc / num_classes)

print('')
print('k-Fold Finish.')
print('')



trainloader = toTensor(train, batch_size, True)

net = GCN(num_vars, 1, num_classes, filter)
criterion, optimizer = Parameters(net)

net.to(cuda)
net.train()

ep = epoch_list[acc_box.index(max(acc_box))]
print('ep: %d' % ep)

print('This is Last Learning')
for epoch in tqdm(range(ep)):
    running_loss = 0.
    tcnt = 0

    for i, data in enumerate(trainloader):

        inputs, support, labels = data
        inputs = inputs.unsqueeze(-1)

        inputs = inputs.to(cuda)
        labels = labels.to(cuda)

        optimizer.zero_grad()

        outputs = net((inputs, support))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        tcnt += len(data)
        running_loss += loss.item()

print('Epoch: %5d / Final Loss: %5.5f' % (ep, running_loss / tcnt))


PATH = './path/GCN_' + str(ep) + '_epoch.pth'
torch.save(net.state_dict(), PATH)

testloader = toTensor(test, batch_size, False)

net = GCN(num_vars, 1, num_classes, filter)
net.load_state_dict(torch.load(PATH))
net.eval()
net.to(cuda)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, support, labels = data
        inputs = inputs.unsqueeze(-1)

        inputs = inputs.to(cuda)
        labels = labels.to(cuda)

        outputs = net((inputs, support))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %5.3f %%' % (100 * correct / total))

class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

with torch.no_grad():
    for data in testloader:
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
    print('Accuracy of %5s : %5.3f %%' % (
        i, 100 * class_correct[i] / class_total[i]))





