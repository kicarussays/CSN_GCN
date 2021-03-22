"""
Written on December 29, 2020 by Junmo Kim

    이 코드는 CrossEntropyLoss를 이용하여 암의 Subtype을 분류합니다.

        1. 데이터 로드 및 처리
        2. 네트워크 로드
        3. 학습
        4. 정확도 계산

"""


from class_DeepNetwork import DNN, DNN_decreasing
from import_data import CLINICAL_DATA_LOAD, DATA_JOIN, KFOLD, NORMALIZATION
from configuration import *
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


if torch.cuda.is_available():
    print("Let's go CUDA!!!!!")
cuda = torch.device('cuda')


def DataLoad(clinical_filepath, gene_data, mode):
    """
        clinical_filepath: 환자 id, 임상정보를 가지고 있는 데이터 경로
        gene_data: 환자 id, 유전자 정보를 가지고 있는 데이터 경로
        mode: 정규화 방식 (Standard or Minmax)

    """
    # Loading clinical data
    status = CLINICAL_DATA_LOAD(clinical_filepath)

    # Loading X, Y data / # of classes / # of variables(genes)
    training, num_classes = DATA_JOIN(status, gene_data, mode)
    num_vars = len(training[0]) - 1
    
    return training, num_vars, num_classes

def toTensor(training, batch_size, shuffle):
    # Convert numpy array to Tensor
    x = torch.Tensor(training[:, :-1]).to(cuda)
    y = torch.LongTensor(training[:, -1]).to(cuda)

    # Convert raw Tensor to DataLoader (This process is for minibatch)
    ds = TensorDataset(x, y)
    trainloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return trainloader


def Parameters(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    return criterion, optimizer


training, num_vars, num_classes = DataLoad(clinical_filepath, gene_data, mode)


# 여기는 GCN 포함
if gcn_mode:
    from net_emerging import gf
    gf = NORMALIZATION(gf, 'standard')
    training = np.transpose(np.vstack((np.transpose(gf), [training[:, -1]])))


batch_size = 32
kfold = 5

np.random.seed(777)
torch.manual_seed(777)

np.random.shuffle(training)
_split_1 = KFOLD(training, kfold)
test = _split_1[0]

_train_tmp = []
for i in range(1, 5):
    _train_tmp.append(_split_1[i])
train = np.vstack(_train_tmp)

# thres = int(0.8 * len(training))
# train = training[:thres]
# test = training[thres:]

split = KFOLD(train, kfold)

epoch_list = range(100, 2000, 100)


acc_box = []

for ep in epoch_list:
    acc = 0
    for i in range(kfold): # 5-fold validation
        torch.cuda.empty_cache()
        
        val_set = split[i]
        train_set = []

        for j in range(kfold):
            if j != i:
                train_set.append(split[j])

        train_set = np.vstack(train_set)
        
        
        trainloader = toTensor(train_set, batch_size, True)
        testloader = toTensor(val_set, batch_size, False)

        net = DNN_decreasing(num_vars, num_classes)
        criterion, optimizer = Parameters(net)

        net.to(cuda)
        net.train()

        for epoch in range(ep):

            running_loss = 0.

            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                inputs = inputs.to(cuda)
                labels = labels.to(cuda)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 20 == 19:  # print every 2000 mini-batches
                    # print('[%d, %5d] loss: %.3f' %
                    #       (epoch + 1, i + 1, running_loss / batch_size))
                    running_loss = 0.0
            
#         PATH = './path/cifar_net_' + str(ep) + '_epoch.pth'
#         torch.save(net.state_dict(), PATH)
        
#         net = DNN(num_vars, num_classes)
#         net.load_state_dict(torch.load(PATH))
        net.eval()
        net.to(cuda)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(cuda)
                labels = labels.to(cuda)
                
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

#         print('Accuracy: %d %%' % (100 * correct / total))
        
        acc += 100 * correct / total
        

        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1


#         for i in range(num_classes):
#             print('Accuracy of %5s : %2d %%' % (
#                 i, 100 * class_correct[i] / class_total[i]))
        
    print('Epoch: %d' % ep)
    print('Accuracy: %.2f %%' % (acc / num_classes))
    acc_box.append(acc / num_classes)


trainloader = toTensor(train, batch_size, True)

net = DNN_decreasing(num_vars, num_classes)
criterion, optimizer = Parameters(net)

net.to(cuda)


ep = acc_box[acc_box.index(max(acc_box))]

print('ep: %d' % ep)

for epoch in range(ep):
    running_loss = 0.

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs = inputs.to(cuda)
        labels = labels.to(cuda)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %f' % (epoch + 1, i + 1, running_loss / batch_size*100))
            running_loss = 0.0
        
PATH = './path/cifar_net_' + str(ep) + '_epoch.pth'
torch.save(net.state_dict(), PATH)
                  

testloader = toTensor(test, batch_size, False)

net = DNN_decreasing(num_vars, num_classes)
net.load_state_dict(torch.load(PATH))
net.eval()
net.to(cuda)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(cuda)
        labels = labels.to(cuda)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (100 * correct / total))


class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(num_classes):
    print('Accuracy of %5s : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))


