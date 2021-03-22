import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from class_Layer import GraphConvolution



class DNN(nn.Module):

    def __init__(self, num_genes, num_classes):
        super(DNN, self).__init__()

        h1 = nn.Linear(num_genes, 1024)
        h2 = nn.Linear(1024, 1024)
        h3 = nn.Linear(1024, num_classes)
        bn = nn.BatchNorm1d(1024)
        relu = nn.ReLU() 
        
        self.hid = nn.Sequential(
            h1, bn, relu,
            h2, bn, relu,
            h2, bn, relu,
            h3
        )
        
        self.hid = self.hid.cuda()

    def forward(self, x):
        x = self.hid(x)

        return x


    
class DNN_decreasing(nn.Module):

    def __init__(self, num_genes, num_classes):
        super(DNN_decreasing, self).__init__()

        h1 = nn.Linear(num_genes, 4096)
        h2 = nn.Linear(4096, 2048)
        h3 = nn.Linear(2048, 1024)
        h4 = nn.Linear(1024, 512)
        h5 = nn.Linear(512, 256)
        h6 = nn.Linear(256, 128)
        h7 = nn.Linear(128, 64)
        dropout = nn.Dropout(p=0.5)
        
        bn = []
        for i in range(7):
            bn.append(nn.BatchNorm1d(2**(12 - i)))
        
        hfinal = nn.Linear(64, num_classes)
        relu = nn.ReLU() 
        
        self.hid = nn.Sequential(
            h1, bn[0], relu, dropout,
            h2, bn[1], relu, dropout,
            h3, bn[2], relu, dropout,
            h4, bn[3], relu, dropout,
            h5, bn[4], relu, dropout,
            h6, bn[5], relu, dropout,
            h7, bn[6], relu, dropout,
            hfinal
        )
        
        self.hid = self.hid.cuda()

    def forward(self, x):
        x = self.hid(x)

        return x


class GCN(nn.Module):

    def __init__(self, node_dim, input_dim, output_dim, hidden_dim):
        super(GCN, self).__init__()

        self.gcn = GraphConvolution(input_dim, hidden_dim)
        self.node_dim = node_dim
        self.fc1 = nn.Linear(node_dim*hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x, network = inputs
        x = self.gcn((x, network))

        flat = []
        for i in x:
            flat.append(torch.flatten(i))
        # print('flatten size 111')
        # print(flat)

        flat = torch.stack(flat)
        # print('flatten size 222')
        # print(flat)

        x = self.relu(flat)
        x = self.fc1(x)

        return x


class GCN_2(nn.Module):

    def __init__(self, node_dim, input_dim, output_dim, hidden_dim):
        super(GCN_2, self).__init__()

        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.node_dim = node_dim
        self.fc1 = nn.Linear(node_dim*hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(node_dim)

    def forward(self, inputs):
        x, network = inputs
        # print('x0')
        # print(len(x))
        # print(x)
        # for i in x:
        #     print('lego')
        #     print(len(i))
        #     print(i)

        x = self.gcn1((x, network))
        # print('x')
        # print(len(x))
        # print(x)
        # for i in x:
        #     print('lego')
        #     print(len(i))
        #     print(i)
        x = torch.stack(x)
        x = self.bn1(x)
        # print('no problem?')

        x = self.gcn2((x, network))

        flat = []
        for i in x:
            flat.append(torch.flatten(i))
        # print('flatten size 111')
        # print(flat)

        flat = torch.stack(flat)
        # print('flatten size 222')
        # print(flat)

        # x = self.relu(flat)
        x = self.fc1(flat)

        return x





