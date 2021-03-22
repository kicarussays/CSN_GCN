import torch
import torch.nn as nn

cuda = torch.device('cuda')

class GraphConvolution(nn.Module):

    def __init__(self, feature_dim, hidden_dim):
        super(GraphConvolution, self).__init__()

        self.weight = nn.Parameter(torch.randn(feature_dim, hidden_dim))
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, inputs):
        x, network = inputs
        # print('x')
        # print(x.size())
        # print(x)
        # print('network')
        # print(network)

        x = self.dropout(x)
        xw = torch.matmul(x, self.weight)

        # print('xwsize')
        # print(xw.size())
        # print(xw)


        out = []
        for i, data in enumerate(xw):
            out.append(torch.sparse.mm(network[i].to(cuda), data))

        return out



