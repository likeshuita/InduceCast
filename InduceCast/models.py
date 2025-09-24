import torch
import torch.nn as nn
import torch.nn.functional as F
import functions as fn
import copy
from torch_cluster import random_walk
import numpy as np

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations):
        super(FC, self).__init__()
        self.units = units
        self.activations = activations
        self.linear = nn.Linear(input_dims, units)

    def forward(self, inputs):
        x = self.linear(inputs)
        if self.activations is not None:
            x = self.activations(x)
        return x


class MAB(nn.Module):
    def __init__(self, K, d, input_dim, output_dim):
        super(MAB, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=11, units=D, activations=F.relu)
        self.FC_k = FC(input_dims=11, units=D, activations=F.relu)
        self.FC_v = FC(input_dims=11, units=D, activations=F.relu)
        self.FC = FC(input_dims=D, units=11, activations=F.relu)

    def forward(self, Q, K, batch_size):
        query = self.FC_q(Q)
        key = self.FC_k(K)
        value = self.FC_v(K)

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention_weights = F.softmax(attention, dim=-1)
        result = torch.matmul(attention_weights, value)

        result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)
        result = self.FC(result)

        return result, attention_weights


class BottleAttention(nn.Module):
    def __init__(self, K, d, set_dim):
        super(BottleAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.set_dim = set_dim
        self.I = nn.Parameter(torch.Tensor(1, 1, set_dim, 11))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(K, d, 11, 11)
        self.mab1 = MAB(K, d, 11, 11)

    def forward(self, X):
        batch_size = X.shape[0]
        X = X.reshape(batch_size, -1, 11)
        X = X.unsqueeze(1)

        I = self.I.repeat(batch_size, 1, 1, 1)
        H, attn_weights_0 = self.mab0(I, X, batch_size)
        result, attn_weights_1 = self.mab1(X, H, batch_size)
        result = result.squeeze(1)

        return result, attn_weights_0, attn_weights_1


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch, seq, feature]
        x = x.transpose(1, 2)  # Convert to [batch, feature, seq]
        x = self.network(x)
        x = x.transpose(1, 2)  # Convert back to [batch, seq, feature]
        return x


class BAnet(nn.Module):
    def __init__(self, a_sparse, seq=12, kcnn=2, k=6, m=2, walk_length=5):
        super(BAnet, self).__init__()
        self.feature = seq
        self.seq = seq - kcnn + 1
        self.alpha = 0.5
        self.m = m

        self.a_sparse = a_sparse.to(device)
        self.nodes = a_sparse.shape[0]
        self.walk_length = walk_length
        self.device = device


        self.create_random_walk_matrix()


        self.rw_transform = nn.Linear(seq, seq)
        self.rw_attention = nn.Linear(seq, seq)


        self.conv2d = nn.Conv2d(1, 1, (kcnn, 2))


        self.gcn_inducing = nn.Linear(in_features=self.seq, out_features=self.seq)


        self.bottle_attention1 = BottleAttention(
            K=11,
            d=1,
            set_dim=32
        )
        self.bottle_attention2 = BottleAttention(
            K=11,
            d=1,
            set_dim=32
        )

        # 特征投影层，对应论文公式12和13
        self.feature_proj1 = nn.Linear(in_features=self.seq, out_features=self.seq)
        self.feature_proj2 = nn.Linear(in_features=self.seq, out_features=self.seq)

        # TCN与全连接层
        num_channels = [m, m]  # 两层TCN
        self.tcn = TCN(
            num_inputs=m,
            num_channels=num_channels,
            kernel_size=3,
            dropout=0.2
        )
        self.fc1 = nn.Linear(in_features=self.seq - 1, out_features=k)
        self.fc2 = nn.Linear(in_features=k, out_features=m)
        self.fc3 = nn.Linear(in_features=k + m, out_features=1)
        self.decoder = nn.Linear(self.seq, 1)

        # Activation
        self.dropout = nn.Dropout(p=0.5)
        self.LeakyReLU = nn.LeakyReLU()


        adj1 = copy.deepcopy(self.a_sparse.to_dense())
        adj2 = copy.deepcopy(self.a_sparse.to_dense())
        for i in range(self.nodes):
            adj1[i, i] = 0.000000001
            adj2[i, i] = 0
        degree = 1.0 / (torch.sum(adj1, dim=0))
        degree_matrix = torch.zeros((self.nodes, self.feature), device=device)
        for i in range(12):
            degree_matrix[:, i] = degree
        self.degree_matrix = degree_matrix
        self.adj2 = adj2

    def create_random_walk_matrix(self):

        indices = self.a_sparse.indices()
        row, col = indices[0].to(self.device), indices[1].to(self.device)


        start = torch.arange(self.nodes, device=self.device)


        walk = random_walk(row, col, start, walk_length=self.walk_length)


        self.rw_adj = torch.zeros((self.nodes, self.nodes), device=self.device)
        self.rw_adj = torch.scatter(self.rw_adj, 1, walk, 1).to_sparse()


        degree = torch.sparse.sum(self.rw_adj, dim=1).to_dense()
        degree = torch.pow(degree, -1)
        degree[torch.isinf(degree)] = 0
        D_inverse = torch.diag(degree, diagonal=0).to_sparse()


        self.rw_adj = torch.sparse.mm(D_inverse, self.rw_adj)

    def random_walk_enhance(self, x, feature_type):
        # x shape: [batch_size, nodes, seq]
        batch_size = x.size(0)


        enhanced_features = []

        for i in range(batch_size):

            current_features = x[i]  # [nodes, seq]


            rw_features = torch.matmul(self.rw_adj.to_dense(), current_features)


            attention_weights = torch.sigmoid(self.rw_attention(rw_features))


            transformed_features = self.rw_transform(rw_features)


            if feature_type == 'occ':

                enhanced = current_features + attention_weights * transformed_features
            else:  # prc

                enhanced = current_features * (1 + attention_weights * transformed_features)

            enhanced_features.append(enhanced)


        return torch.stack(enhanced_features), None

    def forward(self, occ, prc):
        b, n, s = occ.shape


        occ_enhanced, _ = self.random_walk_enhance(occ, 'occ')
        prc_enhanced, _ = self.random_walk_enhance(prc, 'prc')

        data = torch.stack([occ_enhanced, prc_enhanced], dim=3).reshape(b * n, s, -1).unsqueeze(1)


        h_conv = self.conv2d(data)
        h_conv = h_conv.squeeze().reshape(b, n, -1)

        h1, attn_weights_01, attn_weights_11 = self.bottle_attention1(h_conv)


        z1 = self.dropout(self.LeakyReLU(self.feature_proj1(h1)))

        h2, attn_weights_02, attn_weights_12 = self.bottle_attention2(z1)
        z2 = self.dropout(self.LeakyReLU(self.feature_proj2(h2)))

        inner_fusion = (1 - self.alpha) * z1 + self.alpha * h_conv
        outer_fusion = (1 - self.alpha) * z2 + self.alpha * inner_fusion
        occ_conv = outer_fusion
        occ_conv1 = occ_conv.view(b * n, self.seq)
        occ_conv2 = occ_conv1
        x = torch.stack([occ_conv1, occ_conv2], dim=2)
        tcn_out = self.tcn(x)
        ht = tcn_out[:, -1, :]
        hw = tcn_out[:, :-1, :]
        hw = torch.transpose(hw, 1, 2)
        Hc = self.fc1(hw)
        Hn = self.fc2(Hc)
        ht = torch.unsqueeze(ht, dim=2)
        a = torch.bmm(Hn, ht)
        a = torch.sigmoid(a)
        a = torch.transpose(a, 1, 2)
        vt = torch.matmul(a, Hc)
        ht = torch.transpose(ht, 1, 2)
        hx = torch.cat((vt, ht), dim=2)
        y = self.fc3(hx)
        y = y.view(b, n)
        return y, (attn_weights_01, attn_weights_11, attn_weights_02, attn_weights_12)