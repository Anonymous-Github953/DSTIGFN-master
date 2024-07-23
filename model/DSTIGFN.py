import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.linear1 = nn.Linear(features, features)
        self.linear2 = nn.Linear(features, features)
        self.linear3 = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.linear3(out)
        return out.permute(0, 3, 2, 1)


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        self.time = time
        self.time_day = nn.Embedding(time, features // 2)
        self.time_week = nn.Embedding(7, features // 2)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day((day_emb * self.time).long()).transpose(1, 2).contiguous()

        week_emb = x[..., 2]
        time_week = self.time_week(week_emb.long()).transpose(1, 2).contiguous()

        tem_emb = torch.cat([time_day, time_week], dim=-1)
        tem_emb = tem_emb.permute(0, 3, 1, 2)
        return tem_emb


class AGSG(nn.Module):
    def __init__(self, device, num_nodes, channels, alph):
        super(AGSG, self).__init__()
        self.device = device
        self.alph = alph
        self.num_nodes = num_nodes
        self.channels = channels
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.mhsg = MHSG(device, num_nodes, alph)

    def forward(self, x):
        initial_S = F.relu(torch.mm(self.memory.transpose(0, 1), self.memory)).to(self.device)
        initial_S = torch.where(torch.eye(self.num_nodes, device=self.device) == 1, torch.full_like(initial_S, 0.1), initial_S)

        S_w = F.softmax(initial_S, dim=1).to(self.device)
        support_set = [torch.eye(self.num_nodes).to(self.device), S_w]

        for k in range(2, self.num_nodes + 1):
            support_set.append(torch.mm(S_w, support_set[k - 1]))

        supports = torch.stack(support_set, dim=0).to(self.device)
        A_p = torch.softmax(F.relu(torch.einsum("bcnt, knm->bnm", x, supports).contiguous() / math.sqrt(x.shape[1])), -1)
        A_l = self.mhsg(x, S_w)
        return A_p, A_l


class MHSG(nn.Module):
    def __init__(self, device, num_nodes, alph):
        super(MHSG, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.alph = alph

    def forward(self, x, s):
        T = x.size(3)
        Bootstrap_num = np.random.choice(range(T), size=(T,))
        Bootstrap_num.sort()

        supra_laplacian = torch.zeros(size=(self.num_nodes * Bootstrap_num.shape[0], self.num_nodes * Bootstrap_num.shape[0])).to(self.device)
        inter_diagonal_matrix = np.eye(self.num_nodes, dtype=np.float32) * self.alph
        inter_diagonal_matrix = torch.FloatTensor(inter_diagonal_matrix).to(self.device)

        for i in range(Bootstrap_num.shape[0]):
            for j in range(Bootstrap_num.shape[0]):
                if i == j:
                    supra_laplacian[self.num_nodes * i: self.num_nodes * (i + 1), self.num_nodes * i: self.num_nodes * (i + 1)] = s
                elif j > i:
                    supra_laplacian[self.num_nodes * i: self.num_nodes * (i + 1), self.num_nodes * j: self.num_nodes * (j + 1)] = inter_diagonal_matrix

        x_window = x.view(x.size(0), x.size(1), -1)
        x_window = F.relu(torch.einsum("bcn, nm->bcn", x_window, supra_laplacian) / math.sqrt(x_window.shape[1]))
        adj_dyn = torch.softmax(x_window, -1)
        x_w_s = adj_dyn.view(adj_dyn.size(0), -1, self.num_nodes, Bootstrap_num.shape[0])

        A_l = torch.softmax(F.relu(torch.einsum("bcn, bcm->bnm", x_w_s.sum(-1), x_w_s.sum(-1)).contiguous() / math.sqrt(x_w_s.shape[1])), -1)
        return A_l


class DGGC(nn.Module):
    def __init__(self, device, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, alph=0.0, gama=0.8, emb=None):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.alph = alph
        self.gama = gama
        self.conv = nn.Conv2d(channels, channels, (1, 1))
        self.agsc = AGSG(device, num_nodes, channels, alph)
        self.fc = nn.Conv2d(2, 1, (1, 1))
        self.emb = emb
        self.conv_gcn = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        skip = x
        x = self.conv(x)
        A_p, A_l = self.agsc(x)

        A_f = torch.cat([A_p.unsqueeze(-1), A_l.unsqueeze(-1)], dim=-1)
        A_f = torch.softmax(self.fc(A_f.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).squeeze(), -1)

        topk_values, topk_indices = torch.topk(A_f, k=int(A_f.shape[1] * self.gama), dim=-1)
        mask = torch.zeros_like(A_f)
        mask.scatter_(-1, topk_indices, 1)
        A_f = A_f * mask

        out = []
        for i in range(self.diffusion_step):
            x = torch.einsum("bcnt,bnm->bcmt", x, A_f).contiguous()
            out.append(x)

        x = torch.cat(out, dim=1)
        x = self.conv_gcn(x)
        x = self.dropout(x)
        x = x * self.emb + skip
        return x


class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return self.even(x), self.odd(x)


class STIF(nn.Module):
    def __init__(self, device, alph, gama, channels=64, diffusion_step=1, splitting=True, num_nodes=170, dropout=0.2, emb=None):
        super(STIF, self).__init__()
        self.device = device
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.split = Splitting()

        pad_l, pad_r = 3, 3
        k1, k2 = 5, 3

        def make_tconv():
            return nn.Sequential(
                nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
                nn.Conv2d(channels, channels, kernel_size=(1, k1)),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(self.dropout),
                nn.Conv2d(channels, channels, kernel_size=(1, k2)),
                nn.Tanh(),
            )

        self.tconv1 = make_tconv()
        self.tconv2 = make_tconv()
        self.tconv3 = make_tconv()
        self.tconv4 = make_tconv()

        self.dggc = DGGC(device, channels, num_nodes, diffusion_step, dropout, alph, gama, emb)

    def forward(self, x):
        if self.splitting:
            x_even, x_odd = self.split(x)
        else:
            x_even, x_odd = x, x

        def process(x_even, x_odd, tconv1, tconv2):
            x1 = tconv1(x_even)
            x1 = self.dggc(x1)
            d = x_odd * torch.tanh(x1)

            x2 = tconv2(x_odd)
            x2 = self.dggc(x2)
            c = x_even * torch.tanh(x2)

            return c, d

        c, d = process(x_even, x_odd, self.tconv1, self.tconv2)
        x_odd_update = d + self.dggc(self.tconv3(c))
        x_even_update = c + self.dggc(self.tconv4(d))

        return x_even_update, x_odd_update

class STGIF(nn.Module):
    def __init__(self, device, channels=64, diffusion_step=1, num_nodes=170, dropout=0.1, alph=0.0, gama=0.8):
        super(STGIF, self).__init__()

        self.library1 = nn.Parameter(torch.randn(channels, num_nodes, 6))
        self.library2 = nn.Parameter(torch.randn(channels, num_nodes, 3))
        self.library3 = nn.Parameter(torch.randn(channels, num_nodes, 3))

        self.STIF1 = STIF(device, alph, gama, splitting=True, channels=channels, diffusion_step=diffusion_step, num_nodes=num_nodes, dropout=dropout, emb=self.library1)
        self.STIF2 = STIF(device, alph, gama, splitting=True, channels=channels, diffusion_step=diffusion_step, num_nodes=num_nodes, dropout=dropout, emb=self.library2)
        self.STIF3 = STIF(device, alph, gama, splitting=True, channels=channels, diffusion_step=diffusion_step, num_nodes=num_nodes, dropout=dropout, emb=self.library3)

    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        length = even.shape[0]
        _ = []
        for i in range(length):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x):
        x_even_update1, x_odd_update1 = self.STIF1(x)
        x_even_update2, x_odd_update2 = self.STIF2(x_even_update1)
        x_even_update3, x_odd_update3 = self.STIF3(x_odd_update1)

        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        output = concat0 + x
        return output

class DSTIGFN(nn.Module):
    def __init__(self, device, input_dim, num_nodes, channels, granularity, dropout=0.1, alph=0.0, gama=0.8):
        super(DSTIGFN, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.output_len = 12

        self.Temb = TemporalEmbedding(granularity, channels)
        self.input_proj = nn.Linear(input_dim, channels)

        self.tree = STGIF(device, channels=channels * 2, diffusion_step=2, num_nodes=self.num_nodes, dropout=dropout, alph=alph, gama=gama)
        self.glu = GLU(channels * 2, dropout)
        self.regression_layer = nn.Conv2d(channels * 2, self.output_len, kernel_size=(1, self.output_len))

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, input):
        x = input
        time_emb = self.Temb(input.permute(0, 3, 2, 1))
        x = torch.cat([self.input_proj(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1), time_emb], dim=1)
        x = self.tree(x)
        glu = self.glu(x) + x
        prediction = self.regression_layer(F.relu(glu))
        return prediction
