from torch_geometric import nn
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.attn import FullAttention, ProbAttention, AttentionLayer

seed = 123456789
np.random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class HGAT(torch.nn.Module):
    def __init__(self, d_model, ecod_in, seq_len, rr_num=None, use_PreAttn=False, use_HidAttn=False, distil=True, el=3,
                 c_in=32, c_out=32, device='cuda:1', Hedge_index=False, encoder='GRU', col_train=None):
        super(HGAT, self).__init__()
        if col_train is None:
            col_train = ['rr1', 'rr5', 'rr10', 'rr20', 'rr30', 'volumn_ratio']
        self.c_in = c_in
        self.c_out = c_out
        self.encoder = encoder
        self.Hedge_index = Counter(Hedge_index)
        self.device = device
        self.rr_num = rr_num
        self.PreAttn = use_PreAttn
        self.HidAttn = use_HidAttn
        self.gru = torch.nn.GRU(input_size=len(col_train), hidden_size=c_in, batch_first=True)
        self.lstm = torch.nn.LSTM(input_size=len(col_train), hidden_size=c_in, batch_first=True)
        self.Attn = ProbAttention
        self.drop = torch.nn.Dropout(p=0.2)
        self.tokenConv = torch.nn.Conv1d(ecod_in, out_channels=d_model,
                                         kernel_size=3, padding=1, padding_mode='circular')
        self.informer_encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    self.Attn(False, factor=6, attention_dropout=0.2,
                              output_attention=True), d_model=d_model, n_heads=8, mix=False),
                d_model=d_model,
                d_ff=d_model,
                dropout=0.2,
                activation='gelu') for l in range(el)],
            [ConvLayer(
                d_model
            ) for l in range(el - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        if self.PreAttn:
            self.hatt1 = nn.HypergraphConv(self.c_in, self.c_out, use_attention=True, heads=1, concat=False,
                                           negative_slope=0.2, dropout=0.2, bias=True)
            self.linear2 = torch.nn.Linear(seq_len * c_in, c_in)
        else:
            self.hatt1 = nn.HypergraphConv(self.c_in, self.c_out, use_attention=False, heads=1, concat=False,
                                           negative_slope=0.2, dropout=0.2, bias=True)
            self.linear2 = torch.nn.Linear(int(d_model * seq_len / (2 ** (el - 1))), c_in)
        if self.HidAttn:
            self.hatt2 = nn.HypergraphConv(self.c_out, self.c_in, use_attention=True, heads=1, concat=False,
                                           negative_slope=0.2, dropout=0.2, bias=True)
        else:
            self.hatt2 = nn.HypergraphConv(self.c_out, self.c_in, use_attention=False, heads=1, concat=False,
                                           negative_slope=0.2, dropout=0.2, bias=True)
        #         self.ones = torch.ones((1, self.c_in))
        self.ones = torch.ones((1, 1))
        self.linear1 = torch.nn.Linear(self.c_in, len(self.rr_num))
        self.linear4 = torch.nn.Linear(c_in, c_in)
        self.linear5 = torch.nn.Linear(c_in, c_in)
        self.linear3 = torch.nn.Linear(d_model * seq_len, self.c_out)
        self.softmax = torch.nn.Softmax(dim=0)
        self.bn = torch.nn.BatchNorm1d(len(rr_num))

    def Hattn(self, weights, concept, e):
        HG_Attn = torch.zeros((torch.max(e[1]) + 1, weights.shape[1])).to(self.device)
        start = 0
        for edge in self.Hedge_index:
            node_index = np.arange(start, start + self.Hedge_index[edge])
            Pnode = weights[e[0][node_index]]
            HG_Attn[edge] = torch.sum((concept[node_index].unsqueeze(1) * Pnode), dim=0).float()
            start += self.Hedge_index[edge]
        return HG_Attn

    def Hidden_edge(self, x_hid, volumn):
        volumn = volumn.squeeze(0).numpy()
        stock_cosine = torch.zeros((x_hid.shape[0]), dtype=torch.int8)
        cosine = torch.zeros((x_hid.shape[0]))
        for i in range(x_hid.shape[0]):
            col_x = x_hid.clone()
            col_x[i] = 0
            cos_sim = F.cosine_similarity(x_hid[i], col_x, dim=1)
            stock_cosine[i] = torch.argmax(cos_sim).int()
            cosine[i] = cos_sim[stock_cosine[i]]
        HidEdge = []
        HidNode = []
        # cp_x = x_hid.cpu().detach().numpy()
        Edge_index = 0
        for i in range(x_hid.shape[0]):
            if cosine[i] < 0.5:
                continue
            rel_index = torch.where(stock_cosine == i)[0]
            if len(rel_index) == 0:
                continue
            else:
                Edge_index += 1
            HidNode = np.append(HidNode, rel_index).astype(int)
            HidNode = np.append(HidNode, i)
            # if np.sum(volumn[HidNode]) < 1e-8:
            e_hid = torch.mean(x_hid[HidNode], dim=0).reshape(1, -1)
            # else:
            #     e_hid = np.sum(cp_x[HidNode] * (volumn[HidNode] / np.sum(volumn[HidNode])).reshape(-1, 1), axis=0)
            if Edge_index == 1:
                HidAttn = e_hid
            else:
                HidAttn = torch.cat((HidAttn, e_hid))
            HidEdge = np.append(HidEdge, np.full(rel_index.shape[0] + 1, Edge_index - 1)).astype(int)
        Hid_index = torch.stack((torch.from_numpy(HidNode), torch.from_numpy(HidEdge))).to(self.device)
        # HidAttn = torch.from_numpy(HidAttn).to(self.device)
        return Hid_index, HidAttn

    def forward(self, price_input, e, concept, volumn, compare=0):
        if self.encoder == 'Informer':
            price_input = self.tokenConv(price_input.permute(0, 2, 1)).transpose(1, 2)
            context, query = self.informer_encoder(price_input, attn_mask=None)
            output = weights = self.linear2(context.reshape(context.shape[0], -1))
        elif self.encoder == 'LSTM':
            context, query = self.lstm(price_input)
            weights, output = context.reshape(context.shape[0], -1), query[0].reshape(context.shape[0], -1)
        else:
            context, query = self.gru(price_input)
            weights, output = context.reshape(context.shape[0], -1), query[0].reshape(context.shape[0], -1)
        # output = F.leaky_relu(self.linear2(output))
        # output = self.linear4(output)
        if compare:
            output = self.linear4(output)
            return F.leaky_leaky_relu(self.linear1(output))

        # PreC_HG_PreAttention
        if self.PreAttn:
            HG_AttnP = self.Hattn(output, concept, e)
            # HG_AttnP = self.ones.to(self.device) * HG_AttnP.unsqueeze(1)
            x = F.leaky_relu(self.hatt1(output, e, hyperedge_attr=HG_AttnP))
        else:
            x = F.leaky_relu(self.hatt1(output, e))

        # PreC_HG_Hidden_Attention
        if self.HidAttn:
            x = x - self.linear5(self.linear4(output))
            e_hid, HG_AttnH = self.Hidden_edge(x, volumn)
            # HG_AttnH = torch.from_numpy(HG_AttnH).to(self.device)
            # HG_AttnH = self.ones.to(self.device) * HG_AttnH.unsqueeze(1)
            x = F.leaky_relu(self.hatt2(x, e_hid, hyperedge_attr=HG_AttnH))

        else:
            x = F.leaky_relu(self.hatt2(x, e))

        x = F.leaky_relu(self.linear1(x))
        # return (x - torch.min(x, dim=0)[0]) / (torch.max(x, dim=0)[0] - torch.min(x, dim=0)[0])
        return x
