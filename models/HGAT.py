from torch_geometric import nn
import torch
import torch.nn.functional as F
import numpy as np

from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.attn import FullAttention, ProbAttention, AttentionLayer

seed = 123456789
np.random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class HGAT(torch.nn.Module):
    def __init__(self, d_model, ecod_in, seq_len, rr_num=None, use_PreAttn=False, use_HidAttn=False, distil=True, el=3, c_in=32, c_out=32, device='cuda:1'):
        super(HGAT, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.device = device
        self.rr_num = rr_num
        self.PreAttn = use_PreAttn
        self.HidAttn = use_HidAttn
        self.gru = torch.nn.GRU(input_size=6, hidden_size=c_in, batch_first=True)
        self.Attn = ProbAttention
        self.drop = torch.nn.Dropout(p=0.2)
        self.tokenConv = torch.nn.Conv1d(ecod_in, out_channels=d_model,
                                         kernel_size=3, padding=1, padding_mode='circular')
        self.informer_encoder_full = Encoder([
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
                                           negative_slope=0.2, dropout=0.5, bias=True)
            self.linear2 = torch.nn.Linear(seq_len * c_in, c_in)
        else:
            self.hatt1 = nn.HypergraphConv(self.c_in, self.c_out, use_attention=False, heads=4, concat=False,
                                           negative_slope=0.2, dropout=0.5, bias=True)
            self.linear2 = torch.nn.Linear(int(d_model * seq_len / (2 ** (el - 1))), c_in)
        if self.HidAttn:
            self.hatt2 = nn.HypergraphConv(self.c_out, self.c_in, use_attention=True, heads=1, concat=False,
                                           negative_slope=0.2, dropout=0.5, bias=True)
        else:
            self.hatt2 = nn.HypergraphConv(self.c_out, self.c_in, use_attention=False, heads=1, concat=False,
                                       negative_slope=0.2, dropout=0.5, bias=True)
        #         self.ones = torch.ones((1, self.c_in))
        self.ones = torch.ones((1, 1))
        self.linear1 = torch.nn.Linear(self.c_in, len(self.rr_num))
        self.linear4 = torch.nn.Linear(c_in, c_in)
        self.linear3 = torch.nn.Linear(d_model * seq_len, self.c_out)
        self.softmax = torch.nn.Softmax(dim=0)


    def Hattn(self, weights, concept, e):
        Pedge = torch.zeros((concept.shape[0], weights.shape[1])).to(self.device)
        Pnode = torch.zeros((concept.shape[0], weights.shape[1])).to(self.device)
        for edge in range(e[1][-1] + 1):
            node_index = torch.where(e[1] == edge)[0]
            Pnode[node_index] = weights[e[0][node_index]]
            Pedge[node_index] = torch.sum((concept[node_index].unsqueeze(1) * Pnode[node_index]), dim=0).float()
        HG_Attn = (F.cosine_similarity(Pnode, Pedge) + 1) / 2
        return HG_Attn


    def Hidden_edge(self, x_hid):
        stock_cosine = torch.zeros((x_hid.shape[0]), dtype=torch.int8)
        stock_sim = np.zeros((x_hid.shape[0]))
        for i in range(x_hid.shape[0]):
            cp_x = x_hid.clone()
            cp_x[i] = 0
            cos_sim = F.cosine_similarity(x_hid[i], cp_x, dim=1)
            stock_cosine[i] = torch.argmax(cos_sim).int()
            stock_sim[i] = cos_sim[stock_cosine[i]]
        HidEdge = []
        HidNode = []
        HidAttn = []
        for i in range(x_hid.shape[0]):
            rel_index = torch.where(stock_cosine == i)[0]
            HidNode = np.append(HidNode, rel_index)
            HidEdge = np.append(HidEdge, np.full(rel_index.shape[0], i))
            HidAttn = np.append(HidAttn, stock_sim[rel_index])
        e_hid = torch.stack((torch.from_numpy(HidNode.astype(np.int64)), torch.from_numpy(HidEdge.astype(np.int64)))).to(self.device)
        return e_hid, HidAttn

    def forward(self, price_input, e, concept):
        # price_input = self.tokenConv(price_input.permute(0, 2, 1)).transpose(1,2)
        # context, query = self.informer_encoder_full(price_input, attn_mask=None)
        context, query = self.gru(price_input)
        # context, query = self.informer_encoder(price_input, attn_mask=None)
        weights, output = context.reshape(context.shape[0], -1), query[0].reshape(context.shape[0], -1)

        # output = F.leaky_relu(self.linear2(output))
        output = F.leaky_relu(output)
        # output = self.drop(output)

        # PreC_HG_PreAttention
        if self.PreAttn:
            HG_AttnP = self.Hattn(weights, concept, e)
            HG_AttnP = self.ones.to(self.device) * HG_AttnP.unsqueeze(1)
            x = F.leaky_relu(self.hatt1(output, e, hyperedge_attr=HG_AttnP), 0.2)
        else:
            x = F.leaky_relu(self.hatt1(output, e), 0.2)
        # PreC_HG_Hidden_Attention
        if self.HidAttn:
            x = F.leaky_relu(x - self.linear4(output))
            e_hid, HG_AttnH = self.Hidden_edge(x)
            HG_AttnH = torch.from_numpy(HG_AttnH).to(self.device)
            HG_AttnH = self.ones.to(self.device) * HG_AttnH.unsqueeze(1)
            x = F.leaky_relu(self.hatt2(x, e_hid, hyperedge_attr=HG_AttnH), 0.2).to(torch.float32)
        else:
            x = F.leaky_relu(self.hatt2(x, e), 0.2)


        # x = self.drop(x)
        return F.leaky_relu(self.linear1(x))
