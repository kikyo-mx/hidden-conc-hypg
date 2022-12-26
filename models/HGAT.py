from torch_geometric import nn
import torch
import torch.nn.functional as F
import torch.nn
import numpy as np

from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.attn import FullAttention, ProbAttention, AttentionLayer

seed = 123456789
np.random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = 'cuda:0'


class HGAT(torch.nn.Module):
    def __init__(self, d_model, ecod_in, seq_len, rr_num=None, use_attn=False, distil=True, el=3, c_in=32, c_out=32):
        super(HGAT, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.rr_num = rr_num
        self.attn = use_attn
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
        if self.attn:
            self.hatt1 = nn.HypergraphConv(self.c_in, self.c_out, use_attention=True, heads=1, concat=False,
                                           negative_slope=0.2, dropout=0.5, bias=True)
            self.liear2 = torch.nn.Linear(seq_len * c_in, c_in)
        else:
            self.hatt1 = nn.HypergraphConv(self.c_in, self.c_out, use_attention=False, heads=4, concat=False,
                                           negative_slope=0.2, dropout=0.5, bias=True)
            self.liear2 = torch.nn.Linear(int(d_model * seq_len / (2 ** (el - 1))), c_in)
        self.hatt2 = nn.HypergraphConv(self.c_out, self.c_in, use_attention=False, heads=1, concat=False,
                                       negative_slope=0.2, dropout=0.5, bias=True)
        #         self.ones = torch.ones((1, self.c_in))
        self.ones = torch.ones((1, 1))
        self.liear1 = torch.nn.Linear(self.c_in, len(self.rr_num))
        self.liear3 = torch.nn.Linear(d_model * seq_len, self.c_out)

    def forward(self, price_input, e, concept):
        # price_input = self.tokenConv(price_input.permute(0, 2, 1)).transpose(1,2)
        # context, query = self.informer_encoder_full(price_input, attn_mask=None)
        context, query = self.gru(price_input)
        # context, query = self.informer_encoder(price_input, attn_mask=None)
        output, weights = context.reshape(context.shape[0], -1), query[0].reshape(context.shape[0], -1)

        output = F.leaky_relu(self.liear2(output))
        # output = self.drop(output)

        # PreC_HG_Attention
        if self.attn:
            Hedge = torch.zeros((concept.shape[0], weights.shape[1])).to(device)
            Hnode = torch.zeros((concept.shape[0], weights.shape[1])).to(device)
            for edge in range(e[1][-1] + 1):
                node_index = torch.where(e[1] == edge)[0]
                Hnode[node_index] = weights[e[0][node_index]]
                Hedge[node_index] = torch.sum((concept[node_index].unsqueeze(1) * Hnode[node_index]), dim=0).float()
            HG_Attn = F.cosine_similarity(Hnode, Hedge) + 1 / 2
            HG_Attn = self.ones.to(device) * HG_Attn.unsqueeze(1)
            x = F.leaky_relu(self.hatt1(output, e, hyperedge_attr=HG_Attn), 0.2)
        else:
            x = F.leaky_relu(self.hatt1(output, e), 0.2)

        # PreC_HG_Attention2
        # Hedge = torch.zeros((e[1].max().item() + 1, weights.shape[1])).to(device)
        # for edge in range(e[1][-1] + 1):
        #     node_index = torch.where(e[1] == edge)
        #     node_set = weights[e[0][node_index]]
        #     Hedge[edge] = torch.sum((concept[node_index].unsqueeze(1) * node_set), dim=0).float()
        # print(Hedge.shape)

        # hidden concept




        # output = F.leaky_relu(self.liear3(output))
        # print(output.shape, 'out')
        x = F.leaky_relu(self.hatt2(x, e), 0.2)
        # x = self.drop(x)
        return F.leaky_relu(self.liear1(x))
