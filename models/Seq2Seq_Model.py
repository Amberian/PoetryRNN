import torch
import torch.nn as nn
from torch.nn import  functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size=hidden_size
        self.embeddding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded=self.embeddding(input).view(1,1,-1)
        output=embedded
        output, hidden=self.gru(output, hidden)
        return output, hidden

    def initHidden(self,use_gpu):
        return torch.zeros(1,1,self.hidden_size, device='cuda' if use_gpu else 'cpu')

class Decoder(nn.Module):
    def __init__(self,hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(output_size, hidden_size)
        self.gru=nn.GRU(hidden_size, hidden_size)
        self.out=nn.Linear(hidden_size, output_size)
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embed=self.embedding(input).view(1,1,-1)
        output=F.relu(embed)
        output,hidden=self.gru(output, hidden)
        output=self.out(output[0])#output[0]应该是shape为(*,*)的矩阵
        output=self.softmax(output)
        return output, hidden
    def initHidden(self, device):
        return torch.zeros(1,1,self.hidden_size, device=device)


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_rate=0.1, input_len=80):
        super(AttentionDecoder, self).__init__()
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.dropout_rate=dropout_rate
        self.input_len=input_len

        self.embedding=nn.Embedding(output_size, hidden_size)
        self.attn=nn.Linear(hidden_size*2, input_len)#把embed和pre_hidden拼接在一起，所以是*2
        self.attn_combine=nn.Linear(hidden_size*2, hidden_size)#把attn_applied和embedded拼接在一起，所以是*2
        self.dropout=nn.Dropout(dropout_rate)
        self.gru=nn.GRU(hidden_size, hidden_size)
        self.out=nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output):
        embedded=self.dropout(self.embedding(input).view(1,1,-1))
        # print(embedded.shape, hidden.shape)
        attn_weights=F.softmax(self.attn(torch.cat((embedded[0], hidden[0]),1)), dim=1)#通过torch.cat将两个向量拼接在一起
        # print(attn_weights.shape,attn_weights.unsqueeze(0).shape,encoder_output.unsqueeze(0).shape)
        attn_applied=torch.bmm(attn_weights.unsqueeze(0), encoder_output.unsqueeze(0))#两个batch的矩阵相乘，矩阵维度(b,m,p)， b为batch size, 矩阵的形状为m*p, 两个batch(10, 3, 4)和(10, 4, 5)相乘之后的维度为(10, 3, 5)
        attn_combine=self.attn_combine(torch.cat((embedded[0], attn_applied[0]),1)).unsqueeze(0)
        out=F.relu(attn_combine)
        out,hidden=self.gru(out, hidden)
        out=F.log_softmax(self.out(out[0]), dim=1)
        return out, hidden, attn_weights

    def initHidden(self,device):
        return torch.zeros(1,1,self.hidden_size, device=device)






