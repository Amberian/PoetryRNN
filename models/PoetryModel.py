import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class PoetryModel(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim):
        super(PoetryModel,self).__init__()
        self.hidden_dim=hidden_dim#隐藏层维度
        self.embed=nn.Embedding(vocab_size,embed_dim)#其实是为每个词建立了一个初始索引，这个索引是一个embed_dim维度的向量，后期通过学习的方式改进
        self.lstm=nn.LSTM(embed_dim, self.hidden_dim,num_layers=2)#input的特征维度，隐藏层的特征数，rnn层数=2
        self.linear=nn.Linear(self.hidden_dim,vocab_size)
        self.dropout=nn.Dropout(0.2)#

    def forward(self, input, hidden=None):#hidden表示增加的隐藏层层数
        seq_len,batch_size,=input.shape
        if hidden is None:#如果隐藏层的权重和单元的初值没有给出，就使用以下默认值
            h_0=input.data.new(2,batch_size,self.hidden_dim).fill_(0).float()#shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch
            c_0=input.data.new(2,batch_size,self.hidden_dim).fill_(0).float()#shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch
        else:#否则使用给定的初始值
            h_0,c_0=hidden#LSTM的输入为LSTM(input, (h0, c0)),输出为output, (hn, cn)
        embeds=self.embed(input)#输出：一首诗的长度，batch_size，每个词的词向量维度
        output,hidden=self.lstm(embeds,(h_0,c_0))#输出：一首诗的长度，batch_size，隐藏层输出的特征维度
        output=self.linear(output.view(seq_len*batch_size,-1))#输出：一首诗的长度*batch_size，vocab_siize
        return output,hidden
