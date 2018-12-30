import torch as t
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from visdom import Visdom
from torchnet import meter
from tqdm import tqdm
import numpy as np
import ipdb
import random

import sys,os
sys.path.append('..')
from data.data_op import get_data
from models.Seq2Seq_Model import Encoder, Decoder, AttentionDecoder
from gensim.models.word2vec import Word2Vec

#训练词向量
def word2vec_train(opt, data,ix2word):
    params={
        'sg':1,#0:CBOW, 1:Skip-Gram。 小样本适合用Skip-Gram
        'size':opt.embed_size,
        'alpha':0.01,#在随机梯度下降法中迭代的初始步长
        'min_alpha':0.0005,#最小的迭代步长值
        'window': 10,#词向量上下文最大距离，推荐[5,10]
        'min_count': 1,#词向量的最小词频
        'seed': 1,#Seed for the random number generator
        "workers": 4,# Use these many worker threads to train the model
        "negative": 0,#使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]
        "hs": 1,  # 0: negative sampling, 1:Hierarchical  softmax, 这里选择1
        'compute_loss': True,
        'iter': 50,#随机梯度下降法中迭代的最大次数
        'cbow_mean': 0#仅用于CBOW在做投影的时候，为0，则算法中的xw为上下文的词向量之和，为1则为上下文的词向量的平均值,默认值是1
    }
    data_list = data.tolist()
    data_list=[[ix2word[i] for i in d] for d in data_list]
    model=Word2Vec(**params)
    model.build_vocab(data)
    train_word_count, raw_word_count=model.train(data, compute_loss=True, total_examples=model.corpus_count, epochs=model.epochs)

#给定几个词，根据这几个词生成接下来的词，生成一首完整的诗歌
def generate(opt,encoder, decoder,start_words,ix2word,word2ix,prefix_word=None):
    with t.no_grad():
        results=list(start_words)
        #手动设置第一个字为<START>
        start_words=[word2ix[w] for w in start_words[:opt.input_len]]
        tmp=start_words+[word2ix['</s>'] for i in range(opt.input_len-len(start_words))]
        input=np.array(tmp)
        input=t.from_numpy(input).long().view(-1,1)
        if opt.use_gpu:
            input = input.cuda()
        input_len=input.size(0)
        output_len=opt.maxlen
        encoder_hidden = encoder.initHidden(opt.use_gpu)
        encoder_outputs = t.zeros(opt.input_len, encoder.hidden_size, device='cuda' if opt.use_gpu else 'cpu')
        # encoder:
        for ei in range(input_len):
            encoder_output, encoder_hidden = encoder(input[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]
        # decoder:
        decoder_input = t.tensor([[word2ix['<START>']]], device='cuda' if opt.use_gpu else 'cpu')
        decoder_hidden = encoder_hidden
        for di in range(output_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,encoder_outputs)
            topv, topi = decoder_output.topk(1)  # 预测出来的下一个字
            decoder_input = topi.squeeze().detach()  # detach阻断这个节点上的反向传播
            if decoder_input.item() == word2ix['<END>']:
                break
            else:
                results.append(ix2word[topi.item()])
        return results

def train(opt):
    #创建可视化对象
    if opt.use_env:
        #需要在pycharm里的Terminal先启动visdom服务器：python -m visdom.server
        vis=Visdom(env=opt.env)

    #获取数据
    data,ix2word,word2ix=get_data(opt)#
    #word_embeds=word2vec_train(opt, data, ix2word)#预先训练词向量，代替模型中的embed层，因为要做text rank提取关键字
    #print(word_embeds)
    data=t.from_numpy(data)#转成torch
    # print(data.shape, data)
    dataloader=DataLoader(data,batch_size=opt.batch_size,shuffle=True,num_workers=1)

    #模型定义
    encoder=Encoder(len(word2ix), opt.hidden_dim)
    decoder=AttentionDecoder(opt.hidden_dim, len(word2ix),opt.dropout_rate,opt.input_len)

    en_optimizer=t.optim.SGD(encoder.parameters(),lr=opt.lr)
    de_optimizer=t.optim.SGD(decoder.parameters(),lr=opt.lr)

    criterion=nn.CrossEntropyLoss()

    # if opt.model_path:
    #     model.load_state_dict(t.load(opt.model_path))

    if opt.use_gpu:
        # model=model.cuda()
        encoder=encoder.cuda()
        decoder=decoder.cuda()
        criterion=criterion.cuda()

    loss_meter=meter.AverageValueMeter()#原来用的是loss=0.0，现在用这个自动计算平均值
    # count = 0
    for epoch in range(opt.epoch):
        # print(epoch)
        loss_meter.reset()  # 重置为0
        count=0
        for i,data_ in enumerate(dataloader):#tqdm 在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
            #训练
            #data_: 一个batch,torch.Size([128, 80]), 每个batch是128首诗，每首诗最长是80个字。后续的hidden_size=embed_size=128
            # print(data_.shape)
            # print(data_)
            data_=data_.long().contiguous()#transpose交换第0维度和第1维度，所以data从batch_size*seq_len变成seq_len*batch_size; contoguouse()把tensor变成在内存中连续分布的形式。
            if opt.use_gpu:
                data_=data_.cuda()
            #loss_meter.reset()  # 重置为0
            for poetry in data_:#poetry是data_的一行，也就是一首诗
                loss = 0
                encoder_hidden = encoder.initHidden(opt.use_gpu)
                en_optimizer.zero_grad()  # 置零
                de_optimizer.zero_grad()
                count+=1
                # print(poetry)
                #print(epoch,':',count)
                input_, target_=poetry[:opt.input_len].view(-1,1),poetry[1:].view(-1,1)#输入为每首诗的前10个字，target为整首诗, 形状!!!!!,
                input_len=input_.size(0)
                output_len=target_.size(0)
                encoder_outputs=t.zeros(opt.input_len, encoder.hidden_size, device='cuda' if opt.use_gpu else 'cpu')

                #loss_meter.reset()  # 重置为0
                # encoder_hidden = encoder.initHidden(opt.use_gpu)
                # en_optimizer.zero_grad()  # 置零
                # de_optimizer.zero_grad()
                #encoder:
                for ei in range(input_len):
                    encoder_output, encoder_hidden=encoder(input_[ei], encoder_hidden)
                    encoder_outputs[ei]=encoder_output[0]
                #decoder:
                #use_teacher_forcing=True if random.random()<opt.teacher_forcing_ratio else False
                use_teacher_forcing = True
                decoder_input=t.tensor([[word2ix['<START>']]], device='cuda' if opt.use_gpu else 'cpu')
                decoder_hidden=encoder_hidden
                if use_teacher_forcing:
                    for di in range(output_len):
                        decoder_output, decoder_hidden, decoder_attention=decoder(decoder_input, decoder_hidden, encoder_outputs)
                        decoder_input = target_[di]
                        loss += criterion(decoder_output, target_[di])
                        #loss_meter.add(loss.item())
                else:
                    for di in range(output_len):
                        decoder_output, decoder_hidden, decoder_attention=decoder(decoder_input, decoder_hidden, encoder_outputs)
                        topv, topi=decoder_output.topk(1)#预测出来的下一个字
                        decoder_input=topi.squeeze().detach()#detach阻断这个节点上的反向传播
                        loss+=criterion(decoder_output, target_[di])
                        #loss_meter.add(loss.item())
                        if decoder_input.item()==word2ix['<END>']:
                            break

                loss.backward()
                en_optimizer.step()
                de_optimizer.step()
                #可视化
                if count%opt.plot_every==0:
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()#设置断点
                    #显示loss值
                    #print(type(loss_meter.value()[0]),loss_meter.value()[0])
                    vis.line(X=np.array([(count//opt.plot_every)/opt.plot_every]),Y=np.array([loss.item()/output_len]),win='loss',update='None'if count//opt.plot_every==0 else 'append')
                    #显示诗歌原文
                    #print(type(poetry),poetry.shape,poetry.tolist())
                    p=[ix2word[k] for k in poetry.tolist()]#输出原诗

                    vis.text(' '.join(p),win=u'origin_poem')

                    # #显示生成的诗
                    start_words='床前明月光，疑似地上霜'
                    gen_poetry = ''.join(generate(opt, encoder, decoder, start_words, ix2word, word2ix))
                    print(i,':',gen_poetry)
                    vis.text(''.join(gen_poetry), win=u'generate_poem')
                    # #保存模型
                    t.save(encoder.state_dict(),'%s/seq2seq/1_%s.pth'%(opt.model_prefix,epoch))
                    t.save(decoder.state_dict(),'%s/seq2seq/1_%s.pth'%(opt.model_prefix,epoch))
