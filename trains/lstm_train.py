import torch as t
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from visdom import Visdom
from torchnet import meter
from tqdm import tqdm
import numpy as np
import ipdb

import sys,os
sys.path.append('..')
from data.data_op import get_data
from models.LSTM_Model import LSTM_Model


#给定几个词，根据这几个词生成接下来的词，生成一首完整的诗歌
def generate(opt,model,start_words,ix2word,word2ix,prefix_word=None):
    results=list(start_words)
    #手动设置第一个字为<START>
    input=t.Tensor([word2ix['<START>']]).view(1,1).long()
    if opt.use_gpu:
        input=input.cuda()
    #用prefix_word控制意境
    #因为是在模型训练过程中调用generate，所以这里传进去的意境诗歌会影响model的参数，用这几个词训练完之后，记下LSTM中隐藏层的参数hidden，传到下次正式开始产生的模型里
    hidden=None
    if opt.prefix_words:
        for p in opt.prefix_words:
            output,hidden=model(input,hidden)
            input=input.data.new([word2ix[p]]).view(1,1)#新建tensor，用以表明是cuda
    #开始根据start_words产生诗歌
    start_words_len=len(start_words)
    #print('len_start:',start_words_len)
    for i in range(opt.max_gen_len):
        output,hidden=model(input,hidden)
        if i<start_words_len:
            #这里的start_words只有一个字，
            #print('<=len_start:',i)
            w=results[i]
            input=input.data.new([word2ix[w]]).view(1,1)#传入start_words的一个字
        else:
            #用预测的词来作为下一次的输入
            top1=output.data[0].topk(1)[1][0].item()#该词在ix2word中的ix
            w=ix2word[top1]
            results.append(w)
            input = Variable(input.data.new([top1])).view(1, 1)
        if w=='<END>':
            #print(i,':',w)
            del results[-1]
            break

    return results





def train(opt):
    #创建可视化对象
    if opt.use_env:
        #需要在pycharm里的Terminal先启动visdom服务器：python -m visdom.server
        vis=Visdom(env=opt.env)

    #获取数据
    data,ix2word,word2ix=get_data(opt)#
    data=t.from_numpy(data)#转成torch
    print('!!!!',data.shape)
    dataloader=DataLoader(data,batch_size=opt.batch_size,shuffle=True,num_workers=1)

    #模型定义
    model=LSTM_Model(len(word2ix),opt.embed_size,opt.hidden_dim)
    #print(model)

    optimizer=t.optim.Adam(model.parameters(),lr=opt.lr)
    criterion=nn.CrossEntropyLoss()

    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))

    if opt.use_gpu:
        model=model.cuda()
        criterion=criterion.cuda()
    loss_meter=meter.AverageValueMeter()#原来用的是loss=0.0，现在用这个自动计算平均值

    count=0
    for epoch in range(opt.epoch):
        loss_meter.reset()#重置为0
        print(epoch)
        for i,data_ in enumerate(dataloader):#tqdm 在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
            #训练
            print('@@@@',data_.shape)
            data_=data_.long().transpose(0,1).contiguous()#transpose交换第0维度和第1维度，所以data从batch_size*seq_len变成seq_len*batch_size; contoguouse()把tensor变成在内存中连续分布的形式。
            if opt.use_gpu:
                data_=data_.cuda()
            optimizer.zero_grad()#置零

            #input和target错开
            input_,target_=data_[:-1,:],data_[1:,:]
            #print(input_.shape)
            output_,_=model(input_,)
            print('&*&*&*&*&*',output_.shape, target_.shape, target_.view(-1).shape)
            loss=criterion(output_,target_.view(-1))
            loss.backward()
            optimizer.step()#更新参数
            #print(loss.data.shape)
            loss_meter.add(loss.item())

            #可视化
            if (i+1)%opt.plot_every==0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()#设置断点

                #显示loss值
                #print(type(loss_meter.value()[0]),loss_meter.value()[0])
                vis.line(X=np.array([count]),Y=np.array([loss_meter.value()[0]]),win='loss',update='None'if count==0 else 'append')
                count+=1
                #显示每批次前5首诗歌原文
                poetrys=[[ix2word[_w] for _w in data_[:,_i].tolist()] for _i in range(5)]#将一个128批次的前16首诗转为二维list

                vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]),win=u'origin_poem')

                #显示生成的诗
                gen_poetries=[]
                # out=generate(model, '春', ix2word, word2ix)
                # print('generate:',out)
                # for word in list(u'春江花月夜凉如水'):
                #     gen_poetry=''.join(generate(model,word,ix2word,word2ix))#根据一个字生成一首诗
                #     gen_poetries.append(gen_poetry)#一共得到8首诗
                # vis.text('</br>'.join([''.join(gen) for gen in gen_poetries]),win=u'generate_poem')
                # start_words='月夜凉如水'
                # # gen_poetry = '</br>'.join(''.join(generate(model, word, ix2word, word2ix) )for word in start_words)
                # # vis.text(''.join(gen_poetry), win=u'generate_poem')
                # gen_poetry = ''.join(generate(model, start_words, ix2word, word2ix))
                # vis.text(''.join(gen_poetry), win=u'generate_poem')
        start_words='月夜凉如水'
        gen_poetry = ''.join(generate(opt, model, start_words, ix2word, word2ix))
        print(epoch,':',gen_poetry)
        #保存模型
        t.save(model.state_dict(),'%s_%s.pth'%(opt.model_prefix,epoch))

def test(opt,start_words=None):
    data, ix2word, word2ix = get_data(opt)
    model=LSTM_Model(len(word2ix),opt.embed_size,opt.hidden_dim)
    model.load_state_dict(t.load('checkpoints/poetry4_49.pth'))#可替换
    if opt.use_gpu:
        model=model.cuda()
    if start_words==None:
        start_words=opt.start_words
    gen_poetry = ''.join(generate(model, start_words, ix2word, word2ix))
    print(gen_poetry)






