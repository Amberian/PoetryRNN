import numpy as np
import pandas as pd
import nltk
import re
import json
from collections import Counter
import os
def json2numpy(opt):
    #提取json文件中的诗词，存成二维list
    def ajson2numpy(filename):
        with open(filename,'rb') as f:
            js=json.load(f)
        p=[]
        for A in js:
            para=A['paragraphs']
            t = ['<START>']
            for seg in para:
                res = re.sub(u'-.*-', '', seg)
                res= re.sub(u'（.*）', '',res)
                res = re.sub(u'{.*}', '', res)
                res = re.sub(u'《.*》', '', res)
                res = re.sub(u'[\]\[]', '', res)#去除[或]
                # 可能出现（和）分别出现在上下句的情况，所以分开处理
                res=re.sub(u'（.*','',res)#去除*****）
                res=re.sub(u'.*）','',res)#去除（*****
                res=re.sub(u'。。+','。',res)
                res=re.sub(u'，，+','，',res)
                res = re.sub(u'，。，', '，', res)
                res=re.sub(u' ','',res)
                for w in res:
                    t.append(w)
            if len(t)<opt.maxlen//10:
                continue
            if len(t) > opt.maxlen-1:
                t=t[:opt.maxlen-1]
            t.append('<END>')
            while len(t)<opt.maxlen:
                t.append('</s>')
            p.append(t)
        return p

    poetry=[]
    for filename in os.listdir(opt.data_path):
        if 'authors' not in filename.split('.')[0] and 'tang' in filename.split('.')[1] :
            poetry.extend(ajson2numpy(opt.data_path+filename))
    #print(poetry[0])
    #从二维list tang提取出词典ix2word,word2ix
    all_words=[word for A in poetry for word in A]#将二维的list连接成一个一维的list
    words_count=Counter(all_words)#词及其词频
    count_pairs=sorted(words_count.items(),key=lambda x:-x[1])#按词频排序
    words=[w[0] for w in count_pairs]#生成词典，接下来用list下标来指代这个词，为了去掉生僻字，只选取前2000个高频词
    words.remove('<START>')
    words.remove('<END>')
    words.remove('</s>')
    word2ix=dict(zip(words,range(0,len(words))))#{词：下标}
    word2ix['<START>']=len(words)
    word2ix['<END>']=len(words)+1
    word2ix['</s>']=len(words)+2
    ix2word={ix:word for word,ix in list(word2ix.items())}
    #用word2ix将原文转成向量
    data=[list(map(lambda w:word2ix.get(w,len(words)+2),A)) for A in poetry]
    #data = [list(map(lambda w: word2ix.get(w, 0), all_words))]
    # print('all_words:',len(all_words),all_words)
    #
    # yu=len(data)%opt.maxlen
    # if yu!=0:
    #     data+=[len(words)]*(opt.maxlen-yu)
    # print('data1.shape:',len(data))
    #print('data1:',data)
    #将data、ix2word、word2ix转成Numpy数组，并打包存储
    data=np.array(data)#这里一定要检查每首诗是不是都是125的长度，否则在转成数组的时候不会变成(57591, 125)
    # data.reshape(-1,opt.maxlen)
    ix2word=np.array(ix2word)
    word2ix=np.array(word2ix)
    #print('data.shape:',data.shape)
    #print('data:',data)
    np.savez(opt.pickle_path,data=data,ix2word=ix2word,word2ix=word2ix)
    return data,ix2word.item(),word2ix.item()

#将序号转成诗歌
def numpy2poetry(opt):
    datas=np.load(opt.pickle_path)
    data=datas['data']
    ix2word=datas['ix2word']
    poem=data[0]
    poem_txt=[ix2word[i] for i in poem]
    print(''.join(poem_txt))

#主程序调用接口
def get_data(opt):#opt为主程序传进的参数，为配置选项，是Config对象
    if os.path.exists(opt.pickle_path):
        datas=np.load(opt.pickle_path)
        data,ix2word,word2ix=datas['data'],datas['ix2word'].item(),datas['word2ix'].item()
        return data,ix2word,word2ix
    else:
        return json2numpy(opt)