import sys,os

from config import Config
# from trains.lstm_train import train, test
from trains.seq2seq_train import train
if __name__ == '__main__':#不加这行会报错，这是window下运行的特殊情况
    #开始前需要先启动visdom：python -m visdom.server,然后打开浏览器输入:localhost:8097
    opt=Config()
    train(opt)
    #start_words='床前明月光'
    # test(opt)