参考[《深度学习框架 PyTorch入门与实践》](https://github.com/Amberian/pytorch-book/tree/master/chapter9-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%86%99%E8%AF%97(CharRNN))用LSTM生成古诗
===
数据来源：[复旦大学中文古诗简体库](https://github.com/chinese-poetry/chinese-poetry-zhCN)，将其chinese-poetry-zhCN-master文件夹放入本项目data文件夹下

文件说明
---
checkpoints/: 保存训练好的模型

data/: 包括数据预处理、dataset实现等

models/: 模型定义

results/: 用以记录生成的古诗，本项目简单的用txt记录

config.py: 配置文件，包括所有可配置的变量及其默认值，本项目在需要参数的地方基本都直接调用的这里的参数

main.py: 主文件，训练和测试的入口

requirements.txt: 程序依赖的第三方库

程序运行
---
在main函数中直接修改，选择train或者test,test的时候可以修改start_words或者不写直接用默认值

```ython

if __name__ == '__main__':#不加这行会报错，这是window下运行的特殊情况
    opt=Config()
    #train(opt)
    #start_words='床前明月光'
    test(opt)
```
注意：由于在程序中需要运行visdom，所以需要在运行main之前，先在命令行中开启visdom服务器
```python
python -m visdom.server
```

具体程序中都有相关注释，这里就不赘述

生成效果
---
这里挑了一些好的，但是不一定都能这样，还是要多试，发现用常见的四字作为start_words生成效果比较好
```text
一、prefix_words='床前明月光，疑似地上霜'
1. start_words:杨柳青青
生成：杨柳青青，翠黛红唇无影态，红颜绿色如花烟。相逢不见相思处，空忆长安万里中

2. start_words=山寺桃花
生成：山寺桃花，竹枝风雨香风。此日一年无限思，春风不改花前花。

二、prefix_words='黄沙百战金甲'
1. start_words=山雨
生成：山雨白，江城绿柳黄金勒。马嘶风雨过，行人去心远。

2. start_words=杨柳青青
生成：杨柳青青树色，白日如花不见。今日一枝风，莫愁憔悴泪，暮春春又归。

3. start_words=千树万树梨花开
生成：千树万树梨花开，罗衣锦带垂朱旗。美人歌，马蹄回，金鱼歌舞歌。何人唤唱歌舞，不见君王娇妒多。

```