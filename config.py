class Config(object):
    data_path='data/chinese-poetry-zhCN-master/poetry/'#存放诗歌的路径
    pickle_path='data/poetry5.npz'#预处理好的二进制文件
    debug_file='/tmp/debugp'
    model_path=None#预训练好的模型路径

    lr=1e-3
    use_gpu=True
    epoch=50#20
    batch_size=128#128
    embed_size=128
    hidden_dim=256
    use_env=True#是否使用visom
    env='poetry'#
    plot_every=1#每20个batch可视化一次,20

    author=None #只学习某位作者的诗歌
    constrain=None  #长度限制
    category='poet.tang'#诗歌类别
    maxlen=80#数据集中的诗词每首最长长度
    max_gen_len=80#生成诗歌最长长度

    #生成诗歌相关配置
    prefix_words='黄沙百战金甲'#用来控制诗歌的意境
    start_words='杨柳青青'#诗歌开始
    acrostic=False#是否是藏头诗
    model_prefix='checkpoints/poetry5'#模型保存路径

    
