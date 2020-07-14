# NCF
He Xiangnan 2017 Neural Collaborative Filtering

http://dx.doi.org/10.1145/3038912.3052569

# Introduce

## 此代码不和源码一致，
  1、提取数据，获取训练数据正样本以及负采样。负采样比例 默认50。如需，可修改DataExtract.py中load_user_item()函数中negative_num。
  2、模型只设计了NCF。
  3、评估采用HQ


# Environment Settings
  pytorch
  
# Example to run the codes.
  ```
  python main.py --path './u1.base' --epoches 101 --batch_size 128 --embedding_size 8 --lamba 0.0 --lr 0.001 --learner 'adam'
  ```
  获取帮助
  ```
  python main.py --help
         -h, --help            show this help message and exit
        --path [PATH]         Input data path, the content format must be [userid,
                              itemid, rate, timestep].
        --epochs EPOCHS       Numbers of epochs.
        --batch_size BATCH_SIZE
                              Batch size.
        --embedding_size EMBEDDING_SIZE
                              Embedding size.
        --lamdba LAMDBA       Regularization size, default is l2 reg.
        --lr LR               Learing Rate.
        --learner [LEARNER]   Specify an optimizer : adagrad, adam, sgd, rmsprop.

  ```
