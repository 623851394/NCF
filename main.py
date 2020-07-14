#!/usr/bin/env python
# coding: utf-8

# In[20]:


import torch
import torch.nn as nn
import argparse
import time

from torch.utils import data
from DataExtract import DataExtract
from DataSet import TrainDataSet, TestDataSet
from NCF import NeuMF 


# In[21]:


def evaluate_HQ(get_items, label):
    if label in get_items:
        return 1
    return 0


# In[22]:


def parse_agrs():
    parser = argparse.ArgumentParser(description="Run NCf")
    parser.add_argument('--path', nargs='?', default='./u1.base',
                       help='Input data path, the content format must be [userid, itemid, rate, timestep].')
    parser.add_argument('--epochs', type=int, default=101,
                       help='Numbers of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size.')
    parser.add_argument('--embedding_size', type=int, default=8,
                       help='Embedding size.')
    parser.add_argument('--lamdba', type=float, default=0.0,
                       help='Regularization size, default is l2 reg.')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learing Rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                       help='Specify an optimizer : adagrad, adam, sgd, rmsprop.')
    
    return parser.parse_args()
    


# In[24]:


if __name__ == '__main__':
    args = parse_agrs()
    
    path = args.path
    epochs = args.epochs
    batch_size = args.batch_size
    embedding_size = args.embedding_size
    lamdba = args.lamdba
    lr = args.lr
    learner = args.learner
    
    topK = 10   # 测HQ
    
    print("NCF arguments: %s" % (args))
    
    print("start time :", time.asctime( time.localtime(time.time()) ))
    
    t1 = time.time()
    
    dataextract = DataExtract(path)
    print("Data extract finished!!", time.asctime( time.localtime(time.time()) ))
    train_data = TrainDataSet(dataextract.train_user_item, dataextract.train_label)
    test_data  = TestDataSet(dataextract.test_user_label, dataextract.test_user_item)
    train_loader = data.DataLoader(train_data, batch_size= batch_size, shuffle=True)
    test_loader  = data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    print("Data Loaded finished!!", time.asctime( time.localtime(time.time()) ))
    
    model = NeuMF(dataextract.max_user_id, dataextract.max_item_id, emb_dim=embedding_size)
    device = torch.device("cpu")
    model.to(device)
    # 设置合适的损失函数和优化器
    criterion = nn.BCELoss()
    if learner == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lamdba)
    elif learner == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr,weight_decay=lamdba)
    elif learner == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=lamdba)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,weight_decay=lamdba)
    
    train_losses = []
    HQs = []
    print("start traing!!", time.asctime( time.localtime(time.time()) ))
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for user, item, label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.to(device)
            label = torch.as_tensor(label, dtype=torch.float32)
            out = model(user, item)
#             print(out.dtype, label.dtype)
            loss = criterion(out.view(len(out)), label.view(len(label)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader) )
            
            
        if epoch % 20 == 0:
            print("start test!!", time.asctime( time.localtime(time.time()) ))
            model.eval()
            test_loss = 0
            hq = 0
            for user, item, label in test_loader:
                
                out = []
                user = user.view(-1, 1)
                item = item.view(-1, 1)
                user = user.to(device)
                item = item.to(device)
                
                out = model(user,item)
                
                val, inx = torch.topk(out.view(1, -1), topK)
                hq  += evaluate_HQ(inx, label)
            
            HQs.append(hq / len(test_loader))
            print("epoch:{}, train_loss is {:.4f}, HQ is {:.4f}".format(epoch,
                                                                        train_loss / len(train_loader),
                                                                       hq / len(test_loader)))
    
    print("finished!!", time.asctime( time.localtime(time.time()) ))
   




