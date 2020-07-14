#!/usr/bin/env python
# coding: utf-8

# In[118]:


import numpy as np
from DataExtract import DataExtract

import torch
import torch.nn as nn
from torch.utils import data 

class TrainDataSet(data.Dataset):
    
    def __init__(self, user_item,label):
        self.data = user_item
        self.label = label
        
        
    def __getitem__(self,index):
        user = self.data[index][0]
        item = self.data[index][1]
        
        label = self.label[index]
        
        label = torch.as_tensor(label,dtype=torch.long)
        return user, item, label
        
    def __len__(self, ):
        return len(self.data)
    

class TestDataSet(data.Dataset):
    
    def __init__(self, user_label, item):
        '''
            user_item: 用户，和喜欢的电影标签
            item ： 用户不喜欢电影的集合
            user_item: 用户数量
            item_item:电影数量
        '''
        
        self.data = user_label
        self.item = item
        
    def __getitem__(self, index):
        user = torch.tensor(self.data[index][0])
        
        items = np.concatenate((self.data[index, 1].reshape(1, 1), self.item[index].reshape(1, 99)),axis=1)
        items = np.random.permutation(items[0])
        items = torch.from_numpy(items)
        
#         找到 用户喜欢电影的下标
        for inx, val in enumerate(items):
            if val == self.data[index, 1]:
                label = inx  
                break
        users = [user.item() for i in range(len(items))]
        users = torch.as_tensor(users, dtype=torch.float32)
        return users, items, label
        
    
    def __len__(self, ):
        return len(self.data)
    
    
#     def trans_data(self,index):
#         user = self.data[index, 0]
#         user_onehot = torch.zeros([1, self.user_num])
#         user_onehot[0, user - 1] = 1
#         items = np.concatenate((self.data[index, 1].reshape(1, 1), self.item[index].reshape(1, 99)),axis=1)
#         ## 随机打乱
#         items = np.random.permutation(items[0])
#         item_onehots = []
        
#         for inx, val in enumerate(items):
            
#             item_onehot = [0 for i in range(self.item_num)]
#             if val == self.data[index, 1]:
#                 label = inx           ## 获取用户感兴趣得电影对应的下标
#             item_onehot[val - 1] = 1
#             item_onehots.append(item_onehot)
        
#         return user_onehot, torch.Tensor(item_onehots), torch.tensor(label)
        



