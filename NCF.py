#!/usr/bin/env python
# coding: utf-8

# In[20]:


import torch
from torch.utils import data
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from DataSet import TrainDataSet
import DataSet
from DataExtract import DataExtract


# In[71]:


class NeuMF(nn.Module):
    
    def __init__(self, user_num, item_num, emb_dim=8, layer_one=64, layer_two=32, layer_three=16, layer_out=8,):
        super(NeuMF, self).__init__()
        self.emb_dim = emb_dim
        self.input_dim = emb_dim * 2
        self.layer_one = layer_one
        self.layer_two = layer_two
        self.layer_three= layer_three
        self.layer_out = layer_out
        self.user_num  = user_num
        self.item_num  = item_num
        
        
        self.MLP = nn.Sequential(
                            # 16->64>32>16->8
            nn.Linear(self.input_dim, self.layer_one),
            nn.Linear(self.layer_one, self.layer_two),
            nn.Linear(self.layer_two, self.layer_three),
            nn.Linear(self.layer_three, self.layer_out),
        )
        self.user_GMF_emb = nn.Embedding(self.user_num+1, self.emb_dim)
        self.user_MLP_emb = nn.Embedding(self.user_num+1, self.emb_dim)
        
        self.item_GMF_emb = nn.Embedding(self.item_num+1, self.emb_dim)
        self.item_MLP_emb = nn.Embedding(self.item_num+1, self.emb_dim)
        
        self.out = nn.Linear(self.input_dim, 1)
    def forward(self, user, item):
        user = torch.as_tensor(user, dtype=torch.long)
        item = torch.as_tensor(item, dtype=torch.long)
        
        ## GMF
        user_g_emb = self.user_GMF_emb(user)
        item_g_emb = self.item_GMF_emb(item)
        user_g_F = user_g_emb.view(len(user), -1)
        item_g_F = item_g_emb.view(len(item), -1)
        G_out = torch.mul(user_g_F, item_g_F)
        ## MLP
        user_m_F = self.user_MLP_emb(user).view(len(user), -1)
        
        item_g_F = self.item_MLP_emb(item).view(len(item), -1)
        user_item = torch.cat((user_m_F, item_g_F), axis=1)
        M_out = self.MLP(user_item)

        GMF_MLP_cat = torch.cat((G_out, M_out),axis=1)
        
        out = torch.sigmoid(self.out(GMF_MLP_cat))
        
        return out







