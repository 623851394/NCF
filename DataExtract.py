#!/usr/bin/env python
#coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os

class DataExtract():
    def __init__(self, filename, isFilter=True):
        '''
            isFilter: 是否过滤一些评分小于20个电影的用户
            max_user_id: 用户数量
            max_item_id:
            rate_info: 评分信息 用户id、电影id、评分（1--5）
            rating_Matrix：用户与电影的评分经过one_hot处理，即有评分为1， 没有评分为0
            train_user_item：用户id， 电影id。
            train_label: 训练集标签
            test_user_item：测试集：用户没有评论过的电影的负采样99个
            test_user_label：用与预测，用户喜欢的一个电影。
            
            load_rate_info():加载评分信息
            load_rate_as_Matrix(): 将评分矩阵经过one_hot处理，即有评分为1， 没有评分为0
            filter_rate_Matrix();是否过滤一些评分小于20个电影的用户
            load_user_item(): 获得训练数据 和测试集数据
        '''
        
        
        
        self.filename = filename

        self.max_user_id = 0
        self.max_item_id = 0
        self.isFilter = isFilter
        self.rate_info = self.load_rate_info()
        self.rating_Matrix = self.load_rate_as_Matrix()
        
        if self.isFilter:
            self.rating_Matrix = self.filter_rate_Matrix()
            self.max_user_id = self.rating_Matrix.shape[0]
            self.max_item_id = self.rating_Matrix.shape[1]
        self.train_user_item, self.train_label, self.test_user_item, self.test_user_label = self.load_user_item()
        
    def load_rate_info(self,):
        rate_Matrix = []
        
        with open(self.filename, 'r') as f:
            txt = f.readlines()
            for t in txt:
                arr = t.split('\t')
                user, item, rate = int(arr[0]), int(arr[1]),int(arr[2])
                self.max_user_id, self.max_item_id = max(self.max_user_id, user), max(self.max_item_id, item)
                rate_Matrix.append([user, item, rate])
        return rate_Matrix
    
    def load_rate_as_Matrix(self,):

        rating_Matrix = np.zeros([self.max_user_id, self.max_item_id])
        for arr in self.rate_info:
            if arr[2] > 1:
                rating_Matrix[arr[0]-1, arr[1]-1] = 1
        
        return rating_Matrix
        
    def filter_rate_Matrix(self, ):
        Matrix = []
        rate_sum = self.rating_Matrix.sum(axis=1)
        arr_bool = rate_sum > 20
        for index, val in enumerate(arr_bool):
            if val:
                Matrix.append(self.rating_Matrix[index])
        return np.array(Matrix)
    
    def load_user_item(self, ):
        user_item = []
        label = []
        test_user_item = []
        test_label = []
        for i in range(self.max_user_id):
            temp_test = []
            isTest = True
            negative_num = 50
            for j in range(self.max_item_id):
                if self.rating_Matrix[i][j]:
                    user = i + 1
                    item = j + 1
                    
                    judge = random.choice([False, True, False, False])
                    if isTest and judge:
                        test_label.append([user, item])
                        isTest = False
                        self.rating_Matrix[i][j] = 0
                    else:
                        user_item.append([user, item])
                        label.append(1)
                elif random.choice([False, True]) and len(temp_test) < 99:
                    user = i + 1
                    item = j + 1
                    temp_test.append([item])
                    
                    ## 添加负采样 
                    if random.choice([False, True,False]) and negative_num > 0:
                        user_item.append([user, item])
                        label.append(0)
                        negative_num -= 1
                    
            test_user_item.append(temp_test)
                
        return np.array(user_item), np.array(label), np.array(test_user_item).reshape(-1, 99), np.array(test_label)
        
        
        

if __name__ == '__main__':
    pass


