#!/usr/bin/env python

"""DIVERSITY SAMPLING
 
Diversity Sampling examples for Active Learning in PyTorch 

This is an open source example to accompany Chapter 4 from the book:
"Human-in-the-Loop Machine Learning"

It contains four Active Learning strategies:
1. Model-based outlier sampling
2. Cluster-based sampling
3. Representative sampling
4. Adaptive Representative sampling


"""

import torch
import torch.nn as nn #pytorch构建神经网络的库
import torch.nn.functional as F #pytorch神经网络函数库，如激活函数和损失函数
import torch.optim as optim #pytorch优化器库，包含常见的优化算法，如SGD、Adam等
import random #python随机数库
import math #python数学库，包含一些基本数学函数和常数
import datetime #python日期时间库
import csv #python csv文件读写库，用于读写csv文件
import re #python正则表达式库，用于处理和匹配字符串
import os #python操作系统库，用于操作文件和目录
import getopt, sys #getopt库用于解析命令行参数，sys库用于访问与Python解释器交互的变量

from random import shuffle #random库的函数，用于随机打乱列表
#collections库是Python内建的一个集合模块，实现了特殊的容器数据类型，提供了 Python 内置的通用数据类型（dict、list、set、tuple）的替代选择
from collections import defaultdict #字典的子类，提供了一个工厂函数，为字典查询提供了默认值 

# from numpy import rank

from uncertainty_sampling import UncertaintySampling #从uncertainty_sampling.py文件中导入UncertaintySampling类
from pytorch_clusters import CosineClusters #从pytorch_clusters.py文件中导入CosineClusters类
from pytorch_clusters import Cluster #从pytorch_clusters.py文件中导入Cluster类

if sys.argv[0] == "diversity_sampling.py":
    import active_learning


__author__ = "Robert Munro"
__license__ = "MIT"
__version__ = "1.0.1"

   
class DiversitySampling():


    def __init__(self, verbose=False):
        self.verbose = verbose


    
    def get_cluster_samples(self, data, num_clusters=5, max_epochs=5, limit=5000):
        """
        使用余弦相似度创建聚类
        关键词参数：
        - `data` -- 要进行聚类的数据
        - `num_clusters` -- 要创建的聚类数量
        - `max_epochs` -- 创建聚类的最大轮次
        - `limit` -- 仅从指定数量的项目中进行采样以加快聚类速度（-1表示无限制）
        使用K-Means聚类算法创建聚类，采用余弦相似度而不是更常见的欧氏距离
        创建聚类直到收敛或达到最大轮次  
        """ 
        
        if limit > 0: #如果设置了采样限制
            shuffle(data) #打乱数据
            data = data[:limit] #只取前limit个数据
        
        cosine_clusters = CosineClusters(num_clusters) #创建余弦相似度聚类群对象，包含num_clusters个聚类
        
        cosine_clusters.add_random_training_items(data) #按顺序将data中的项依次添加到每个聚类中
        
        #迭代max_epochs次，将data中的项添加到最佳聚类中
        for i in range(0, max_epochs): #迭代max_epochs次
            print("Epoch "+str(i)) #打印当前轮次
            added = cosine_clusters.add_items_to_best_cluster(data) #将data中的项添加到最佳聚类中，返回的added为移动到新聚类的项的数目
            if added == 0: #如果没有项移动到新聚类中，则提前结束迭代
                break
        
        #获取更新后的聚类群的中心（一个列表，每个元素为挨个聚类的中心项，列表每个元素也是一个列表，列表第一个元素是中心值对应的项/语句的id，第二项为对应的项/语句，第四个元素为"cluster_centroid"，第五个元素为最佳匹配度值（一个余弦相似度值））
        centroids = cosine_clusters.get_centroids()
        #获取更新后的聚类群的异常（一个列表，每个元素为挨个聚类的异常项，列表每个元素也是一个列表，第一个元素是异常值对应的项/语句的id，第二项为对应的项/语句，第四个元素为"cluster_outlier"，第五个元素为1-最大异常度值（一个余弦相似度值））
        outliers = cosine_clusters.get_outliers()
        #获取更新后的聚类群的3个随机项列表
        randoms = cosine_clusters.get_randoms(3, self.verbose)
        
        return centroids + outliers + randoms
             
    
    def get_representative_samples(self, training_data, unlabeled_data, number=20, limit=10000):
        """
        获取相对于训练数据最具代表性的未标记项
        关键词参数：
        - `training_data` -- 带有标签的数据，是当前模型训练使用的数据
        - `unlabeled_data` -- 尚未标记的数据
        - `number` -- 要采样的项目数量
        - `limit` -- 仅从指定数量的项目中进行采样以加快速度（-1表示无限制）
        为每个数据集（训练数据和未标记数据）创建一个聚类
        """ 
        #根据采样限制截取数据
        if limit > 0: #如果设置了采样限制
            shuffle(training_data) #打乱训练数据
            training_data = training_data[:limit] #只取前limit个训练数据
            shuffle(unlabeled_data) #打乱未标记数据
            unlabeled_data = unlabeled_data[:limit] #只取前limit个未标记数据
        
        #将训练数据添加到一个聚类training_cluster中
        training_cluster = Cluster() #创建一个聚类对象，用于存储训练数据
        for item in training_data: #对于训练数据中的每一项
            training_cluster.add_to_cluster(item) #将训练数据的每一项都添加到该聚类中
        
        #将未标记数据添加到一个聚类unlabeled_cluster中
        unlabeled_cluster = Cluster() #创建一个聚类对象，用于存储未标记数据
        for item in unlabeled_data: #对于未标记数据中的每一项
            unlabeled_cluster.add_to_cluster(item) #将未标记数据的每一项都添加到该聚类中
    
        
        for item in unlabeled_data:#对于未标记数据中的每一项
            training_score = training_cluster.cosine_similary(item) #计算未标记数据的每一项与储存训练数据的聚类的余弦相似度
            unlabeled_score = unlabeled_cluster.cosine_similary(item) #计算未标记数据的每一项与储存未标记数据的聚类的余弦相似度
                        
            representativeness = unlabeled_score - training_score #计算未标记数据的每一项的代表性，即未标记数据的每一项与储存未标记数据的聚类的余弦相似度减去未标记数据的每一项与储存训练数据的聚类的余弦相似度
            
            item[3] = "representative" #将未标记数据的每一项的第四个元素设为"representative"        
            item[4] = representativeness #将未标记数据的每一项的第五个元素设为该项的代表性值
                
        #迭代结束后，未标记数据集被更新，其每一项都是一个数据/列表，列表第一项是数据/列表的id，第二项是数据/列表的文本，第四项是"representative"，第五项是该项的代表性值
        unlabeled_data.sort(reverse=True, key=lambda x: x[4])  #对未标记数据集进行（key是一个函数用于在排序过程中对某个元素进行某种计算；lambda是匿名函数，接受参数x，返回x[4]），按照每个元素第五个元素（代表性值）降序排列
        return unlabeled_data[:number:]  #获取未标记数据集unlabled_data的前number个项  
    
    
    def get_adaptive_representative_samples(self, training_data, unlabeled_data, number=20, limit=5000):
        """
        自适应获取与训练数据相比最具代表性的未标记项
        关键词参数：
        - `training_data` -- 带有标签的数据，是当前模型训练使用的数据
        - `unlabeled_data` -- 尚未标记的数据
        - `number` -- 要采样的项目数量
        - `limit` -- 仅从指定数量的项目中进行采样以加快速度（-1表示无限制）
        该函数是上面的get_representative_samples()函数的自适应版本，其中在每次选择后更新训练数据以增加样本的多样性
        """
        
        samples = [] #初始化一个空列表，用于存储采样的数据
        
        for i in range(0, number): #迭代number次
            print("Epoch "+str(i)) #打印当前轮次
            representative_item = self.get_representative_samples(training_data, unlabeled_data, 1, limit)[0] #调用上面的get_representative_samples()函数，获取未标记数据集中与训练数据集最具代表性的项的第一个元素（项/文本的id）
            samples.append(representative_item) #将获取的项的id添加到samples列表中
            #remove函数接受一个参数，用于删除列表中的某个元素
            unlabeled_data.remove(representative_item) #将获取的项从未标记数据集中删除，因为该项已经被选为训练数据
            
        return samples
    
    
    
    def get_validation_rankings(self, model, validation_data, feature_method):
        """Get model outliers from unlabeled data 
    
        Keyword arguments:
            model -- current Machine Learning model for this task
            unlabeled_data -- data that does not yet have a label
            validation_data -- held out data drawn from the same distribution as the training data
            feature_method -- the method to create features from the raw text
            number -- number of items to sample
            limit -- sample from only this many items for faster sampling (-1 = no limit)
    
        An outlier is defined as 
        unlabeled_data with the lowest average from rank order of logits
        where rank order is defined by validation data inference 
    
        """
                
        validation_rankings = [] # 2D array, every neuron by ordered list of output on validation data per neuron    
    
        # Get per-neuron scores from validation data
        if self.verbose:
            print("Getting neuron activation scores from validation data")
    
        with torch.no_grad():
            v=0
            for item in validation_data:
                textid = item[0]
                text = item[1]
                
                feature_vector = feature_method(text)
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True)  
        
                neuron_outputs = logits.data.tolist()[0] #logits
                
                # initialize array if we haven't yet
                if len(validation_rankings) == 0:
                    for output in neuron_outputs:
                        validation_rankings.append([0.0] * len(validation_data))
                            
                n=0
                for output in neuron_outputs:
                    validation_rankings[n][v] = output
                    n += 1
                            
                v += 1
        
        # Rank-order the validation scores 
        v=0
        for validation in validation_rankings:
            validation.sort() 
            validation_rankings[v] = validation
            v += 1
          
        return validation_rankings 
    
    def rt(str):
    	return str
    
    def get_rank(self, value, rankings):
        """ get the rank of the value in an ordered array as a percentage 
    
        Keyword arguments:
            value -- the value for which we want to return the ranked value
            rankings -- the ordered array in which to determine the value's ranking
        
        returns linear distance between the indexes where value occurs, in the
        case that there is not an exact match with the ranked values    
        """
        
        index = 0 # default: ranking = 0
        
        for ranked_number in rankings:
            if value < ranked_number:
                break #NB: this O(N) loop could be optimized to O(log(N))
            index += 1        
        
        if(index >= len(rankings)):
            index = len(rankings) # maximum: ranking = 1
            
        elif(index > 0):
            # get linear interpolation between the two closest indexes 
            
            diff = rankings[index] - rankings[index - 1]
            perc = value - rankings[index - 1]
            linear = perc / diff
            index = float(index - 1) + linear
        
        absolute_ranking = index / len(rankings)
    
        return(absolute_ranking)
    
                
    
    def get_model_outliers(self, model, unlabeled_data, validation_data, feature_method, number=5, limit=10000):
        """Get model outliers from unlabeled data 
    
        Keyword arguments:
            model -- current Machine Learning model for this task
            unlabeled_data -- data that does not yet have a label
            validation_data -- held out data drawn from the same distribution as the training data
            feature_method -- the method to create features from the raw text
            number -- number of items to sample
            limit -- sample from only this many items for faster sampling (-1 = no limit)
    
        An outlier is defined as 
        unlabeled_data with the lowest average from rank order of logits
        where rank order is defined by validation data inference 
    
        """
    
        # Get per-neuron scores from validation data
        validation_rankings = self.get_validation_rankings(model, validation_data, feature_method)
        
        # Iterate over unlabeled items
        if self.verbose:
            print("Getting rankings for unlabeled data")
    
        outliers = []
        if limit == -1 and len(unlabeled_data) > 10000 and self.verbose: # we're drawing from *a lot* of data this will take a while                                               
            print("Get rankings for a large amount of unlabeled data: this might take a while")
        else:
            # only apply the model to a limited number of items                                                                            
            shuffle(unlabeled_data)
            unlabeled_data = unlabeled_data[:limit]
    
        with torch.no_grad():
            for item in unlabeled_data:
                text = item[1]
    
                feature_vector = feature_method(text)
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True)            
                
                neuron_outputs = logits.data.tolist()[0] #logits
                   
                n=0
                ranks = []
                for output in neuron_outputs:
                    rank = self.get_rank(output, validation_rankings[n])
                    ranks.append(rank)
                    n += 1 
                
                item[3] = "logit_rank_outlier"
                
                item[4] = 1 - (sum(ranks) / len(neuron_outputs)) # average rank
                
                outliers.append(item)
                
        outliers.sort(reverse=True, key=lambda x: x[4])       
        return outliers[:number:]       
      
                

    
