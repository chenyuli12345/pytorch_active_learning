#!/usr/bin/env python

import torch 
import math #python数学库，包含一些基本数学函数和常数
from random import shuffle #random库的函数，用于随机打乱列表


__author__ = "Robert Munro"
__license__ = "MIT"
__version__ = "1.0.1"

   

class UncertaintySampling():
    
    def __init__(self, verbose=False): #类的构造函数，用于初始化新创建对象的状态（这里是初始化verbose属性），接受一个默认为False的可选参数verbose，用于决定类的实例是否应该在执行某些输出时输出详细信息
        self.verbose = verbose
    

    #最小置信度方法，接受两个参数，第一个是prob_dist是概率分布（张量），第二个sorted是一个布尔值，用于指示概率分布是否已经从大到小排序，默认为False。最后返回最小置信度得分
    def least_confidence(self, prob_dist, sorted=False): 
        """ 
        返回使用最小置信采样计算概率分布的不确定性得分，范围为0-1，其中1是最不确定的。
        假设概率分布是一个pytorch张量，如：tensor([0.0321, 0.6439, 0.0871, 0.2369])
        关键词参数为：
        prob_dist -- 一个实数张量，代表概率分布情况，介于0和1之间，总和为1.0
        sorted -- 如果概率分布是从大到小排序的
        """
        if sorted: #如果sorted为True，即概率分布是从大到小排序的
            simple_least_conf = prob_dist.data[0] # 在排序情况下，最大概率（最有信心的预测）为概率分布张量的第一个元素
        else:
            simple_least_conf = torch.max(prob_dist) #在未排序的情况下，最大概率（最有信心的预测）为概率分布张量的最大值
                    
        num_labels = prob_dist.numel() # 计算概率分布张量的元素数量，即标签的数量
         
        normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels -1)) #计算归一化的最小置信度（最小置信度得分），公式为：(1 - 最大概率) * (标签数量 / (标签数量 - 1))，得分为0-1，1表示最不确定
        
        return normalized_least_conf.item() #返回最小置信度得分，item()方法将张量转换为python数字
    
    #置信度边缘方法，接受两个参数，第一个是prob_dist是概率分布（张量），第二个sorted是一个布尔值，用于指示概率分布是否已经从大到小排序，默认为False。最后返回置信度边缘得分
    def margin_confidence(self, prob_dist, sorted=False):
        """ 
        返回使用置信度边缘方法计算概率分布的不确定性得分，范围为0-1，其中1表示最不确定
        假设概率分布是一个pytorch张量，如：tensor([0.0321, 0.6439, 0.0871, 0.2369])
        关键词参数为：
        prob_dist -- 一个实数张量，代表概率分布情况，介于0和1之间，总和为1.0
        sorted -- 如果概率分布是从大到小排序的
        """
        if not sorted: #如果sorted为False，即概率分布不是从大到小排序的
            prob_dist, _ = torch.sort(prob_dist, descending=True) # 采用函数torch.sort()对概率分布进行排序（降序），_接受该函数返回的第二个值，即原始数据的索引（不需要）
        
        difference = (prob_dist.data[0] - prob_dist.data[1]) # 计算最大概率和第二大概率之间的差值
        margin_conf = 1 - difference #计算置信度边缘得分，公式为：1-差值，得分为0-1，1表示最不确定
        
        return margin_conf.item() #返回置信度边缘得分，item()方法将张量转换为python数字
        
    #置信度比率方法，接受两个参数，第一个是prob_dist是概率分布（张量），第二个sorted是一个布尔值，用于指示概率分布是否已经从大到小排序，默认为False。最后返回置信度比率得分
    def ratio_confidence(self, prob_dist, sorted=False):
        """ 
        返回使用置信度比率方法计算概率分布的不确定性得分，范围为0-1，其中1表示最不确定。
        假设概率分布是一个pytorch张量，如：tensor([0.0321, 0.6439, 0.0871, 0.2369])
        关键词参数为：
        prob_dist -- 一个实数张量，代表概率分布情况，介于0和1之间，总和为1.0
        sorted -- 如果概率分布是从大到小排序的
        """
        if not sorted: #如果sorted为False，即概率分布不是从大到小排序的
            prob_dist, _ = torch.sort(prob_dist, descending=True) # 采用函数torch.sort()对概率分布进行排序（降序），_接受该函数返回的第二个值，即原始数据的索引（不需要）
            
        ratio_conf = prob_dist.data[1] / prob_dist.data[0] # 计算次大概率与最大概率的比值，得分为0-1，1表示最不确定
        
        return ratio_conf.item() #返回置信度比率得分，item()方法将张量转换为python数字
    
    #熵方法，接受两个参数，第一个是prob_dist是概率分布（张量），第二个sorted是一个布尔值，用于指示概率分布是否已经从大到小排序，默认为False。最后返回熵得分
    def entropy_based(self, prob_dist):
        """ 
        返回使用熵方法计算概率分布的不确定性得分，范围为0-1，其中1表示最不确定。
        假设概率分布是一个pytorch张量，如：tensor([0.0321, 0.6439, 0.0871, 0.2369])
        关键词参数为：
        prob_dist -- 一个实数张量，代表概率分布情况，介于0和1之间，总和为1.0
        sorted -- 如果概率分布是从大到小排序的
        """
        log_probs = prob_dist * torch.log2(prob_dist) # 先计算概率分布张量中每个元素（概率）的对数，然后与原概率分布张量相乘
        raw_entropy = 0 - torch.sum(log_probs) # 计算原始熵，公式为：0 - 上一步得到的张量的和
    
        normalized_entropy = raw_entropy / math.log2(prob_dist.numel()) #将原始熵除以原概率分布张量中的元素数量的对数（以2为底），得到归一化的熵。这样做是为了将熵的值映射到 0-1 的范围，其中 1 表示最不确定。
        
        return normalized_entropy.item() #返回熵得分，item()方法将张量转换为python数字
        
 
    #softmax方法，接受两个参数，第一个是scores是模型最后一层输出的一组原始分数（logits），第二个是base是指数的底数，默认为e。最后返回概率分布
    def softmax(self, scores, base=math.e):
        """
        将模型的一组原始分数（logits）通过softmax转换为概率分布
        假设概率分布是一个pytorch张量，如：tensor([0.0321, 0.6439, 0.0871, 0.2369])
        关键词参数为：
        prediction -- 一个包含任意正/负实数的张量
        base -- 指数的底数，默认为e
        """
        exps = (base**scores.to(dtype=torch.float)) # 先将scores列表转换为浮点数类型的张量，然后计算每个元素的指数
        sum_exps = torch.sum(exps) # 计算指数化的张量exps的所有元素的和

        prob_dist = exps / sum_exps # 将指数化的张量每个元素除以上一步得到的和，得到概率分布（归一化的指数）
        return prob_dist #返回概率分布张量，每个元素的范围为0-1，总和为1.0
        
   
        
        
    def get_samples(self, model, unlabeled_data, method, feature_method, number=5, limit=10000):
        """
        通过给定的不确定性采样方法从未标记数据中获取样本
        关键词参数：
        - `model` —— 当前任务的机器学习模型
        - `unlabeled_data` —— 尚未标记的数据
        - `method` —— 不确定性采样的方法（例如：least_confidence()）
        - `feature_method` —— 从数据中提取特征的方法
        - `number` —— 需要采样的项目数量
        - `limit` —— 仅从此数量的预测中进行采样以加快速度（-1表示无限制）

        返回根据最小置信度方法得到的最不确定的项
        """
        samples = [] #初始化一个空列表，用于存储采样的数据
    
        if limit == -1 and len(unlabeled_data) > 10000 and self.verbose: # 如果limit为-1，且未标记数据的数量大于10000，且verbose为True
            print("Get predictions for a large amount of unlabeled data: this might take a while") #打印：获取大量未标记数据的预测：这可能需要一段时间。
        else: 
            # only apply the model to a limited number of items                                                                            
            shuffle(unlabeled_data) #打乱未标记数据的顺序
            unlabeled_data = unlabeled_data[:limit] #将unlabeled_data截取到limit的长度
        
        with torch.no_grad():#下面的计算不需要计算梯度
            v=0
            for item in unlabeled_data: #对于未标记数据中的每一项
                text = item[1] #获取未标记数据的第二项，记为text
                
                feature_vector = feature_method(text) #用传入参数的feature_method方法将text转换为特征向量
                hidden, logits, log_probs = model(feature_vector, return_all_layers=True) #用模型对特征向量进行预测，返回所有层的隐藏状态、原始分数和概率分布
    
                prob_dist = torch.exp(log_probs) # 将模型输出得到的对数概率分布乘上e，得到概率分布
                
                #data方法返回与原张量共享内存的新张良，但不会加入原张量的计算历史中
                score = method(prob_dist.data[0]) #首先从上面得到的概率分布中取出第一个元素，然后用传入的method方法计算得分
                
                item[3] = method.__name__ # 将method方法的名称赋值给item的第四个元素
                item[4] = score # 将上面得到的不确定度得分赋值给item的第五个元素
                
                samples.append(item) #将item添加到samples列表中
                
                
        samples.sort(reverse=True, key=lambda x: x[4]) #使用sort方法对迭代得到的samples列表进行排序，reverse=True表示降序，key=lambda x: x[4]表示按照item的第五个元素（即不确定度得分）进行排序    
        return samples[:number:] #返回samples的前number个项
        
    

        
