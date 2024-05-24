#!/usr/bin/env python

"""
用于KMeans类型聚类的余弦距离核
共有三种方法：
1.基于聚类的采样
2.代表性采样
3.主动式代表性采样
"""
import torch
import torch.nn.functional as F #pytorch神经网络函数库，如激活函数和损失函数
from random import shuffle #random库的函数，用于随机打乱列表

#表示数据集上的一组聚类，采用余弦相似度作为度量。是一个聚类群
class CosineClusters():    
    #构造函数，初始化一组聚类和一个字典来存储每个项的聚类
    def __init__(self, num_clusters=100):  #接受一个参数num_clusters，用于指定聚类的数量，默认为100
        
        self.clusters = [] # 初始化聚类列表为空列表，用于存储所有的聚类，空包含num_clusters个聚类
        self.item_cluster = {} # 初始化项聚类字典为空字典，存储每一项的聚类，键为每一项的id，值为该项所在的聚类

        # 创建初始化的聚类
        for i in range(0, num_clusters): #循环num_clusters次（聚类的数量）
            self.clusters.append(Cluster()) #将一个新的聚类添加到聚类列表中
        
     #按顺序将项目添加到聚类中。items包含多个item，每个item第一个数据为文本id，第二个数据为文本，依次将每一个项添加到聚类中，当所有聚类都添加了一个项后再从第一个聚类进一步添加，直到所有项都添加到聚类中
    def add_random_training_items(self, items):
        cur_index = 0 #初始化变量cur_index为0，用于跟踪当前要添加到哪个聚类中
        for item in items: #对于items中的每一个项
            self.clusters[cur_index].add_to_cluster(item) #将当前项添加到当前索引指向的聚类中
            textid = item[0] #获取当前项的id
            self.item_cluster[textid] = self.clusters[cur_index] #将当前项的id作为键，值为当前项所在的聚类
            
            cur_index += 1 #索引加一
            if cur_index >= len(self.clusters): #如果索引超过了聚类的数量
                cur_index = 0  #将索引重置为零

    #将多个项目添加到最佳聚类中。它遍历每个项目，并将其添加到最适合的聚类中（并且更新聚类字典item_cluster）。最后返回的是一个数值，表示添加到新聚类中的项的数量
    def add_items_to_best_cluster(self, items):
        added = 0 #初始化变量added为0，用于追踪已经成功添加到聚类中的项的数量
        for item in items: #对于items中的每一个项
            new = self.add_item_to_best_cluster(item) #依次将当前项添加到最适合的聚类中，返回值是true或false，表示项目是否移动到新的聚类中
            if new: #如果项目移动到了新的聚类中，则将added加一
                added += 1 
                
        return added 


    #找到最适合给定项目的聚类。它遍历每个聚类，并计算给定项目与聚类的余弦相似度，最后返回了一个列表，第一个元素为最佳匹配的聚类，第二个元素最佳相似度。不更新聚类字典item_cluster和聚类列表clusters
    def get_best_cluster(self, item):
        best_cluster = None #初始化最佳聚类为空
        best_fit = float("-inf")  #初始化最佳匹配度为负无穷      
             
        for cluster in self.clusters: #对于聚类列表中的每一个聚类
            fit = cluster.cosine_similary(item) #依次计算输入项与每个聚类的余弦相似度
            if fit > best_fit: #如果当前聚类的余弦相似度大于最佳匹配度
                best_fit = fit #更新最佳匹配度为当前聚类的余弦相似度
                best_cluster = cluster #更新最佳聚类为当前聚类 
        #迭代结束后，best_cluster为与输入项最适合的聚类，best_fit为输入项的最佳匹配度

        return [best_cluster, best_fit]
    
       
    #将项添加到最适合的聚类中。如果项目已经存在于一个聚类中，它将首先从那个聚类中移除，然后添加到最适合的聚类中，并且更新聚类字典item_cluster。返回值是true或false，表示项目是否移动到新的聚类中
    def add_item_to_best_cluster(self, item):
        best_cluster = None #初始化最佳聚类为空
        best_fit = float("-inf") #初始化最佳匹配度为负无穷
        previous_cluster = None #初始化前一个聚类为空
        
        #如果项目已经存在于一个聚类中，则将其从那个聚类中移除
        textid = item[0] #获取输入项的id
        if textid in self.item_cluster: #如果该项的id在聚类字典item_cluster中
            previous_cluster = self.item_cluster[textid] #获取该id对应的聚类
            previous_cluster.remove_from_cluster(item) #从该聚类中移除该项

        #搜寻与该项余弦相似度最高的聚类    
        for cluster in self.clusters: #对于聚类列表中的每一个聚类
            fit = cluster.cosine_similary(item) #依次计算输入项与每个聚类的余弦相似度
            if fit > best_fit: #如果当前聚类的余弦相似度大于最佳匹配度
                best_fit = fit #更新最佳匹配度为当前聚类的余弦相似度
                best_cluster = cluster #更新最佳聚类为当前聚类
        #迭代结束后，best_cluster为与输入项最适合的聚类，best_fit为输入项的最佳匹配度
        
        #将输入项添加到最适合的聚类中，并更新聚类字典item_cluster中
        best_cluster.add_to_cluster(item) #将输入项添加到最适合的聚类中
        self.item_cluster[textid] = best_cluster #将该项对应的聚类更新为最适合的聚类
        
        #检查输入项是否移动到了新的聚类中，如果不是，则返回False；如果是，则返回True
        if best_cluster == previous_cluster:
            return False
        else:
            return True
 


    #返回输入项所在的聚类。如果项存在于某个聚类中，则返回该聚类；否则返回None
    def get_items_cluster(self, item):  
        textid = item[0] #每个项的第一个元素代表该项/文本语句的id
        
        if textid in self.item_cluster: #如果该项的id在item_cluster字典中
            return self.item_cluster[textid] #返回该项所在的聚类
        else: #如果该项的id不在item_cluster字典中
            return None #返回空值
        

    #返回所有聚类的中心；最后返回一个列表，列表每一个元素均为对应聚类中的中心，每个元素也是一个列表，列表第一个元素是中心值对应的项/语句的id，第二项为对应的项/语句，第四个元素为"cluster_centroid"，第五个元素为最佳匹配度值（一个余弦相似度值）
    def get_centroids(self):  
        centroids = [] #初始化一个空列表centroids，用于存储所有聚类的中心
        for cluster in self.clusters: #对于聚类列表中每一个聚类
            centroids.append(cluster.get_centroid()) #将当前聚类的中心添加到列表centroids中
        
        return centroids
    
    #返回所有聚类的异常值的函数；最后返回一个列表，列表的每一个元素均为对应聚类中的异常值，每个元素也是一个列表，第一个元素是异常值对应的项/语句的id，第二项为对应的项/语句，第四个元素为"cluster_outlier"，第五个元素为1-最大异常度值（一个余弦相似度值）    
    def get_outliers(self):  
        outliers = [] #初始化一个空列表outliers，用于存储所有聚类的异常值
        for cluster in self.clusters: #对于聚类列表中每一个聚类
            outliers.append(cluster.get_outlier()) #将当前聚类的异常值添加到列表outliers中
        
        return outliers
 
    #返回每个聚类中的随机项 ？？？
    def get_randoms(self, number_per_cluster=1, verbose=False):  #接受两个默认参数，
        randoms = [] #初始化一个空列表randoms，用于存储
        for cluster in self.clusters: #对于聚类列表中每一个聚类
            randoms += cluster.get_random_members(number_per_cluster, verbose) #从当前聚类中获取number_per_cluster个随机项，并将其添加到列表randoms中
        
        return randoms
   
    #返回一个字符串，其中每一个元素为对应索引的聚类包含项的数目
    def shape(self):  
        lengths = [] #初始化一个空列表
        for cluster in self.clusters:  #对于聚类列表中每一个聚类
            lengths.append(cluster.size()) #将每个聚类中的项的数量添加到列表lengths中
        
        return str(lengths) #返回一个字符串，其中每一个元素为对应索引的聚类包含项的数目




#一个用于无监督或轻度监督聚类的聚类
class Cluster():
    feature_idx = {} #一个空字典，用于存储每个特征的索引，其中键是特征（这里是单词），值是特征的索引；例如{'Hello,': 0, 'World!': 1, 'Python!': 2}；


    def __init__(self):
        self.members = {} # 空字典，此聚类中每个项的id，其中键是文本ID，值是每个项/文本；例如{1: [1, 'Hello, World! Hello, Python!']}
        self.feature_vector = [] # 空列表，此聚类的特征向量；例如[2, 1, 1]，表示在文本中，'Hello,'出现了两次，'World!'和'Python!'各出现了一次，其中出现的位置对应上面的单词/特征的索引
    

    #将新的项添加到聚类中的函数，接受的item是一个项，第一个元素代表文本id，第二个元素代表文本
    #假如调用add_to_cluster([1, 'Hello, World! Hello, Python!'])，则1为文本id，'Hello, World! Hello, Python!'为文本；之后feature_idx将变为{'Hello,': 0, 'World!': 1, 'Python!': 2}；
    #members将变为{1: [1, 'Hello, World! Hello, Python!']}；feature_vector将变为[2, 1, 1]，因为文本中，'Hello,'出现了两次，'World!'和'Python!'各出现了一次

    def add_to_cluster(self, item): #接受一个参数item，是一个项，第一个元素代表文本id，第二个元素代表文本
        textid = item[0] #将item的第一个元素赋值给textid
        text = item[1] #将item的第二个元素赋值给text，也就是文本
        
        self.members[textid] = item #这行代码将项目添加到聚类的成员字典中，键是文本ID，值是每个项
                
        #split方法在 Python 中用于将字符串分割成子字符串列表。如果不传递任何参数（如在这个例子中），split() 默认在所有的空白字符处进行分割，包括空格、换行符 \n、制表符 \t 等。
        #例如，如果 text 是 "Hello, World!"，那么 text.split() 将返回 ['Hello,', 'World!']
        words = text.split() #将text按空格分割，得到一个列表，赋值给words。相当于把文本分割为单词

        for word in words: #对于words中的每一个单词
            if word in self.feature_idx: #检查当前单词是否在特征索引feature_idx中，如果在
                while len(self.feature_vector) <= self.feature_idx[word]: #当特征向量的长度小于当前单词的索引时
                    self.feature_vector.append(0) #向特征向量中添加0，直到特征向量的长度等于当前单词的索引，代表多了几个新特征/单词
                    
                self.feature_vector[self.feature_idx[word]] += 1 #将特征向量中当前单词的索引位置的值加1，为例记录当前单词在文本中出现的次数

            else: #如果当前单词不在特征索引feature_idx中，即这是一个新特征，还不属于任何聚类
                self.feature_idx[word] = len(self.feature_vector) #将当前单词word添加到特征索引feature_idx中，键是单词/特征，值为特征的索引，设置为当前特征向量的长度（因为是新元素，因此索引就是当前特征向量的长度）
                self.feature_vector.append(1) #将特征向量中添加1，表示当前单词在文本中出现的次数为1
                
        
    #从聚类中移除一个项，包括移除存储聚类每个项id的members，和修改存储此聚类特征向量的feature_vector中的特征的数量

    def remove_from_cluster(self, item): #接受一个参数item，是一个项，第一个元素代表文本id，第二个元素代表文本
        textid = item[0] #将item的第一个元素赋值给textid
        text = item[1] #将item的第二个元素赋值给text，也就是文本
        
        exists = self.members.pop(textid, False) #从字典members中移除键为textid的项。如果该项存在，pop方法会返回该项并将其从字典中移除，然后赋值给exists。如果该项不存在，pop方法会返回默认值False
        
        if exists: #若exists为True，也就是说该项/文本在members中存在
            words = text.split() #将text按空格分割，得到一个列表，赋值给words。相当于把文本分割为单词
            for word in words: #对于words中的每一个单词
                index = self.feature_idx[word] #获取当前单词（作为键）在特征索引feature_idx中的索引（对应的值）
                if index < len(self.feature_vector): #如果索引小于特征向量的长度（特征向量的长度代表有几个特征/单词）
                    self.feature_vector[index] -= 1 #将特征向量中当前单词的索引位置的值减1，为例记录当前单词在文本中出现的次数
        
        
    #计算给定项与此聚类的余弦相似度的函数，接受一个参数item，是一个项，第一个元素代表文本id，第二个元素代表文本。最后比较的是聚类的特征向量（feature_vector）和该项的特征向量（vector）之间的余弦相似度。此外还要更新特征索引feature_idx和特征向量feature_vector
    def cosine_similary(self, item): #接受一个参数item，是一个项，第一个元素代表文本id，第二个元素代表文本
        text = item[1] #将item的第二个元素赋值给text，也就是文本
        words = text.split() #将text按空格分割，得到一个列表，赋值给words。相当于把文本分割为单词
        
        vector = [0] * len(self.feature_vector) #创建一个新的列表vector，其长度与此聚类的特征向量feature_vector的长度相同，且每个元素都为0，代表此项（item）的特征向量
        #更新该项的特征向量与聚类的特征向量、特征索引
        for word in words: #对于words中的每一个单词
            if word not in self.feature_idx: #如果当前单词不在特征索引feature_idx中，即这是一个新特征，还不属于聚类
                self.feature_idx[word] = len(self.feature_vector) #将当前单词word添加到特征索引feature_idx中，键是单词/特征，值为特征的索引，设置为当前特征向量的长度（因为是新元素，因此索引就是当前特征向量的长度）
                self.feature_vector.append(0) #向特征向量中添加0，代表多了新特征/单词，但聚类中没有，所以数量为0
                vector.append(1) #向列表vector中添加1，代表当前单词在文本中出现的次数为1（因为是新单词）
            else: #如果当前单词在特征索引feature_idx中
                while len(vector) <= self.feature_idx[word]: #当列表vector的长度小于当前单词的索引时
                    vector.append(0) #向列表vector中添加0
                    self.feature_vector.append(0) #向特征向量中添加0，代表多了新特征/单词
                              
                vector[self.feature_idx[word]] += 1 #将列表vector中对应当前单词的索引位置（索引位置要与聚类的特征向量保持一致）的值加1，为例记录当前单词在文本中出现的次数
        
        item_tensor = torch.FloatTensor(vector) #将列表vector转换为浮点张量，赋值给item_tensor
        cluster_tensor = torch.FloatTensor(self.feature_vector) #将特征向量feature_vector转换为浮点张量，赋值给cluster_tensor
        
        #得到该项与聚类的特征向量，这里聚类的特征向量不会添加该项的特征，但如果有新单词/特征会补0


        #该函数用于计算两个张量之间的余弦相似度。余弦相似度是一种在多维空间中比较两个向量方向的度量，它的值范围是-1到1。两个向量的方向完全相同时值为1；两个向量的方向完全相反时为-1；两个向量是正交的（即角度为90度）为0
        similarity = F.cosine_similarity(item_tensor, cluster_tensor, 0) #（按行）计算两个张量的余弦相似度，赋值给similarity（按行是计算两个张量每一行之间的余弦相似度，返回一个一维张量，其中每个元素都是对应行的余弦相似度）
        
        # 也可以使用函数`F.pairwise_distance()`，但需要先归一化聚类
        
        return similarity.item() # 返回余弦相似度的值，item()方法将张量转换为浮点数
    


    #计算字典members的长度，也就是文本语句的数量
    def size(self):
        return len(self.members.keys()) #keys函数返回一个包含所有键的视图对象（不是列表或集合，但可以用在需要迭代键的地方）
        #其实相当于len(self.members)
 
    #计算聚类的中心，也就是聚类中的最佳项，即与聚类的余弦相似度最高的项。最后返回的是一个列表，第一个元素是中心值对应的项/语句的id，第二项为对应的项/语句，第四个元素为"cluster_centroid"，第五个元素为最佳匹配度值（一个余弦相似度值）
    def get_centroid(self):
        #首先判断members是否为空，即聚类是否存在文本语句，不存在直接返回空列表
        if len(self.members) == 0:
            return []
        
        best_item = None #初始化最佳项为空
        best_fit = float("-inf") #初始化最佳匹配度为负无穷
        
        for textid in self.members.keys(): #对于聚类的每一个文本语句的id
            item = self.members[textid] #获取当前项对应的文本语句的内容
            similarity = self.cosine_similary(item) #计算当前项与聚类（包含该项）的余弦相似度
            if similarity > best_fit: #如果当前项与聚类的余弦相似度大于最佳匹配度
                best_fit = similarity #更新最佳匹配度为当前项与聚类的余弦相似度
                best_item = item #更新最佳项为当前项
                
        best_item[3] = "cluster_centroid" #将最佳项的第四个元素赋值为"cluster_centroid"
        best_item[4] = best_fit  #将最佳匹配度的值（一个余弦相似度值）赋值给最佳项的第五个元素
                
        return best_item
     
         
    #计算聚类的异常值，也就是与聚类的余弦相似度最低的项。最后返回的是一个列表，第一个元素是异常值对应的项/语句的id，第二项为对应的项/语句，第四个元素为"cluster_outlier"，第五个元素为1-最大异常度值（一个余弦相似度值）
    def get_outlier(self):
        #首先判断members是否为空，即聚类是否存在文本语句，不存在直接返回空列表
        if len(self.members) == 0:
            return []
        
        best_item = None #初始化最佳项为空
        biggest_outlier = float("inf") #初始化最大异常度为负无穷
        
        for textid in self.members.keys(): #对于聚类的每一个文本语句的id
            item = self.members[textid] #获取当前项对应的文本语句的内容
            similarity = self.cosine_similary(item) #计算当前项与聚类（包含该项）的余弦相似度
            if similarity < biggest_outlier: #如果当前项与聚类的余弦相似度小于最佳匹配度
                biggest_outlier = similarity #更新最大异常度为当前项与聚类的余弦相似度
                best_item = item #更新最佳项为当前项

        best_item[3] = "cluster_outlier" #将最佳项的第四个元素赋值为"cluster_outlier"
        best_item[4] = 1 - biggest_outlier #将最大异常度的值（一个余弦相似度值）赋值给最佳项的第五个元素
                
        return best_item


    #返回聚类中number个随机项，接受两个默认参数，number为随机项的数量，verbose为是否打印随机项的文本语句。最后返回的是一个列表，列表的每一个元素都是一个项，项的第一个元素为文本id，第二个元素为文本，第四个元素为"cluster_member"，第五个元素为该项与聚类的余弦相似度
    def get_random_members(self, number=1, verbose=False): #接受两个默认参数
        #首先判断members是否为空，即聚类是否存在文本语句，不存在直接返回空列表
        if len(self.members) == 0: 
            return []        
        
        keys = list(self.members.keys()) #keys函数返回一个包含所有键的视图对象（不是列表或集合，但可以用在需要迭代键的地方），并将它们转换为列表，赋值给keys。相当于keys是一个包含所有键/文本语句id的列表
        shuffle(keys) #随机打乱文本语句id的顺序

        randoms = [] #初始化一个空列表randoms
        for i in range(0, number): #循环number次（默认参数，为1）
            if i < len(keys): #如果当前循环次数小于文本语句id的数量
                textid = keys[i] #获取当前循环次数对应的文本语句id（已经打乱）
                item = self.members[textid] #获取该id对应的内容（列表，包括id和文本）
                item[3] = "cluster_member" #将该项的第四个元素赋值为"cluster_member"
                item[4] = self.cosine_similary(item) #将该项的第五个元素赋值为该项与聚类的余弦相似度

                randoms.append(item) #将该项添加到列表randoms中
         
        if verbose: #如果verbose为True
            print("\nRandomly items selected from cluster:") #打印提示信息：从集群中随机选择的项
            for item in randoms: #对于randoms中的每一个项
                print("\t"+item[1]) #\t表示在输出前加一个制表符，即缩进一个tab，然后打印该项的第二个元素，也就是文本语句     
                
        return randoms #返回随机项的列表
    




         
