#!/usr/bin/env python
# coding: utf-8

# # Configuration

# In[1]:


batch_size = 32
num_workers = 4
n_batch = 10
DATA_PATH = '../input/shopee-product-matching/'

class CFG:
    image_size = 640
    s = 30
    margin = 0.5
    model_arch = "eca_nfnet_l1" # densenet121, tf_efficientnet_b0_ns, efficientnet_b3, tf_efficientnet_b4, eca_nfnet_l0, eca_nfnet_l1,
    model_path = "../input/shopeemodel/fold0_2814_eca_nfnet_l1_640_epoch2_4_cv83304.pth"
    
    img_thres = 0.95
    addition = 0.2
    txt_thres = 0.75
    multi_gpu = True
    use_fc = False
    fc_dim = 512


# In[2]:


import pandas as pd
import numpy as np
import sys
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')
import os
import sys
import time
import cv2
import PIL.Image
import random
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import roc_auc_score
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pylab import rcParams
import timm
from warnings import filterwarnings
from sklearn.preprocessing import LabelEncoder
import math
import glob
from torch.nn import Parameter
from cuml.neighbors import NearestNeighbors
from collections import OrderedDict
filterwarnings("ignore")
device = torch.device('cuda')


# In[3]:


test = pd.read_csv(DATA_PATH + 'test.csv') # 导入test.csv为dataframe
test['file_path'] = DATA_PATH + 'test_images/' + test['image']  # 加入图片的文件路径


# In[4]:


# 设置随机种子，以便实验复现
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set True to be faster
seed_everything(42)


# # 1.CNN（使用商品图片做匹配）

# In[6]:


# 对 test set 做数据增强
transforms_valid = albumentations.Compose([
    albumentations.Resize(CFG.image_size, CFG.image_size),
    albumentations.Normalize()
])

transforms_valid1 = albumentations.Compose([
    albumentations.Resize(384, 384),
    albumentations.Normalize()
])


# In[8]:


# 定义数据集结构
class SHOPEEDataset(Dataset):
    def __init__(self, df, mode, transform=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode # 数据集模式（train模式或test模式）
        self.transform = transform # 数据增强
        
    def __len__(self):
        return len(self.df) # 获取dataframe行数
    
    def __getitem__(self, index):
        row = self.df.loc[index] # 获取指定（index）行
        img = cv2.imread(row.file_path) # 用cv2读入图片数据
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2默认读入是BGR格式，现在转换成RGB格式
        
        # 如果有数据增强，则做数据增强
        if self.transform is not None: 
            res = self.transform(image=img) 
            img = res['image']
        
        # 调整一下数据格式
        img = img.astype(np.float32)
        img = img.transpose(2,0,1)
        
        # 返回 获取到的图片
        if self.mode == 'test':
            return torch.tensor(img).float() # 如果test模式，则只返回图片
        else:
            return torch.tensor(img).float(), torch.tensor(row.label_group).float() # 如果其他模式，则返回 图片 和 其所属类别标签


# In[9]:


# 载入ArcFace函数，下面代码不加注释，想搞明白的同学可以微信提问。
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# In[10]:


# 主模型网络
class SHOPEENet(nn.Module):
    def __init__(self, out_feature, backbone='densenet121', use_fc=True, pretrained=True):
        super(SHOPEENet, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained) # 创建骨干网络
        self.out_feature = out_feature # 模型输出的类别数(本次比赛数据是11014)
        self.pooling = nn.AdaptiveAvgPool2d(1) # 自定义一个pooling层

        # 替换骨干网络的最后一部分
        if "efficientnet" in CFG.model_arch:
            self.in_features = self.backbone.classifier.in_features
            self.backbone.global_pool = nn.Identity()
            self.backbone.classifier = nn.Identity()
        elif "nfnet" in CFG.model_arch:
            self.in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()
        elif "swin" in CFG.model_arch:
            self.in_features = self.backbone.head.in_features
            # self.backbone.avgpool = nn.Identity()
            self.backbone.head = nn.Identity()

        print(self.in_features)
        
        # 定义全连接层（本方案不使用）
        self.use_fc = use_fc
        if self.use_fc:
            self.dropout = nn.Dropout(p=0.0)
            self.fc = nn.Linear(self.in_features, CFG.fc_dim)
            self.bn = nn.BatchNorm1d(CFG.fc_dim)
            self._init_params()
            self.in_features = CFG.fc_dim
            
        # 定义arcface层
        self.final = ArcMarginProduct(self.in_features, self.out_feature, s=CFG.s, m=CFG.margin, easy_margin=False, ls_eps=0.0)

    
    def _init_params(self): # 全连接层的初始化（本方案不使用）
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0) 
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, labels=None):
        # 前向计算微调后的模型
        if "efficientnet" in CFG.model_arch or "nfnet" in CFG.model_arch:
            batch_size = x.shape[0]
            features = self.backbone(x)
            features = self.pooling(features).view(batch_size, -1)
            features = F.normalize(features)

        elif "swin" in CFG.model_arch:
            batch_size = x.shape[0]
            features = self.backbone(x)
            # features = self.pooling(features).view(batch_size, -1)
            features = F.normalize(features)

        # 使用全连接层（本方案不使用）
        if self.use_fc:
            features = self.dropout(features)
            features = self.fc(features)
            features = self.bn(features)

        # 如果有labels（训练阶段）则使用arface，如果没有label(验证或测试阶段)则直接返回图片的features
        if labels is not None:
            return self.final(features, labels)
        return features


# In[11]:


# 验证/测试中获取图片经模型处理后的features
def generate_test_features(test_loader):
    model.eval() # 模型调整到评估模式
    FEAS = [] 
    TARGETS = []
    with torch.no_grad():
        for batch_idx, (images) in enumerate(test_loader): # 从数据管道中导入 图片
            images = images.to(device) # 将数据放入GPU
            features = model(images) # 把数据放入模型获得features
            FEAS += [features.detach().cpu()] # 存下当前features
    FEAS = torch.cat(FEAS).cpu().numpy() 
    return FEAS # 返回所有数据的features





# ## model1

# In[ ]:


# 创建model1
model = SHOPEENet(11014, backbone=CFG.model_arch, use_fc=CFG.use_fc, pretrained=False)
# 载入训练好的model1
state = torch.load(CFG.model_path, map_location='cuda:0') 
if CFG.multi_gpu:
    new_state_dict = OrderedDict()
    for k, v in state.items():
        k=k[7:]
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(state)
model.to(device);
model.eval()
# 创建测试数据集1
dataset_test = SHOPEEDataset(test, 'test', transform=transforms_valid)
# 创建测试数据管道1
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

FEAS0 = generate_test_features(model, test_loader) # 获得model1的features
del model, state, dataset_test, test_loader  # 删掉以上模型和数据集（留出更多内存空间）
gc.collect() # 清理内存
torch.cuda.empty_cache() # 清理显存


# ## model2

# In[ ]:


# 创建model2
CFG.model_arch = "swin_large_patch4_window12_384"
model1 = SHOPEENet(11014, backbone="swin_large_patch4_window12_384", use_fc=CFG.use_fc, pretrained=False)

# 载入训练好的model2
state1 = torch.load("../input/shopeemodel/fold0_0718_swin_large_patch4_window12_384_384_epoch12.pth", map_location='cuda:0')
if CFG.multi_gpu:
    new_state_dict = OrderedDict()
    for k, v in state1.items():
        k=k[7:]
        new_state_dict[k]=v
    model1.load_state_dict(new_state_dict)
else:
    model1.load_state_dict(state1)
model1.to(device);
model1.eval()

# 创建测试数据集2
dataset_test1 = SHOPEEDataset(test, 'test', transform=transforms_valid1)
# 创建测试数据管道2
test_loader1 = torch.utils.data.DataLoader(dataset_test1, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

FEAS1 = generate_test_features(model1, test_loader1) # 获得model2的features
del model1, state1, dataset_test1, test_loader1 # 删掉以上模型和数据集（留出更多内存空间）
gc.collect() # 清理内存
torch.cuda.empty_cache() # 清理显存


# In[ ]:


# 合并model1和model2的features
FEAS = np.concatenate([FEAS0,FEAS1],axis = 1) 
print(FEAS0.shape,FEAS1.shape,FEAS.shape)


# In[13]:


# 使用knn方法，寻找每张图片最近的其他图片，以匹配。
def get_image_neighbors(df, embeddings, KNN=50):
    model = NearestNeighbors(n_neighbors = KNN) # 创建knn模型
    model.fit(embeddings) # 训练features
    distances, indices = model.kneighbors(embeddings) # 获得图片之间的距离（相似度）
    
    predictions = []
    for k in tqdm(range(embeddings.shape[0])): # 每张图片都拿出来两两比对
        idx = np.where(distances[k,] < CFG.img_thres)[0] # 设置一个thres（阈值），来确定匹配的严格程度
        # 对于没有匹配到的其他图片的图片，我们放宽阈值再匹配一次
        if len(idx) == 1:
            idx = np.where(distances[k,] < (CFG.img_thres + CFG.addition))[0] 
        ids = indices[k,idx]
        posting_ids = df['posting_id'].iloc[ids].values # 输出匹配的图片
        predictions.append(posting_ids)
        
    del model, distances, indices
    gc.collect()
    return predictions

# 获得CNN的匹配结果
image_predictions = get_image_neighbors(test, FEAS, KNN=50 if len(test)>3 else 3)
test["preds1"] = image_predictions


# # 2.TEXT（使用TFIDF对商品标题做匹配）

# In[16]:


# 导入tfidf相关的库
import cudf, cuml, cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
print('RAPIDS',cuml.__version__)


# In[19]:


test_gf = cudf.read_csv(DATA_PATH + 'test.csv') # 再次导入test.csv
model = TfidfVectorizer(stop_words='english', binary=True, max_features=25_000) #创建tfidf模型
text_embeddings = model.fit_transform(test_gf.title).toarray()#使用tfidf模型对test数据中的商品标题训练
print('text embeddings shape',text_embeddings.shape) # 获得text的features


# In[21]:


# 分块做匹配。因为数据量大，无法一次性做两两匹配（会超内存）。
preds2 = []
CHUNK = 1024 * 4 # 每个分块的大小

print('Finding similar titles...')
CTS = len(test)//CHUNK
if len(test)%CHUNK!=0: 
    CTS += 1
for j in range( CTS ):
    a = j*CHUNK
    b = (j+1)*CHUNK
    b = min(b,len(test))
    print('chunk',a,'to',b)
    
    # 矩阵相乘计算相似度（features * features.T）
    # COSINE SIMILARITY DISTANCE，余弦相似度的知识，cos = (a*b) / (|a|*|b|)
    cts = cupy.matmul(text_embeddings, text_embeddings[a:b].T).T
    
    for k in range(b-a): 
        IDX = cupy.where(cts[k,] > CFG.txt_thres)[0] # 根据阈值确定匹配商品
        o = test.iloc[cupy.asnumpy(IDX)].posting_id.values
        preds2.append(o)
        
del model, text_embeddings
_ = gc.collect()

test["preds2"] = preds2


# # 3.PHASH

# In[24]:


# 数据csv自带的phash做匹配，phash相同则认为商品匹配
tmp = test.groupby('image_phash').posting_id.agg('unique').to_dict() 
test['preds3'] = test.image_phash.map(tmp)


# # Combine

# In[25]:


# 定义联合匹配函数（CNN,TEXT,PHASH三种方式做交集匹配）
def combine_for_sub(row):
    x = np.concatenate([row.preds1, row.preds2, row.preds3])
    return ' '.join( np.unique(x) )

test['matches'] = test.apply(combine_for_sub,axis=1) # 使用上面的函数进行交集匹配

# 输出submission.csv
test[['posting_id','matches']].to_csv('submission.csv',index=False)
sub = pd.read_csv('submission.csv')

