import sys
import os
sys.path.append('mnist_classes')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from PIL import Image
import random
import torch
import torchvision.transforms as T
from easydict import EasyDict

from dataloader import DisDataset, tensor2img, DisDataset2
from trainer import train_model
from model.efficientNet import DisEfficient, DisEfficient2, DisEfficient3
from model.regnet import Regnet

BAGGING_NUM = 1
BATCH_SIZE = 128
flod_num = 1
save_path = os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

root = r"D:\노지 작물 질병 진단 이미지"
train_dir = os.path.join(root, "train")

df0 = pd.read_csv("질병.csv")
df1 = pd.read_csv("정상.csv")
df2 = pd.read_csv("증강.csv")
meta_df = pd.read_csv("meta_clean.csv") 

# 모델을 학습하고, 최종 모델을 기반으로 테스트 데이터에 대한 예측 결과물을 저장하는 도구 함수이다
args = EasyDict(
    {'exp_num':'0',
     
     # Path settings
     'root':'train_dir',
     'save_dict' : '08_aware_radish',
     'df0':meta_df,
     'df1':df1,
     'df2':df2,

     # Model parameter settings
     'CODER':'regnety_040', # 'regnety_040', 'eff'
     'drop_path_rate':0.2,
     'model_class': Regnet,
     
     # Training parameter settings
     ## Base Parameter
     'img_size':512,
     'BATCH_SIZE':BATCH_SIZE,
     'epochs':50,
     'optimizer':'Lamb',
     'lr':5e-5,
     'weight_decay':1e-3,
     'Dataset' : DisDataset2,
     'aware':False,
     'train_var':True,
     'aug':0.8,
     'fold_num':flod_num,
     'bagging_num':BAGGING_NUM,

     ## sampling mode
      'mode' : 'sub',
      'index' : [3,4,5],
     ## Augmentation
     'aug_ver':2,

     'scheduler':'cos',
     ## Scheduler (OnecycleLR)
     'warm_epoch':5,
     'max_lr':1e-3,

     ### Cosine Annealing
     'min_lr':5e-6,
     'tmax':145,

     ## etc.
     'patience':15,
     'clipping':None,

     # Hardware settings
     'amp':True,
     'multi_gpu':True,
     'logging':False,
     'num_workers':4,
     'seed':42,
     'device':device,

    })

# def seed_everything(seed):
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

if __name__ == '__main__': 
    #seed_everything(np.random.randint(1, 5000))
    print(args.CODER + " train..")
    train_model(args)
