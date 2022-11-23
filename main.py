import sys
import os
sys.path.append('mnist_classes')

import pandas as pd
import numpy as np
import torch
from easydict import EasyDict

from dataloader import DisDataset, tensor2img, DisDataset2, radiusDataset, NonDataset
from trainer import train_model
from model.efficientNet import DisEfficient, DisEfficient2, DisEfficient3
from model.regnet import Regnet
from model.metaf import PoolFormer, PoolFormer_radius, PoolFormer_non
from train import Trainer, Mixup_trainer

save_path = os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# 전체
root = r"D:\노지 작물 질병 진단 이미지"
train_dir = os.path.join(root, "train")
meta_df = pd.read_csv("meta_clean.csv") 
weights = np.load('weight_c.npy') # 30 class radio
#np.load('weights.npy')

# 무
radius_df = pd.read_csv("무_train.csv")
radius_weights = np.array([0.32568149, 0.67431851])
sub = radius_df[radius_df['disease'].isin([0, 1])].reset_index(drop=True)

# 알수없음 모델
weights_non = np.array([0.20476165, 0.2712193 , 0.20382391, 0.18739695, 0.13279819])
non_df = pd.read_csv("non_train.csv")

args = EasyDict(
    {
     # Path settings
     'root':'train_dir',
     'save_dict' : 'we_fix_mixup_in05',
     'df':meta_df,
     # Model parameter settings
     'CODER':'poolformer_m36', # 'regnety_040', 'efficientnet-b0' ,poolformer_m36
     'drop_path_rate':0.2,
     'model_class': PoolFormer,
     'weight':weights,
     'pretrained':False,
     
     # Training parameter settings
     ## Base Parameter
     'img_size':224,
     'test_size':224,
     'BATCH_SIZE':128,
     'epochs':200,
     'optimizer':'Lamb',
     'lr':3e-5,
     'weight_decay':1e-3,
     'Dataset' : DisDataset2,
     'aware':False,
     'fold_num':1,
     'bagging_num':4,

     # sampling Parameter
     'aug':0.8,
     'resampling':False,
     'resample_inter' : 5,

     ## Augmentation
     'incrop' : True,
     'test_ac': True,
     'inrecrop':0.5,
     'pad':True,
     
     ## sampling mode
      'mode' : 'full', #sub, under
      'index' : [3,4,5],
     

     #scheduler 
     'scheduler':'cos',
     ## Scheduler (OnecycleLR)
     'warm_epoch':5,
     'max_lr':1e-3,

     ### Cosine Annealing
     'min_lr':5e-6,
     'tmax':145,

     ## etc.
     'patience':3,
     'clipping':None,

     # Hardware settings
     'amp':True,
     'multi_gpu':True,
     'logging':False,
     'num_workers':4,
     'seed':42,
     'device':device,

    })

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__': 
    seed_everything(np.random.randint(1, 5000))
    print(args.CODER + " train..")
    trainer = Trainer(args)
    # trainer = Mixup_trainer(args)
    trainer.fit()
