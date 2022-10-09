from importlib.metadata import files
import re
import os
import random
from typing import Tuple, Sequence, Callable
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import json

from endecoder import *
import torchvision.transforms.functional as F

class InResize(transforms.Resize):
    """
    bbox를 포함하면서 random crop 후 resize
    """
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)
        self.size=size

    @staticmethod
    def target_size(w, h, x1,y1,x2,y2):
        # print(w, h, x1, y1, x2, y2)
        rnd_y2 = np.random.randint(0, h - y2) if h - y2 > 0 else 0
        rnd_x2 = np.random.randint(0, w - x2) if w - x2 > 0 else 0
        rnd_y1 = np.random.randint(0, y1) if y1 > 0 else 0
        rnd_x1 = np.random.randint(0, x1) if x1 > 0 else 0
        
        size = (x1-rnd_x1, y1-rnd_y1, x2+rnd_x2, y2+rnd_y2)
        return size

    def __call__(self, img, x1,y1,x2,y2):
        w, h = img.size
        target_size = self.target_size(w, h, x1,y1,x2,y2)
        img = img.crop(target_size)
        return F.resize(img, self.size, self.interpolation)

class DisDataset(torch.utils.data.Dataset):
    """
    dir_path : 데이터폴더 경로
    meta_df : 가져올 데이터 csv
    mode : 불러 올 데이터(train or test)
    mix_up : mix_up augmentation 유무
    reverse : reverse augmentation 유무
    sub_data : emnist 로 만든 sub data 사용 유무
    """
    def __init__(self,
                 files,
                 root,
                 mode='train',
                 aware=False,
                 img_size = 224,
                 ):
        
        self.files = files
        self.root = root 
        self.mode = mode
        self.aware = aware
        self.train_mode = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                            ])

        self.test_mode = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])
    
        self.Resize =  transforms.Resize((img_size, img_size))
        self.InResize = InResize((img_size, img_size))
        # self.disease_decoder = disease_decoder
        self.crop_decoder_kr = crop_decoder_kr
        self.crop_decoder_en = crop_decoder_en
        self.disease_aware_decoder = disease_aware_decoder
        self.crop_aware_decoder_kr = crop_aware_decoder_kr

        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        path = self.files[index]
        data = json.load(open(path, "r"))
        
        crop_label = torch.zeros((11))
        crop_label[data['annotations']['crop']] += 1

        if self.aware:
            disease_label = torch.zeros((31))
            dis_idx = self.disease_aware_decoder[str(data['annotations']['disease']).zfill(2)]
            crop_idx = self.crop_aware_decoder_kr[str(data['annotations']['crop']).zfill(2)]
            if not dis_idx:
                disease_label[crop_idx] += 1
            else:
                disease_label[dis_idx] += 1

        else:
            # dis_c = path.split("\\")[-2].split("_")[-1]
            disease_label = torch.zeros((21))
            disease_label[data['annotations']['disease']] += 1
            # crop_code = self.crop_decoder_en[str(data['annotations']['crop']).zfill(2)]
            # crop_code = crop_code + '_' + dis_c if disease_label[0] == 1 else crop_code + '_' + dis_c
        
        crop_dir = os.path.join(self.root + '/images', path.split("\\")[-2])
        image_path = os.path.join(crop_dir, data['description']['image'])
        image = Image.open(image_path).convert('RGB')
        
        sample = {'image': image, 'crop_label': crop_label, 'disease_label':disease_label}

        # train mode transform
        if self.mode == 'train':
            if 0.3 >= np.random.rand():
                x1,y1,x2,y2 = data['annotations']['points'][0].values()
                sample['image'] = self.InResize(sample['image'], x1,y1,x2,y2)
            else:
                sample['image'] = self.Resize(sample['image'])
            sample['image'] = self.train_mode(sample['image'])

        # test mode transform
        elif self.mode == 'test' or self.mode == 'valid':
            sample['image'] = self.test_mode(sample['image'])

        # sample['label'] = torch.FloatTensor(sample['label'])
        return sample
    
    def mixup(self, image, label):
        """
        랜덤하게 4개의 이미지를 붙인다.
        """
        idxs = np.random.randint(1, len(self.meta_df), 3)
        h, w = image.size
        
        images = [Image.open(self.dir_path + str(self.meta_df.iloc[index,0]).zfill(5) + '.png').convert('RGB') for index in idxs]
        images.append(image)
        labels = [self.meta_df.iloc[index, 1:].values.astype('float') for index in idxs]
        labels.append(label)
        
        expand_img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        expand_img[:h, :w, :] = np.array(images[0])
        expand_img[:h, w:int(w*2), :] = np.array(images[1])
        expand_img[h:int(h*2), 0:w, :] = np.array(images[2])
        expand_img[h:int(h*2), w:int(w*2), :] = np.array(images[3])
        
        del images
        return Image.fromarray(expand_img).resize((h,w)), np.clip(np.sum(np.array(labels), axis=0), 0, 1)
    
    def diagonal_reverse(self, img):
        """
        하나의 이미지를 4등분하여 재배열 한다.
        """
        transformed_img = img.clone()
        center = img.shape[1] // 2
        transformed_img[:, 0:center, 0:center] = img[:, center:center + center, center:center + center]
        transformed_img[:, 0:center, center:center + center]  = img[:, center:center*2, 0:center]
        transformed_img[:, center:center + center, 0:center] = img[:, 0:center, center:center*2]
        transformed_img[:, center:center + center, center:center + center] = img[:, 0:center, 0:center]
        
        return transformed_img

class DisDataset2(torch.utils.data.Dataset):
    """
    dir_path : 데이터폴더 경로
    meta_df : 가져올 데이터 csv
    mode : 불러 올 데이터(train or test)
    mix_up : mix_up augmentation 유무
    reverse : reverse augmentation 유무
    sub_data : emnist 로 만든 sub data 사용 유무
    """
    def __init__(self,
                 files,
                 mode='train',
                 img_size = 224,
                 ):
        
        self.files = files
        self.mode = mode
        self.train_mode = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                            ])
        """
        transforms.RandomAffine((20)),
        transforms.RandomRotation(90),
        
        """
        self.test_mode = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])
    
        self.Resize =  transforms.Resize((img_size, img_size))
        self.InResize = InResize((img_size, img_size))
        # self.disease_decoder = disease_decoder
        self.crop_decoder_kr = crop_decoder_kr
        self.crop_decoder_en = crop_decoder_en
        self.disease_aware_decoder = disease_aware_decoder
        self.crop_aware_decoder_kr = crop_aware_decoder_kr

        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = self.files.iloc[index, ]
        
        crop_label = torch.zeros((11))
        crop_label[data['crop']] += 1
        disease_label = torch.zeros((31))
        disease_label[data['disease']] += 1

        image_path = data['path']
        image = Image.open(image_path).convert('RGB')
        # print(torch.argmax(disease_label))
        sample = {'image': image, 'crop_label': crop_label, 'disease_label':torch.argmax(disease_label)}

        # train mode transform
        if self.mode == 'train':
            if 0.3 >= np.random.rand():
                x1,y1,x2,y2 = re.findall(r'\d+', data['bbox'])
                sample['image'] = self.InResize(sample['image'], int(x1),int(y1),int(x2),int(y2))
            else:
                sample['image'] = self.Resize(sample['image'])
            sample['image'] = self.train_mode(sample['image'])

        # test mode transform
        elif self.mode == 'test' or self.mode == 'valid':
            sample['image'] = self.test_mode(sample['image'])

        # sample['disease_label'] = torch.Tensor(sample['disease_label'])
        return sample

class OODDataset(torch.utils.data.Dataset):
    """
    dir_path : 데이터폴더 경로
    meta_df : 가져올 데이터 csv
    mode : 불러 올 데이터(train or test)
    mix_up : mix_up augmentation 유무
    reverse : reverse augmentation 유무
    sub_data : emnist 로 만든 sub data 사용 유무
    """
    def __init__(self,
                 files,
                 mode='train',
                 img_size = 224,
                 ):
        
        self.files = files
        self.mode = mode
        self.train_mode = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(30),
                    # transforms.ColorJitter(0.3, 0.3, 0.3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                            ])

        self.test_mode = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])

        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_path, label_num = self.files[index]
        
        label = torch.zeros((2))
        label[int(label_num)] += 1

        image = Image.open(img_path).convert('RGB')
        
        sample = {'image': image, 'label': label}

        # train mode transform
        if self.mode == 'train':
            sample['image'] = self.train_mode(sample['image'])

        # test mode transform
        elif self.mode == 'test' or self.mode == 'valid':
            sample['image'] = self.test_mode(sample['image'])

        # sample['label'] = torch.FloatTensor(sample['label'])
        return sample


class UnNormalize(object):
    """
    정규화 inverse 한다.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


import matplotlib.pyplot as plt

def tensor2img(img):
    """
    tensor 형태에서 0~255 진행 후 numpy 배열로 바꿔서 반환
    """
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    a = unorm(img).numpy()
    a = a.transpose(1, 2, 0)
    return a