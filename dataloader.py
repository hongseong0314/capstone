from importlib.metadata import files
import re
import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import json

from endecoder import *
import torchvision.transforms.functional as F

def expand2square(pil_img):
    background_color = (0,0,0)
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class InResize(transforms.Resize):
    """
    bbox를 포함하면서 random crop 후 resize
    """
    def __init__(self, size, pad, crop, **kwargs):
        super().__init__(size, **kwargs)
        self.size=size
        self.pad = pad
        self.crop = crop
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
        if self.crop:
            target_size = self.target_size(w, h, x1,y1,x2,y2)
            img = img.crop(target_size)
        else:
            img = img.crop((x1,y1,x2,y2))
        if self.pad:
            img = expand2square(img)
        return F.resize(img, self.size, self.interpolation)

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
                 img_size=224,
                 incrop=True,
                 inrecrop=0.3,
                 pad=True,
                 test_ac = True,
                 ):
        
        self.files = files
        self.mode = mode
        self.incrop = incrop
        self.inrecrop = inrecrop
        self.test_ac = test_ac
        self.pad = pad
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
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])
    
  
        self.Resize =  transforms.Resize((img_size, img_size))
        self.InResize = InResize((img_size, img_size), pad=pad, crop=incrop)
        self.crop_decoder_kr = crop_decoder_kr
        self.crop_decoder_en = crop_decoder_en
        self.disease_aware_decoder = disease_aware_decoder
        self.crop_aware_decoder_kr = crop_aware_decoder_kr
        self.test_activate_size = int(img_size * 1.25)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = self.files.iloc[index, ]
        
        crop_label = torch.zeros((11))
        crop_label[data['crop']] += 1
        disease_label = torch.zeros((30))
        disease_label[data['disease']] += 1

        image_path = data['path']
        image = Image.open(image_path).convert('RGB')
        sample = {'image': image, 'crop_label': crop_label, 'disease_label':torch.argmax(disease_label)}

        # train mode transform
        if self.mode == 'train':
            if self.inrecrop  >= np.random.rand():
                x1,y1,x2,y2 = re.findall(r'\d+', data['bbox'])
                sample['image'] = self.InResize(sample['image'], int(x1),int(y1),int(x2),int(y2))
            else:
                if self.pad:
                    sample['image'] = expand2square(sample['image'])
                sample['image'] = self.Resize(sample['image'])
            sample['image'] = self.train_mode(sample['image'])

        # test mode transform
        elif self.mode == 'test' or self.mode == 'valid':
            if self.test_ac:
                test_size = self.test_activate_size
                if self.pad:
                    sample['image'] = expand2square(sample['image'])
                self.Resize =  transforms.Resize((test_size, test_size))
                sample['image'] = self.Resize(sample['image'])
                sample['image'] = self.test_mode(sample['image'])
            else:
                if self.pad:
                    sample['image'] = expand2square(sample['image'])
                sample['image'] = self.Resize(sample['image'])
                sample['image'] = self.test_mode(sample['image'])
        return sample

class radiusDataset(torch.utils.data.Dataset):
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
                 img_size=224,
                 incrop=True,
                 inrecrop=0.3,
                 pad=True,
                 test_ac = True,
                 ):
        
        self.files = files
        self.mode = mode
        self.incrop = incrop
        self.inrecrop = inrecrop
        self.test_ac = test_ac
        self.pad = pad
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
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])
    
  
        self.Resize =  transforms.Resize((img_size, img_size))
        self.InResize = InResize((img_size, img_size), pad=pad, crop=incrop)
        self.crop_decoder_kr = crop_decoder_kr
        self.crop_decoder_en = crop_decoder_en
        self.disease_aware_decoder = disease_aware_decoder
        self.crop_aware_decoder_kr = crop_aware_decoder_kr
        self.test_activate_size = int(img_size * 1.25)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = self.files.iloc[index, ]
        
        crop_label = torch.zeros((11))
        crop_label[data['crop']] += 1
        disease_label = torch.zeros((2))
        disease_label[data['disease']] += 1

        image_path = data['path']
        image = Image.open(image_path).convert('RGB')
        sample = {'image': image, 'crop_label': crop_label, 'disease_label':torch.argmax(disease_label)}

        # train mode transform
        if self.mode == 'train':
            if self.inrecrop  >= np.random.rand():
                x1,y1,x2,y2 = re.findall(r'\d+', data['bbox'])
                sample['image'] = self.InResize(sample['image'], int(x1),int(y1),int(x2),int(y2))
            else:
                if self.pad:
                    sample['image'] = expand2square(sample['image'])
                sample['image'] = self.Resize(sample['image'])
            sample['image'] = self.train_mode(sample['image'])

        # test mode transform
        elif self.mode == 'test' or self.mode == 'valid':
            if self.test_ac:
                test_size = self.test_activate_size
                if self.pad:
                    sample['image'] = expand2square(sample['image'])
                self.Resize =  transforms.Resize((test_size, test_size))
                sample['image'] = self.Resize(sample['image'])
                sample['image'] = self.test_mode(sample['image'])
            else:
                if self.pad:
                    sample['image'] = expand2square(sample['image'])
                sample['image'] = self.Resize(sample['image'])
                sample['image'] = self.test_mode(sample['image'])
        return sample

class NonDataset(torch.utils.data.Dataset):
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
                 img_size=224,
                 incrop=True,
                 inrecrop=0.3,
                 pad=True,
                 test_ac = True,
                 ):
        
        self.files = files
        self.mode = mode
        self.incrop = incrop
        self.inrecrop = inrecrop
        self.test_ac = test_ac
        self.pad = pad
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
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                            ])
    
  
        self.Resize =  transforms.Resize((img_size, img_size))
        self.InResize = InResize((img_size, img_size), pad=pad, crop=incrop)
        self.crop_decoder_kr = crop_decoder_kr
        self.crop_decoder_en = crop_decoder_en
        self.disease_aware_decoder = disease_aware_decoder
        self.crop_aware_decoder_kr = crop_aware_decoder_kr
        self.test_activate_size = int(img_size * 1.25)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = self.files.iloc[index, ]
        
        disease_label = torch.zeros((5))
        disease_label[data['crop']] += 1

        image_path = data['path']
        image = Image.open(image_path).convert('RGB')
        sample = {'image': image, 'crop_label': torch.tensor([1]), 'disease_label':torch.argmax(disease_label)}

        # train mode transform
        if self.mode == 'train':
            if self.inrecrop  >= np.random.rand():
                x1,y1,x2,y2 = re.findall(r'\d+', data['bbox'])
                sample['image'] = self.InResize(sample['image'], int(x1),int(y1),int(x2),int(y2))
            else:
                if self.pad:
                    sample['image'] = expand2square(sample['image'])
                sample['image'] = self.Resize(sample['image'])
            sample['image'] = self.train_mode(sample['image'])

        # test mode transform
        elif self.mode == 'test' or self.mode == 'valid':
            if self.test_ac:
                test_size = self.test_activate_size
                if self.pad:
                    sample['image'] = expand2square(sample['image'])
                self.Resize =  transforms.Resize((test_size, test_size))
                sample['image'] = self.Resize(sample['image'])
                sample['image'] = self.test_mode(sample['image'])
            else:
                if self.pad:
                    sample['image'] = expand2square(sample['image'])
                sample['image'] = self.Resize(sample['image'])
                sample['image'] = self.test_mode(sample['image'])
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

def tensor2img(img):
    """
    tensor 형태에서 0~255 진행 후 numpy 배열로 바꿔서 반환
    """
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    a = unorm(img).numpy()
    a = a.transpose(1, 2, 0)
    return a