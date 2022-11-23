from pickle import TRUE
import re
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torch_optimizer as optim
from time import time
from torch.nn import Softmax
import numpy as np
import pandas as pd
import os
from random import choice
from sklearn.model_selection import KFold, StratifiedKFold
from torchvision import transforms
from tqdm import tqdm
from endecoder import *
from glob import glob
from sklearn.metrics import accuracy_score, f1_score
from utills import data_sampling, undersample, sub_down_sample
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, grad_scaler
from sklearn.model_selection import train_test_split

class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.aug = args.aug
        self.df = args.df
        pass
    
    def setup(self):
        create_directory(self.args.save_dict + '_stop')
        
        # model setup
        self.model = self.get_model(model=self.args.model_class, pretrained=self.args.pretrained)
        self.model.to(self.device)

        # 옵티마이저 정의
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.args.lr)
        elif self.args.optimizer == 'Lamb':
            self.optimizer = optim.Lamb(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)                                      
        
        # Loss 함수 정의
        if self.args.weight is not None:
            weights = torch.FloatTensor(self.args.weight).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights).to(self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.early_stopping = EarlyStopping(patience=3, verbose = True, path='known_80/{}_{}.pth'.format(self.args.CODER, self.aug))
        
    def fit(self):
        for b in range(self.args.bagging_num):
            print("bagging num : ", b)
            
            previse_name = ''
            best_model_name = ''
            valid_acc_max = 0
            best_loss = np.inf

            if self.args.fold_num <= 1:
                self.train, self.valid = train_test_split(self.df, shuffle=True, stratify=self.df['disease'])
                train_data_loader, valid_data_loader = self.sampling()
                self.setup()

                iter_per_epoch = len(train_data_loader)
                if self.args.scheduler == "cycle":
                    self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr, steps_per_epoch=iter_per_epoch, 
                                                                    epochs=self.args.epochs)
                elif self.args.scheduler == 'cos':
                    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.tmax, 
                                                                                    eta_min=self.args.min_lr, verbose=True) 
                self.warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * self.args.warm_epoch)
                
                for epoch in range(self.args.epochs):
                    if self.args.resampling:
                        if epoch % self.args.resample_inter == 0: 
                            train_data_loader = self.sampling(e=epoch, resample=True)

                    print("-" * 50)
                    if self.args.scheduler == 'cos':
                        if epoch > self.args.warm_epoch:
                            self.scheduler.step()
                    self.scaler = grad_scaler.GradScaler()
                    label_list, pred_list = self.training(train_data_loader, epoch)
                    
                    # 에폭별 평가 출력
                    train_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                    dis_acc = accuracy_score(np.array(label_list), np.array(pred_list))
                    print("epoch:{}, acc:{}, f1:{}".format(epoch, dis_acc, train_f1))

                    valid_losses, label_list, pred_list = self.validing(valid_data_loader, epoch)
                    valid_acc = accuracy_score(np.array(label_list), np.array(pred_list))
                    valid_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                    print("epoch:{}, acc:{}, f1:{}".format(epoch, valid_acc, valid_f1))

                    self.early_stopping(np.average(valid_losses), self.model)

                    # 모델 저장
                    if best_loss > np.average(valid_losses):
                        best_loss = np.average(valid_losses)
                        create_directory(self.args.save_dict)
                        
                        # model_name_bagging_kfold_bestmodel_valid loss로 이름 지정
                        best_model_name = self.args.save_dict + "/model_{}_{}_{:.4f}.pth".format(self.args.CODER, b, best_loss)
                        torch.save(self.model.state_dict(), best_model_name)
                        
                        # if isinstance(self.model, torch.nn.DataParallel): 
                        #     torch.save(self.model.module.state_dict(), best_model_name) 
                        # else:
                        #     torch.save(self.model.state_dict(), best_model_name) 
                        
                        if os.path.isfile(previse_name):
                            os.remove(previse_name)

                        # 갱신
                        previse_name = best_model_name
                    
                    if self.early_stopping.early_stop:
                        print("Early stopping")
                        break
        
            else:
                self.kfold = StratifiedKFold(n_splits=self.args.fold_num, shuffle=True)
                for fold_index, (trn_idx, val_idx) in enumerate(self.kfold.split(self.df, y=self.df['disease']),1):
                    self.train = self.df.iloc[trn_idx,]
                    self.valid  = self.df.iloc[val_idx,]
                    self.setup()
                    train_data_loader, valid_data_loader = self.sampling()

                    iter_per_epoch = len(train_data_loader)
                    if self.args.scheduler == "cycle":
                        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr, steps_per_epoch=iter_per_epoch, 
                                                                        epochs=self.args.epochs)
                    elif self.args.scheduler == 'cos':
                        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.tmax, 
                                                                                        eta_min=self.args.min_lr, verbose=True) 
                    self.warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * self.args.warm_epoch)
                    
                    for epoch in range(self.args.epochs):
                        if self.args.resampling:
                            if epoch % self.args.resample_inter == 0: 
                                train_data_loader = self.sampling(e=epoch, resample=True)

                        print("-" * 50)
                        if self.args.scheduler == 'cos':
                            if epoch > self.args.warm_epoch:
                                self.scheduler.step()
                        self.scaler = grad_scaler.GradScaler()
                        label_list, pred_list = self.training(train_data_loader, epoch)
                        
                        # 에폭별 평가 출력
                        train_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                        dis_acc = accuracy_score(np.array(label_list), np.array(pred_list))
                        print("epoch:{}, acc:{}, f1:{}".format(epoch, dis_acc, train_f1))

                        valid_losses, label_list, pred_list = self.validing(valid_data_loader, epoch)
                        valid_acc = accuracy_score(np.array(label_list), np.array(pred_list))
                        valid_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                        print("epoch:{}, acc:{}, f1:{}".format(epoch, valid_acc, valid_f1))

                        self.early_stopping(np.average(valid_losses), self.model)

                        # 모델 저장
                        if best_loss > np.average(valid_losses):
                            best_loss = np.average(valid_losses)
                            create_directory(self.args.save_dict)
                            
                            # model_name_bagging_kfold_bestmodel_valid loss로 이름 지정
                            best_model_name = self.args.save_dict + "/model_{}_{}_{}_{:.4f}.pth".format(self.args.CODER, b, fold_index, best_loss)
                            torch.save(self.model.state_dict(), best_model_name)
                            
                            if os.path.isfile(previse_name):
                                os.remove(previse_name)

                            # 갱신
                            previse_name = best_model_name
                        
                        if self.early_stopping.early_stop:
                            print("Early stopping")
                            break
    
    def sampling(self, e=0, resample=False):
        if self.args.resampling:
            self.aug = np.logspace(0, -2, self.args.epochs)[e]

        if resample:
            if self.args.mode == "full":
                train_answer = self.train
            elif self.args.mode == "under":
                train_answer = undersample(self.train, radio=self.aug)
            else:
                train_answer = sub_down_sample(self.train, self.args.index)
            
            train_dataset = self.args.Dataset(train_answer, mode='train', img_size = self.args.img_size, incrop=self.args.incrop,
                                            inrecrop=self.args.inrecrop, pad=self.args.pad)

            train_data_loader = DataLoader(
                train_dataset,
                batch_size = self.args.BATCH_SIZE,
                shuffle = True,
                num_workers = 8,
            )
            return train_data_loader
        else:
            if self.args.mode == "full":
                train_answer = self.train
                valid_answer = self.valid
            
            elif self.args.mode == "under":
                train_answer = undersample(self.train, radio=self.aug)
                valid_answer = undersample(self.valid, radio=self.aug)
            else:
                train_answer = sub_down_sample(self.train, self.args.index)
                valid_answer = sub_down_sample(self.valid, self.args.index)

            #Dataset 정의
            train_dataset = self.args.Dataset(train_answer, mode='train', img_size = self.args.img_size, incrop=self.args.incrop,
                                                inrecrop=self.args.inrecrop, pad=self.args.pad)
            
            train_data_loader = DataLoader(
                train_dataset,
                batch_size = self.args.BATCH_SIZE,
                shuffle = True,
                num_workers = 8,
            )
            valid_data_loader_list = []
            for crop in range(1, 11):
                crop_answer = valid_answer[valid_answer.crop == crop].reset_index(drop=True)
                valid_dataset = self.args.Dataset(crop_answer, mode='test', img_size = self.args.test_size, test_ac = self.args.test_ac,pad=self.args.pad)
                valid_data_loader = DataLoader(
                    valid_dataset,
                    batch_size = int(self.args.BATCH_SIZE / 2),
                    shuffle = False,
                    num_workers = 4,
                )
                valid_data_loader_list.append(valid_data_loader)
            return train_data_loader, valid_data_loader_list


    def training(self, train_data_loader, epoch):
        self.model.train()
        pred_list, label_list = [], []
        with tqdm(train_data_loader,total=train_data_loader.__len__(), unit="batch") as train_bar:
            for batch_idx, batch_data in enumerate(train_bar):
                train_bar.set_description(f"Train Epoch {epoch}")
                images, dis_label, crop_label = batch_data['image'], batch_data['disease_label'], batch_data['crop_label']
                images, dis_label, crop_label = Variable(images.cuda()), Variable(dis_label.cuda()), Variable(crop_label.cuda())
                
                if epoch <= self.args.warm_epoch:
                    self.warmup_scheduler.step()

                with torch.set_grad_enabled(True):
                    self.model.zero_grad(set_to_none=True)
                    if self.args.amp:
                        with autocast():
                            if self.args.aware:
                                dis_out  = self.model(images, crop_label) 
                            else:
                                dis_out  = self.model(images) 
                            
                            dis_loss = self.criterion(dis_out, dis_label)
                            loss = dis_loss
                        self.scaler.scale(loss).backward()

                        # Gradient Clipping
                        if self.args.clipping is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clipping)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        if self.args.aware:
                            dis_out  = self.model(images, crop_label) 
                        else:
                            dis_out  = self.model(images) 
                        dis_loss = self.criterion(dis_out, dis_label)
                        loss = dis_loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clipping)
                        self.optimizer.step()

                    if self.args.scheduler == 'cycle':
                        if epoch > self.args.warm_epoch:
                            self.scheduler.step()

                    # 질병 예측 라벨화
                    dis_out = torch.argmax(dis_out, dim=1).detach().cpu()
                    dis_label =dis_label.detach().cpu()
                    
                    pred_list.extend(dis_out.numpy())
                    label_list.extend(dis_label.numpy())

                batch_acc = (dis_out == dis_label).to(torch.float).numpy().mean()
                train_bar.set_postfix(train_loss= loss.item(), 
                                        train_batch_acc = batch_acc,
                                        # F1 = train_f1,
                                    )
        return label_list, pred_list
    
    def validing(self, valid_data_loader_list, epoch):
        valid_dis_acc_list = []
        valid_losses = []
        self.model.eval()
        pred_list, label_list = [], []
        for valid_data_loader in valid_data_loader_list:
            with tqdm(valid_data_loader,total=valid_data_loader.__len__(), unit="batch") as valid_bar:
                for batch_idx, batch_data in enumerate(valid_bar):
                    valid_bar.set_description(f"Valid Epoch {epoch}")
                    images, dis_label, crop_label = batch_data['image'], batch_data['disease_label'], batch_data['crop_label']
                    images, dis_label, crop_label = Variable(images.cuda()), Variable(dis_label.cuda()), Variable(crop_label.cuda())
                
                    with torch.no_grad():
                        if self.args.aware:
                            dis_out  = self.model(images, crop_label) 
                        else:
                            dis_out  = self.model(images) 
                        
                        # loss 계산
                        dis_loss = self.criterion(dis_out, dis_label)
                        valid_loss = dis_loss

                        dis_out = torch.argmax(dis_out, dim=1).detach().cpu()
                        dis_label =dis_label.detach().cpu()
                        
                        pred_list.extend(dis_out.numpy())
                        label_list.extend(dis_label.numpy())

                    # accuracy_score(dis_label, dis_out)
                    dis_acc = (dis_out == dis_label).to(torch.float).numpy().mean()

                    # print(dis_acc, crop_acc)
                    valid_dis_acc_list.append(dis_acc)

                    valid_losses.append(valid_loss.item())
                    valid_dis_acc = np.mean(valid_dis_acc_list)
            
                    valid_bar.set_postfix(valid_loss = valid_loss.item(), 
                                            valid_batch_acc = valid_dis_acc,
                                            )
        return valid_losses, label_list, pred_list

    
    def get_model(self, model, pretrained=False):
        mdl = torch.nn.DataParallel(model(self.args)) if self.args.multi_gpu else model(self.args)
        if not pretrained:
            return mdl
        else:
            print("기학습 웨이트")
            mdl.load_state_dict(torch.load(pretrained))
            return mdl

class Mixup_trainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.aug = args.aug
        self.df = args.df
        pass
    
    def setup(self):
        create_directory(self.args.save_dict + '_stop')
        
        # model setup
        self.model = self.get_model(model=self.args.model_class, pretrained=self.args.pretrained)
        self.model.to(self.device)

        # 옵티마이저 정의
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.args.lr)
        elif self.args.optimizer == 'Lamb':
            self.optimizer = optim.Lamb(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)                                      
        
        # Loss 함수 정의
        if self.args.weight is not None:
            weights = torch.FloatTensor(self.args.weight).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights).to(self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.early_stopping = EarlyStopping(patience=15, verbose = True, path='known_80/{}_{}.pth'.format(self.args.CODER, self.aug))
        
    def fit(self):
        for b in range(self.args.bagging_num):
            print("bagging num : ", b)
            
            previse_name = ''
            best_model_name = ''
            valid_acc_max = 0
            best_loss = np.inf

            self.train, self.valid = train_test_split(self.df, shuffle=True, stratify=self.df['disease'])
            train_data_loader, valid_data_loader = self.sampling()
            self.setup()

            iter_per_epoch = len(train_data_loader)
            if self.args.scheduler == "cycle":
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.max_lr, steps_per_epoch=iter_per_epoch, 
                                                                epochs=self.args.epochs)
            elif self.args.scheduler == 'cos':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.tmax, 
                                                                                eta_min=self.args.min_lr, verbose=True) 
            self.warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * self.args.warm_epoch)
            
            for epoch in range(self.args.epochs):
                print("-" * 50)
                if self.args.scheduler == 'cos':
                    if epoch > self.args.warm_epoch:
                        self.scheduler.step()
                self.scaler = grad_scaler.GradScaler()
                self.training(train_data_loader, epoch)
                valid_losses = self.validing(valid_data_loader, epoch)

                self.early_stopping(valid_losses, self.model)

                # 모델 저장
                if best_loss > valid_losses:
                    best_loss = valid_losses
                    create_directory(self.args.save_dict)
                    
                    # model_name_bagging_kfold_bestmodel_valid loss로 이름 지정
                    best_model_name = self.args.save_dict + "/model_{}_{}_{:.4f}.pth".format(self.args.CODER, b, best_loss)
                    torch.save(self.model.state_dict(), best_model_name)
                    
                    # if isinstance(self.model, torch.nn.DataParallel): 
                    #     torch.save(self.model.module.state_dict(), best_model_name) 
                    # else:
                    #     torch.save(self.model.state_dict(), best_model_name) 
                    
                    if os.path.isfile(previse_name):
                        os.remove(previse_name)

                    # 갱신
                    previse_name = best_model_name
                
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
        
    def sampling(self):
        train_answer = self.train
        valid_answer = self.valid

        train_dataset = self.args.Dataset(train_answer, mode='train', img_size = self.args.img_size, incrop=self.args.incrop,
                                        inrecrop=self.args.inrecrop, pad=self.args.pad)
        valid_dataset = self.args.Dataset(valid_answer, mode='test', img_size = self.args.test_size, test_ac = self.args.test_ac,
                                            pad=self.args.pad)

        train_data_loader = DataLoader(
            train_dataset,
            batch_size = self.args.BATCH_SIZE,
            shuffle = True,
            # num_workers = 8,
        )
        valid_data_loader = DataLoader(
                    valid_dataset,
                    batch_size = int(self.args.BATCH_SIZE / 2),
                    shuffle = False,
                    # num_workers = 4,
                )

        return train_data_loader, valid_data_loader


    def training(self, train_data_loader, epoch):
        self.model.train()
        train_loss = 0
        with tqdm(train_data_loader,total=train_data_loader.__len__(), unit="batch") as train_bar:
            for batch_idx, batch_data in enumerate(train_bar):
                train_bar.set_description(f"Train Epoch {epoch}")
                images, dis_label, crop_label = batch_data['image'], batch_data['disease_label'], batch_data['crop_label']
                images, dis_label, crop_label = Variable(images.cuda()), Variable(dis_label.cuda()), Variable(crop_label.cuda())
                images, targets_a, targets_b, lam = mixup_data(images, dis_label, self.args.device, 1.0, True)
                
                if epoch <= self.args.warm_epoch:
                    self.warmup_scheduler.step()

                with torch.set_grad_enabled(True):
                    self.model.zero_grad(set_to_none=True)
                    if self.args.amp:
                        with autocast():
                            if self.args.aware:
                                dis_out  = self.model(images, crop_label) 
                            else:
                                dis_out  = self.model(images) 
                            
                            loss = mixup_criterion(self.criterion, dis_out, targets_a, targets_b, lam)

                        self.scaler.scale(loss).backward()

                        # Gradient Clipping
                        if self.args.clipping is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clipping)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        if self.args.aware:
                            dis_out  = self.model(images, crop_label) 
                        else:
                            dis_out  = self.model(images) 
                        loss = mixup_criterion(self.criterion, dis_out, targets_a, targets_b, lam)
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clipping)
                        self.optimizer.step()

                    if self.args.scheduler == 'cycle':
                        if epoch > self.args.warm_epoch:
                            self.scheduler.step()

                    train_loss += loss.item()

                train_bar.set_postfix(train_loss= train_loss/(batch_idx+1))
    
    def validing(self, valid_data_loader, epoch):
        self.model.eval()
        vaild_loss = 0
        with tqdm(valid_data_loader,total=valid_data_loader.__len__(), unit="batch") as valid_bar:
            for batch_idx, batch_data in enumerate(valid_bar):
                valid_bar.set_description(f"Valid Epoch {epoch}")
                images, dis_label, crop_label = batch_data['image'], batch_data['disease_label'], batch_data['crop_label']
                images, dis_label, crop_label = Variable(images.cuda()), Variable(dis_label.cuda()), Variable(crop_label.cuda())
                images, targets_a, targets_b, lam = mixup_data(images, dis_label, self.args.device, 1.0, True)
                with torch.no_grad():
                    if self.args.aware:
                        dis_out  = self.model(images, crop_label) 
                    else:
                        dis_out  = self.model(images) 
                    
                    # loss 계산
                    loss = mixup_criterion(self.criterion, dis_out, targets_a, targets_b, lam)             
                    vaild_loss += loss.item()
        
                valid_bar.set_postfix(valid_loss = vaild_loss/(batch_idx+1))
        return vaild_loss/(batch_idx+1)

    
    def get_model(self, model, pretrained=False):
        mdl = torch.nn.DataParallel(model(self.args)) if self.args.multi_gpu else model(self.args)
        if not pretrained:
            return mdl
        else:
            print("기학습 웨이트")
            mdl.load_state_dict(torch.load(pretrained))
            return mdl

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 3
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
            return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, device, alpha=0.4, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
