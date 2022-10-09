# trainer 함수
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
from sklearn.model_selection import KFold
from torchvision import transforms
from tqdm import tqdm
from endecoder import *
from glob import glob
from sklearn.metrics import accuracy_score, f1_score
from utills import data_sampling, undersample, sub_down_sample
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, grad_scaler

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
    
def train_model(args):
    """
    root : dataset dir
    save_dict : model weight save path
    df0 : 질병 데이터셋
    df1 : 정상 데이터셋
    df2 : 증강 데이터셋

    CODER : model name
    aware : 작물의 정보를 아는 경우
    aug : 증강 데이터 사용 비율
    mode : 데이터셋 구성 full or sub
    """
    
    def get_model(model, m=True, pretrained=False):
        # multi-GPU일 경우, Data Parallelism
        mdl = torch.nn.DataParallel(model(args)) if m else model()
        if not pretrained:
            return mdl
        else:
            print("load pretrained model here...")
            # 학습된 모델을 불러온다
            mdl.load_state_dict(torch.load(pretrained))
            return mdl
    
    # bagging_num 만큼 모델 학습을 반복 수행한다
    for b in range(args.bagging_num):
        print("bagging num : ", b)
        create_directory(args.save_dict + 'stop')
        # 교차 검증
        kfold = KFold(n_splits=args.fold_num, shuffle=True)
        best_models = [] 
        
        previse_name = ''
        best_model_name = ''
        valid_acc_max = 0
        best_loss = np.inf
        if args.mode == "full":
            files = undersample(args.df0, args.df1, args.df2, mode="train", radio=args.aug)
        else:
            files = sub_down_sample(args.df0, args.index)
        # print(files)
        for fold_index, (trn_idx, val_idx) in enumerate(kfold.split(files),1):
            print(f'[fold: {fold_index}]')
            torch.cuda.empty_cache()
            
            # kfold dataset 구성
            train_answer = files.iloc[trn_idx,].reset_index(drop=True)
            test_answer  = files.iloc[val_idx,].reset_index(drop=True)

            #Dataset 정의
            train_dataset = args.Dataset(train_answer, mode='train', img_size = args.img_size)
            valid_dataset = args.Dataset(test_answer, mode='test', img_size = args.img_size)
            print(args.img_size)
            #DataLoader 정의
            
            train_data_loader = DataLoader(
                train_dataset,
                batch_size = args.BATCH_SIZE,
                shuffle = True,
                num_workers = 8,
            )
            valid_data_loader = DataLoader(
                valid_dataset,
                batch_size = int(args.BATCH_SIZE / 2),
                shuffle = False,
                num_workers = 4,
            )

            # model setup
            model = get_model(model=args.model_class)
            model.to(args.device)

            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
            elif args.optimizer == 'Lamb':
                optimizer = optim.Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            iter_per_epoch = len(train_data_loader)
            if args.scheduler == "cycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=iter_per_epoch, epochs=args.epochs)
            elif args.scheduler == 'cos':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, 
                                                                                eta_min=args.min_lr, verbose=True)                                       
            
            criterion = torch.nn.CrossEntropyLoss().to(args.device)
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm_epoch)
            early_stopping = EarlyStopping(patience=3, verbose = True, path='known_80/{}_{}_{}.pth'.format(args.CODER, b, fold_index))


            if args.train_var:
                for epoch in range(args.epochs):
                    train_acc_list = []
                    dis_acc = 0
                    pred_list, label_list = [], []
                    model.train()
                    print("-" * 50)

                    if args.scheduler == 'cos':
                        if epoch > args.warm_epoch:
                            scheduler.step()
                    scaler = grad_scaler.GradScaler()

                    with tqdm(train_data_loader,total=train_data_loader.__len__(), unit="batch") as train_bar:
                        for batch_idx, batch_data in enumerate(train_bar):
                            train_bar.set_description(f"Train Epoch {epoch}")
                            images, dis_label, crop_label = batch_data['image'], batch_data['disease_label'], batch_data['crop_label']
                            images, dis_label, crop_label = Variable(images.cuda()), Variable(dis_label.cuda()), Variable(crop_label.cuda())
                            
                            if epoch <= args.warm_epoch:
                                warmup_scheduler.step()

                            with torch.set_grad_enabled(True):
                                model.zero_grad(set_to_none=True)
                                if args.amp:
                                    with autocast():
                                        if args.aware:
                                            dis_out  = model(images, crop_label) 
                                        else:
                                            dis_out  = model(images) 
                                        
                                        dis_loss = criterion(dis_out, dis_label)
                                        loss = dis_loss
                                    scaler.scale(loss).backward()

                                    # Gradient Clipping
                                    if args.clipping is not None:
                                        scaler.unscale_(optimizer)
                                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping)

                                    scaler.step(optimizer)
                                    scaler.update()

                                else:
                                    if args.aware:
                                        dis_out  = model(images, crop_label) 
                                    else:
                                        dis_out  = model(images) 
                                    dis_loss = criterion(dis_out, dis_label)
                                    loss = dis_loss
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping)
                                    optimizer.step()

                                if args.scheduler == 'cycle':
                                    if epoch > args.warm_epoch:
                                        scheduler.step()

                                # 질병 예측 라벨화
                                dis_out = torch.argmax(dis_out, dim=1).detach().cpu()
                                dis_label =dis_label.detach().cpu()
                                
                                pred_list.extend(dis_out.numpy())
                                label_list.extend(dis_label.numpy())

                            batch_acc = (dis_out == dis_label).to(torch.float).numpy().mean()
                            dis_acc += batch_acc
                            train_bar.set_postfix(train_loss= loss.item(), 
                                                    train_batch_acc = batch_acc,
                                                    # F1 = train_f1,
                                                )
                        # 에폭별 평가 출력
                        train_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                        dis_acc = accuracy_score(np.array(label_list), np.array(pred_list))
                        print("epoch:{}, acc:{}, f1:{}".format(epoch, dis_acc, train_f1))
                                                                                                    
                    # epoch마다 valid 계산
                    valid_dis_acc_list = []
                    valid_losses = []
                    valid_acc = 0
                    model.eval()

                    pred_list, label_list = [], []
                    with tqdm(valid_data_loader,total=valid_data_loader.__len__(), unit="batch") as valid_bar:
                        for batch_idx, batch_data in enumerate(valid_bar):
                            valid_bar.set_description(f"Valid Epoch {epoch}")
                            images, dis_label, crop_label = batch_data['image'], batch_data['disease_label'], batch_data['crop_label']
                            images, dis_label, crop_label = Variable(images.cuda()), Variable(dis_label.cuda()), Variable(crop_label.cuda())
                        
                            with torch.no_grad():
                                if args.aware:
                                    dis_out  = model(images, crop_label) 
                                else:
                                    dis_out  = model(images) 
                                
                                # loss 계산
                                dis_loss = criterion(dis_out, dis_label)
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
                    
                            valid_acc += valid_dis_acc
                            valid_bar.set_postfix(valid_loss = valid_loss.item(), 
                                                    valid_batch_acc = valid_dis_acc,
                                                    )
                        valid_acc = accuracy_score(np.array(label_list), np.array(pred_list))
                        valid_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                        print("epoch:{}, acc:{}, f1:{}".format(epoch, valid_acc, valid_f1))
                    
                    early_stopping(np.average(valid_losses), model)

                    # 모델 저장
                    if best_loss > np.average(valid_losses):
                        best_loss = np.average(valid_losses)
                        best_model = model
                        create_directory(args.save_dict)
                        
                        # model_name_bagging_kfold_bestmodel_valid loss로 이름 지정
                        best_model_name = args.save_dict + "/model_{}_{}_{}_{:.4f}.pth".format(args.CODER, b, fold_index, best_loss)
                        torch.save(model.state_dict(), best_model_name)
                        
                        if os.path.isfile(previse_name):
                            os.remove(previse_name)

                        # 갱신
                        previse_name = best_model_name
                    
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
            else:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, 
                                                                        eta_min=0.001, last_epoch=-1) 
                for epoch in range(args.epochs):
                    train_acc_list = []
                    dis_acc = 0
                    pred_list, label_list = [], []
                    model.train()
                    print("-" * 50)
                    optimizer.zero_grad()

                    with tqdm(train_data_loader,total=train_data_loader.__len__(), unit="batch") as train_bar:
                        for batch_idx, batch_data in enumerate(train_bar):
                            train_bar.set_description(f"Train Epoch {epoch}")
                            images, dis_label, crop_label = batch_data['image'], batch_data['disease_label'], batch_data['crop_label']
                            images, dis_label, crop_label = Variable(images.cuda()), Variable(dis_label.cuda()), Variable(crop_label.cuda())

                            with torch.set_grad_enabled(True):
                                # 모델 예측
                                dis_out  = model(images, crop_label) #crop_label
                                # loss 계산
                                dis_loss = criterion(dis_out, dis_label)
                                loss = dis_loss
                                loss.backward()
                                optimizer.step()

                                # train accuracy 계산
                                dis_out = torch.argmax(dis_out, dim=1).detach().cpu()
                                # dis_label = torch.argmax(dis_label, dim=1).detach().cpu()
                                dis_label =dis_label.detach().cpu()
                                
                                pred_list.extend(dis_out.numpy())
                                label_list.extend(dis_label.numpy())

                            batch_acc = (dis_out == dis_label).to(torch.float).numpy().mean()
                            dis_acc += batch_acc
                            train_bar.set_postfix(train_loss= loss.item(), 
                                                    train_batch_acc = batch_acc,
                                                    # F1 = train_f1,
                                                )
                        dis_acc /= len(train_data_loader.dataset)
                        train_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                        print("epoch:{}, acc:{}, f1:{}".format(epoch, dis_acc, train_f1))
                                                                                                    
                    # epoch마다 valid 계산
                    valid_dis_acc_list = []
                    valid_losses = []
                    valid_acc = 0
                    model.eval()

                    pred_list, label_list = [], []
                    with tqdm(valid_data_loader,total=valid_data_loader.__len__(), unit="batch") as valid_bar:
                        for batch_idx, batch_data in enumerate(valid_bar):
                            valid_bar.set_description(f"Valid Epoch {epoch}")
                            images, dis_label, crop_label = batch_data['image'], batch_data['disease_label'], batch_data['crop_label']
                            images, dis_label, crop_label = Variable(images.cuda()), Variable(dis_label.cuda()), Variable(crop_label.cuda())
                            
                            with torch.no_grad():
                                if args.aware:
                                    dis_out  = model(images, crop_label)
                                else:
                                    dis_out  = model(images)
                                
                                # loss 계산
                                dis_loss = criterion(dis_out, dis_label)
                                valid_loss = dis_loss

                                dis_out = torch.argmax(dis_out, dim=1).detach().cpu()
                                # dis_label = torch.argmax(dis_label, dim=1).detach().cpu()
                                dis_label =dis_label.detach().cpu()
                                
                                pred_list.extend(dis_out.numpy())
                                label_list.extend(dis_label.numpy())

                            # accuracy_score(dis_label, dis_out)
                            dis_acc = (dis_out == dis_label).to(torch.float).numpy().mean()

                            # print(dis_acc, crop_acc)
                            valid_dis_acc_list.append(dis_acc)

                            valid_losses.append(valid_loss.item())
                            valid_dis_acc = np.mean(valid_dis_acc_list)
                    
                            valid_acc += valid_dis_acc
                            valid_bar.set_postfix(valid_loss = valid_loss.item(), 
                                                    valid_batch_acc = valid_dis_acc,
                                                    )
                        valid_acc /= len(valid_data_loader.dataset)
                        valid_f1 = f1_score(np.array(label_list), np.array(pred_list), average='macro')
                        print("epoch:{}, acc:{}, f1:{}".format(epoch, valid_acc, valid_f1))
                    
                    # Learning rate 조절
                    lr_scheduler.step()  
                    # create_directory("sub02")
                    early_stopping(np.average(valid_losses), model)

                    # 모델 저장
                    if best_loss > np.average(valid_losses):
                        best_loss = np.average(valid_losses)
                        best_model = model
                        create_directory(args.save_dict)
                        
                        # model_name_bagging_kfold_bestmodel_valid loss로 이름 지정
                        best_model_name = args.save_dict + "/model_{}_{}_{}_{:.4f}.pth".format(args.CODER, b, fold_index, best_loss)
                        torch.save(model.state_dict(), best_model_name)
                        
                        if os.path.isfile(previse_name):
                            os.remove(previse_name)

                        # 갱신
                        previse_name = best_model_name
                    
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
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