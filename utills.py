import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import os
import torch

from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from sklearn.preprocessing import Binarizer
def data_sampling(root, aug=None, num=216, mode="train"):
    label_dir = os.path.join(root, "label")
    label_files = glob(os.path.join(label_dir, "*_0"))
    label_files1 = glob(os.path.join(label_dir, "*_1"))
    
    if aug:
        label_files_2 = glob(os.path.join(label_dir, "*_2"))
        aug_num = int(3240 * aug)
        real_num = int((num + aug_num) / 2.)
    
    else:
        real_num = int(num / 2.)

    if mode == "train":
        
        for i in range(10):
            
            # 질병 데이터 불러오기
            files_0 = glob(os.path.join(label_dir, label_files[i].split("\\")[-1]) + '/*')
            # 정상 데이터 불러오기
            files_1 = glob(os.path.join(label_dir, label_files1[i].split("\\")[-1]) + '/*')
            
            if i == 0:
                train_list = np.random.choice(files_0, num, replace= False)
                train_list = np.r_[train_list, np.random.choice(files_1, real_num, replace= False)]
                
                if aug:
                    # 증강 데이터
                    files_2 = glob(os.path.join(label_dir, label_files_2[i].split("\\")[-1]) + '/*')
                    train_list = np.r_[train_list, np.random.choice(files_2, aug_num, replace= False)]
            else:
                train_list = np.r_[train_list, np.random.choice(files_0, num, replace= False)]
                train_list = np.r_[train_list, np.random.choice(files_1, real_num, replace= False)]
                
                if aug:
                    files_2 = glob(os.path.join(label_dir, label_files_2[i].split("\\")[-1]) + '/*')
                    train_list = np.r_[train_list, np.random.choice(files_2, aug_num, replace= False)]
            

        return train_list
    
    else:
        valid_list = []
        for i in range(10):
            
            # 질병 데이터 불러오기
            files_0 = glob(os.path.join(label_dir, label_files[i].split("\\")[-1]) + '/*')
            # 정상 데이터 불러오기
            files_1 = glob(os.path.join(label_dir, label_files1[i].split("\\")[-1]) + '/*')
            
            valid_list.extend(files_0)
            valid_list.extend(files_1)
            
            if aug:
                files_2 = glob(os.path.join(label_dir, label_files_2[i].split("\\")[-1]) + '/*')
                valid_list.extend(files_2)
            
        return valid_list


def undersample(df0, df1, df2, mode="train", radio=0.2):
    idx0, idx1, idx2 = [], [], []
    
    # 질병
    aug_num = int(2700 * radio)
    for di in [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 24, 25, 26, 28, 29]:
        idx0.extend(np.random.choice(df0[df0['disease'] == di].index, 180, replace= False))
        idx2.extend(np.random.choice(df2[df2['disease'] == di].index, aug_num, replace= False))
    for di in [0, 3, 6, 9, 12, 15, 18, 21, 23, 27]:
        idx1.extend(np.random.choice(df1[df1['disease'] == di].index, 180 + aug_num, replace= False))
    sub0 = df0.iloc[idx0, ]
    sub1 = df1.iloc[idx1, ]
    sub2 = df2.iloc[idx2, ]
    
    return pd.concat([sub0, sub1, sub2]).reset_index(drop=True)


def sub_down_sample(df, index):
    idx0 = []
    down_num = np.min([len(df[df['disease'] == idx]) for idx in index])
    
    for idx in index:
        idx0.extend(np.random.choice(df[df['disease'] == idx].index, down_num, replace= False))
    sub = df.iloc[idx0, ]
    return sub.reset_index(drop=True)

def evaluation(model_class, test_data_loader=None, m=False, path=None, score=None, device=None):
    """
    model_class : 사용할 모델
    test_data_loader : 데이터 로더
    m : 멀티 GPU사용
    path : 모델 파라미터 저장 경로
    score : greedy 앙상블 시 각 모델의 정확도 값
    """
    def get_model(model=model_class, m=m, pretrained=False):
            # multi-GPU일 경우, Data Parallelism
        mdl = torch.nn.DataParallel(model()) if m else model()
        if not pretrained:
            return mdl
        else:
            print("load pretrained model here...")
            # 기학습된 torch.load()로 모델을 불러온다
            mdl.load_state_dict(torch.load(pretrained))
            return mdl

    model = get_model()
    model.to(device)

    if not score:
        for e, m in enumerate(path):
            model.load_state_dict(torch.load(m))
            pred_scores = []
            
            for batch_idx, batch_data in enumerate(test_data_loader):
                with torch.no_grad():
                    # 추론
                    model.eval()
                    images = batch_data['image']
                    images = images.to(device)
                    probs  = model(images)
                    probs = probs.cpu().detach().numpy()
                pred_scores.append(probs)

            # 앙상블 0
            if e == 0:
                final_pred = np.vstack(pred_scores)
                # final_test_fnames = test_fnames
            else:
                final_pred += np.vstack(pred_scores)
                # assert final_test_fnames == test_fnames
        final_pred /= len(path)

        # threshold     
        return final_pred
    else:
        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x))

        model_acc_weight = softmax(score)
        for e, m in enumerate(path):
            model.load_state_dict(torch.load(m))
            pred_scores = []
         
            for batch_idx, batch_data in enumerate(test_data_loader):
                with torch.no_grad():
                    # 추론
                    model.eval()
                    images = batch_data['image']
                    images = images.to(device)
                    probs  = model(images)
                    probs = probs.cpu().detach().numpy()
                pred_scores.append(probs)

            
            # 앙상블 0
            if e == 0:
                final_pred = np.vstack(pred_scores) * model_acc_weight[e]
                # final_test_fnames = test_fnames
            else:
                final_pred += np.vstack(pred_scores) * model_acc_weight[e]
        
        return final_pred

def ooddata(mode="train", size=200):
    if mode == "train":
        files00 = glob(r"D:\고추ood\image\p1\*")
    else:
        files00 = glob(r"D:\고추ood\image\p2\*")
    ood_list = np.random.choice(files00, size, replace= False)
    radio = int(size / 2.)
    files10 = glob(r"D:\노지 작물 질병 진단 이미지\train\images\pepper_0\*")
    files11 = glob(r"D:\노지 작물 질병 진단 이미지\train\images\pepper_1\*")
    
    train_list = np.random.choice(files10, radio, replace= False)
    train_list = np.r_[train_list, np.random.choice(files11, size - radio, replace= False)]
    
    label_list = [1] * size
    label_list.extend([0] * size)
    
    img_list = np.r_[ood_list , train_list]
    data = np.c_[img_list, label_list]
    return data

def nodata(num = 100):

    files10 = glob(r"D:\노지 작물 질병 진단 이미지\train\images\pepper_0\*")
    files11 = glob(r"D:\노지 작물 질병 진단 이미지\train\images\pepper_1\*")
    
    train_list = np.random.choice(files10, num, replace= False)
    train_list = np.r_[train_list, np.random.choice(files11, num, replace= False)]
    
    label_list = [0] * num
    data = np.c_[train_list, label_list]
    return data

# 정확도, 정밀도, 재현율, F1 점수를 출력
def get_clf_eval(y_test , pred):
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}'.format(accuracy, precision, recall, f1))
    

# 임계값 변경   
def get_eval_threshold(y_test, pred, thresholds):
    for custom_thr in thresholds:
        cus_pred = Binarizer(threshold=custom_thr).fit_transform(pred)
        print("*" * 50)
        print(f"Thresholds: {custom_thr}")
        get_clf_eval(y_test, cus_pred)
        
# 다중레이블 분류 혼동행렬 및 분류 리포트 출력
def get_clf_eval_multi(y_test, pred, labels):
    confusion = multilabel_confusion_matrix(y_test, pred)
    # labels = dirty_mnist_answer.columns[1:].tolist()
    conf_mat={}
    for label_col in range(len(labels)):
        y_true_label = y_test[:, label_col]
        y_pred_label = pred[:, label_col]
        conf_mat[labels[label_col]] = confusion_matrix(y_true=y_true_label, y_pred=y_pred_label)
    for label, mat in conf_mat.items():
        print(f"Confusion mat for label : {label}")
        print(mat)
    print("---------Report---------")
    print(classification_report(y_test, pred, target_names=labels))

# ROC 커브 출력
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.grid(True)

