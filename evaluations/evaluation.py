from PIL import Image
import numpy as np
import os
import torch
import argparse
import cv2
import pandas as pd
from  sklearn.metrics import f1_score, precision_recall_curve, log_loss, roc_auc_score as AUC
from sklearn.metrics import roc_curve
import torchvision.transforms.functional as tf
import torchvision
import torch.nn.functional as f
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
from torch import nn

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--dataroot', type=str, default='', help='dataset_dir')
    parser.add_argument('--result_root', type=str, default='', help='dataset_dir')
    parser.add_argument('--filename', type=str, default='', help='dataset_dir')
    parser.add_argument('--dataset_name', type=str, default='ff-all', help='dataset_name')
    parser.add_argument('--evaluation_type', type=str, default="our", help='evaluation type')
    parser.add_argument('--ssim_window_size', type=int, default=11, help='ssim window size')

    return parser.parse_args()



def ff_metrics(testset):
    result=dict()
    temp_set=dict()
    for k,j in enumerate(['Origin','Deepfakes','NeuralTextures','FaceSwap','Face2Face']):
        d=testset[k*140:(k+1)*140]
        temp_set[j]=d

    for i in ['Deepfakes','NeuralTextures','FaceSwap','Face2Face','all']:
        if i!='all':
            rs=test_metric(temp_set[i]+temp_set['Origin'])
        else:
            rs=test_metric(testset) 
        result[i]=rs
    return result

def test_metric(testset):
    video_labels=[]
    video_preds=[]
    for i in testset:
        video_preds.append(i['pred'])
        video_labels.append(i['label'])
    video_thres,video_acc,video_f1=acc_f1_eval(video_labels,video_preds)
    video_auc=AUC(video_labels,video_preds)
    video_log_loss = log_loss(video_labels, video_preds, labels=[0, 1])
    rs={'video_acc':video_acc,'video_threshold':video_thres,'video_auc':video_auc,'video_f1':video_f1, 'video_log_loss':video_log_loss}
    return rs

def acc_eval(labels,preds):
    labels=np.array(labels)
    preds=np.array(preds)
    thres=0.5
    acc=np.mean((preds>=thres)==labels)
    return thres,acc

def acc_f1_eval(labels,preds):
    labels=np.array(labels)
    preds=np.array(preds)
    # lr_precision, lr_recall, _ = precision_recall_curve(labels, preds)
    
    thres=0.5
    thres_result = (preds>=thres)==labels
    acc=np.mean(thres_result)
    f1 = f1_score(labels, thres_result)
    return thres,acc,f1

if __name__ == '__main__':
    opt = parse_args()
    if opt is None:
        exit()
    path = os.path.join(opt.result_root)
    results = pd.read_csv(path)


    label = results['label'].values
    pred = results['pred'].values
    filename = results['filename'].values
    # print(filename)
    results_list = []
    preds = []
    labela = []
    for i, name in enumerate(filename):
        tmp = dict(filename=name, pred=pred[i], label=label[i])
        results_list.append(tmp)

        preds.append(pred[i])
        labela.append(label[i])

    if opt.dataset_name == "ff-all":
        results = ff_metrics(results_list)
    else:
        results = None
    print(results)

    # label = torch.Tensor(labela).unsqueeze(dim=1).float()
    # predict = torch.Tensor(preds).unsqueeze(dim=1).float()
    # loss = nn.BCEWithLogitsLoss()
    # print(loss(label, predict))