from PIL import Image
import numpy as np
import os
import time
import torch
import argparse
import cv2
import pandas as pd
from  sklearn.metrics import f1_score, precision_recall_curve, log_loss, roc_auc_score as AUC, confusion_matrix,roc_curve, auc
from sklearn.metrics import roc_auc_score
import torchvision.transforms.functional as tf
import torchvision
import torch.nn.functional as f
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
from torch import nn
import glob
import matplotlib.pyplot as plt

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_root', type=str, default='', help='dataset_dir')
    parser.add_argument('--dataset_name', type=str, default='FF-ALL', help='dataset_name')

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

def basic_eval(labels, preds,result_root):
    labels=np.array(labels)
    preds=np.array(preds)
    thres=0.5
    thres_result = preds>=thres
    real_basic = confusion_matrix(labels, labels, labels=[0,1])
    pred_basic = confusion_matrix(labels, thres_result, labels=[0,1])
    # (tn, fp, fn, tp)
    # print(basic.ravel())
    tn, fp, fn, tp = pred_basic.ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    # save_roc_image(labels, preds, result_root)
    return str(real_basic.ravel()), str(pred_basic.ravel()),precision,recall

def save_roc_image(labels, preds,result_root):
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(result_root+'roc.jpg')
    plt.show()

def produce_result(labels, predicts, result_root):
    real_basic, pred_basic,precision,recall = basic_eval(labels, predicts, result_root)
    video_thres, video_acc, video_f1=acc_f1_eval(labels,predicts)
    video_auc=roc_auc_score(labels,predicts)
    return video_acc, video_auc, pred_basic,precision,recall

def get_ff_result(videos, predicts, labels,result_root):
    fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures','FF-ALL']
    result_grps = {'Deepfakes':{'labels':[],'predicts':[]}, 'Face2Face':{'labels':[],'predicts':[]}, \
        'FaceSwap':{'labels':[],'predicts':[]}, 'NeuralTextures':{'labels':[],'predicts':[]}, \
            'original_sequences':{'labels':[],'predicts':[]}}

    for video, predict, label in zip(videos, predicts, labels):
        if 'original_sequences' in video:
            result_grps['original_sequences']['labels'].append(label)
            result_grps['original_sequences']['predicts'].append(predict)
        else:
            fake_type = video.split('/')[1]
            result_grps[fake_type]['labels'].append(label)
            result_grps[fake_type]['predicts'].append(predict)
    
    results = []
    labels_r = result_grps['original_sequences']['labels']
    predicts_r = result_grps['original_sequences']['predicts']
    for fake_type in fake_types:
        if fake_type == 'FF-ALL':
            labels_f, predicts_f = [], []
            for fake_type_1 in fake_types[:-1]:
                labels_f += result_grps[fake_type_1]['labels']
                predicts_f += result_grps[fake_type_1]['predicts']
            labels_rf = labels_r + labels_f
            predicts_rf = predicts_r + predicts_f
        else:
            labels_rf = labels_r + result_grps[fake_type]['labels']
            predicts_rf = predicts_r + result_grps[fake_type]['predicts']

        video_acc, video_auc, pred_basic,precision,recall = produce_result(labels_rf, predicts_rf,result_root)
        result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, fake_type)
        results.append(result)
    return results

def final_scores(result_root='', result_file=''):
    if result_file == '':
        result_files = glob.glob(result_root+'*.csv')
    else:
        result_files = [result_file]
        result_root = result_file.replace(os.path.basename(result_file),'')
    scores = []
    for result_file in result_files:
        results = pd.read_csv(result_file)
        label = results['label'].values
        predict = results['predict'].values
        video = results['video'].values
        dataset_name = os.path.basename(result_file)[:-4]
        if dataset_name == 'FF-ALL':
            score = get_ff_result(video, predict, label, result_root)
            scores += score
        else:
            video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict,result_root)
            # result = "ACC=%.2f, AUC=%.2f, D=%s, %s" % (video_acc*100, video_auc*100, pred_basic, dataset_name)
            result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
            scores.append(result)
    with open(result_root+'scores.txt','a+') as file:
        for score in scores:
            file.write(score)
            file.write('\n')
    return video_auc, video_acc
            

def final_scores_withoutcsv(results, result_file=''):

    result_root = result_file.replace(os.path.basename(result_file),'')
    scores = []

    video = [item[0] for item in results]
    label = [item[1] for item in results]
    predict= [item[2] for item in results]
    
    dataset_name = os.path.basename(result_file)[:-4]
    if dataset_name == 'FF-ALL':
        score = get_ff_result(video, predict, label, result_root)
        scores += score
    else:
        video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict,result_root)
        # result = "ACC=%.2f, AUC=%.2f, D=%s, %s" % (video_acc*100, video_auc*100, pred_basic, dataset_name)
        result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
        scores.append(result)
    with open(result_root+'scores.txt','a+') as file:
        for score in scores:
            file.write(score)
            file.write('\n')
    return video_auc, video_acc

def final_scores_withoutcsv_v2(results, result_file=''):

    result_root = result_file.replace(os.path.basename(result_file),'')
    scores = []

    video = [item[0] for item in results]
    label = [item[1] for item in results]
    predict= [item[2] for item in results]
    predict2= [item[3] for item in results]
    predict3= [item[4] for item in results]
    
    
    dataset_name = os.path.basename(result_file)[:-4]
    if dataset_name == 'FF-ALL':
        score = get_ff_result(video, predict, label, result_root)
        scores += score
    else:
        video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict,result_root)
        result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s \n" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
        video_acc2, video_auc2, pred_basic2,precision2,recall2 = produce_result(label, predict2,result_root)
        result2 = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s \n" % (video_acc2*100, video_auc2*100, pred_basic2,precision2*100,recall2*100, dataset_name)
        video_acc3, video_auc3, pred_basic3,precision3,recall3 = produce_result(label, predict3,result_root)
        result3 = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s \n" % (video_acc3*100, video_auc3*100, pred_basic3,precision3*100,recall3*100, dataset_name)
        scores.append(result)
        scores.append(result2)
        scores.append(result3)
    with open(result_root+'scores.txt','a+') as file:
        for score in scores:
            file.write(score)
            # file.write('\n')
    return video_auc, video_acc,video_auc2, video_acc2,video_auc3, video_acc3
            
def final_scores2(result_root='', result_file=''):
    if result_file == '':
        result_files = glob.glob(result_root+'*.csv')
    else:
        result_files = [result_file]
        result_root = result_file.replace(os.path.basename(result_file),'')
    scores = []
    for result_file in result_files:
        results = pd.read_csv(result_file)
        label = results['label'].values
        predict1 = results['predict'].values
        predict2 = results['predict2'].values
        predict3 = results['predict3'].values
        predict = (predict1+predict2+predict3)/3

        # for i in range(len(label)):
        #     if predict1[i] > 0.5 and predict2[i] > 0.5 and predict3[i] < 0.5:
        #         predict[i] = (predict1[i] + predict2[i])/2.0
        #     elif predict1[i] > 0.5 and predict2[i] < 0.5 and predict3[i] < 0.5:
        #         predict[i] = (predict2[i] + predict3[i])/2.0
        #     elif predict1[i] > 0.5 and predict2[i] < 0.5 and predict3[i] > 0.5:
        #         predict[i] = (predict1[i] + predict3[i])/2.0
                
        #     elif predict1[i] < 0.5 and predict2[i] > 0.5 and predict3[i] > 0.5:
        #         predict[i] = (predict2[i] + predict3[i])/2.0                       
        #     elif predict1[i] < 0.5 and predict2[i] > 0.5 and predict3[i] < 0.5:
        #         predict[i] = (predict1[i] + predict3[i])/2.0
        #     elif predict1[i] < 0.5 and predict2[i] < 0.5 and predict3[i] > 0.5:
        #         predict[i] = (predict1[i] + predict2[i])/2.0                             
        #     else:
        #         predict[i] = (predict1[i] + predict2[i] + predict3[i])/3.0

        
        video = results['video'].values
        dataset_name = os.path.basename(result_file)[:-4]
        if dataset_name == 'FF-ALL':
            score = get_ff_result(video, predict, label, result_root)
            scores += score
        else:
            video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict,result_root)
            # result = "ACC=%.2f, AUC=%.2f, D=%s, %s" % (video_acc*100, video_auc*100, pred_basic, dataset_name)
            result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
            scores.append(result)
            
        if dataset_name == 'FF-ALL':
            score = get_ff_result(video, predict1, label, result_root)
            scores += score
        else:
            video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict1,result_root)
            # result = "ACC=%.2f, AUC=%.2f, D=%s, %s" % (video_acc*100, video_auc*100, pred_basic, dataset_name)
            result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
            scores.append(result)
            
        if dataset_name == 'FF-ALL':
            score = get_ff_result(video, predict2, label, result_root)
            scores += score
        else:
            video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict2,result_root)
            # result = "ACC=%.2f, AUC=%.2f, D=%s, %s" % (video_acc*100, video_auc*100, pred_basic, dataset_name)
            result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
            scores.append(result)
            
        if dataset_name == 'FF-ALL':
            score = get_ff_result(video, predict3, label, result_root)
            scores += score
        else:
            video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict3,result_root)
            # result = "ACC=%.2f, AUC=%.2f, D=%s, %s" % (video_acc*100, video_auc*100, pred_basic, dataset_name)
            result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
            scores.append(result)            
            
            
            
    with open(result_root+'scores.txt','a+') as file:
        for score in scores:
            file.write(score)
            file.write('\n')            



def final_scores2head(result_root='', result_file=''):
    if result_file == '':
        result_files = glob.glob(result_root+'*.csv')
    else:
        result_files = [result_file]
        result_root = result_file.replace(os.path.basename(result_file),'')
    scores = []
    for result_file in result_files:
        results = pd.read_csv(result_file)
        label = results['label'].values
        predict1 = results['predict'].values
        predict2 = results['predict2'].values
        predict = (predict1+predict2)/2
 
        video = results['video'].values
        dataset_name = os.path.basename(result_file)[:-4]
        if dataset_name == 'FF-ALL':
            result = get_ff_result(video, predict, label, result_root)
        else:
            video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict,result_root)
            # result = "ACC=%.2f, AUC=%.2f, D=%s, %s" % (video_acc*100, video_auc*100, pred_basic, dataset_name)
            result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
        with open(result_root+'scores.txt','a+') as file:
            file.write('avg:  ')
            file.write(result)
            file.write('\n')                
            
        if dataset_name == 'FF-ALL':
            result = get_ff_result(video, predict, label, result_root)
        else:
            video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict1,result_root)
            # result = "ACC=%.2f, AUC=%.2f, D=%s, %s" % (video_acc*100, video_auc*100, pred_basic, dataset_name)
            result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
        with open(result_root+'scores.txt','a+') as file:
            file.write('branch1:  ')
            file.write(result)
            file.write('\n')  
        
        if dataset_name == 'FF-ALL':
                result = get_ff_result(video, predict, label, result_root)
        else:
            video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict2,result_root)
            # result = "ACC=%.2f, AUC=%.2f, D=%s, %s" % (video_acc*100, video_auc*100, pred_basic, dataset_name)
            result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, dataset_name)
        with open(result_root+'scores.txt','a+') as file:
            file.write('branch2:  ')
            file.write(result)
            file.write('\n')          


def final_scores_video(results, result_file=''):
    result_root = result_file.replace(os.path.basename(result_file),'')
    scores = []
    video = [item[0] for item in results]
    label = [item[1] for item in results]
    predict= [item[2] for item in results]
    pred = [item[3] for item in results]
    dataset_name = os.path.basename(result_file)[:-4]

    real_video = [row for row in results if row[1] == 0]
    fake_video = [row for row in results if row[1] == 1]

    real_samepred = sum([float(row[-1]) for row in real_video])/(256*len(real_video))
    fake_samepred = sum([float(row[-1]) for row in fake_video])/(256*len(fake_video))

    video_acc, video_auc, pred_basic,precision,recall = produce_result(label, predict,result_root)
    result = f"ACC=%.2f, AUC=%.2f, D=%s, P=%.2f, R=%.2f, R_SP=%.2f, F_SP=%.2f, %s" % (video_acc*100, video_auc*100, pred_basic,precision*100,recall*100, (1-real_samepred)*100, fake_samepred*100, dataset_name)
    scores.append(result)

    with open(result_root+'scores.txt','a+') as file:
        for score in scores:
            file.write(score)
            file.write('\n')
    return video_auc, video_acc



if __name__ == '__main__':
    opt = parse_args()
    if opt is None:
        exit()
    final_scores(opt.result_root)
    # y_true = np.array([0, 1])
    # y_scores = np.array([0.005416760221123695, 0.04772132262587547])
    # print(roc_auc_score(y_true, y_scores))
