import os
import sys
from omegaconf import OmegaConf
from tqdm import tqdm
from shutil import copyfile
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from models import *
from datasets import *
from datasets.factory import get_final_dataloader
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..'))
from common.utils import map_util
from common import losses, optimizers
from common.utils import *
from common.utils.map_util import tensor2im,show_mask_on_image
from utils import *
import glob
import csv
from evaluations import video_evaluation
args = get_params()
setup(args)
###########################
# main logic for test #
###########################
def main():
    # use distributed test with nccl backend 
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl', init_method="env://")
    torch.cuda.set_device(args.local_rank)
    args.world_size = dist.get_world_size()
    # set model and wrap it with DistributedDataParallel
    model = eval(args.model.name)(**args.model.params)
    model.cuda(args.local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    # model.resume must be set here.
    result_dir = os.path.dirname(args.config)
    ckpts = sorted(glob.glob(result_dir+'/ckpt/Final*.tar'))
    i = 0
    for ckpt in ckpts:
        if i ==len(ckpts)-1:
            args.model.resume = ckpt
            ckpt_load_path = args.model.resume
            print(f'ckpt_load_path: {ckpt_load_path}')
            checkpoint = torch.load(ckpt_load_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                sd = checkpoint['state_dict']
            else:
                sd = checkpoint
            msg = model.load_state_dict(sd,strict=False)
            print('test_load', msg)
            header = ['Exp', 'Dataset', 'AUC_v', 'ACC_v', 'AUC_f', 'ACC_f']
            methods = [ 'Celeb-DF','DFDC']
            with open('checkpoints/result_all.csv', 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                if csvfile.tell() == 0:
                    csvwriter.writerow(header)
                for method in methods:
                    args.final_test.dataset.params.method = method
                    args.final_test.dataset.params.split = 'test'
                    test_dataloader = get_final_dataloader(args, 'test')
                    result = test(test_dataloader, model, args)
                    print(method, result)
        i +=1



def test(dataloader, model, args):
    model.eval()
    test_label = 'test'
    y_outputs, y_labels, y_idxes, y_pd = [], [], [], []
    y_outputs_f, y_labels_f, y_idxes_f, y_pd_f = [], [], [], []
    with torch.no_grad():
        for idx, datas in enumerate(tqdm(dataloader)):
            images = datas['images']
            labels = datas['labels']
            video_path = datas['video_path']
            idxes = datas['index']
            landmark5 = datas['landmark']
            images = images.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank)
            idxes = idxes.cuda(args.local_rank)
            output = model(images, eval=True, landmark = landmark5)
            if len(output.shape) > 2:
                outputs=torch.nn.functional.softmax(output, dim=2)[:,:,1]  # [B, N*T, 2]
                count = (outputs > 0.5).sum(dim=1)
                outputs = torch.mean(outputs, dim=1)
            y_pd.extend(count)
            y_outputs.extend(outputs)
            y_labels.extend(labels)
            y_idxes.extend(idxes)
            labels_f = labels.unsqueeze(1).repeat(1, output.size(1)).flatten(0,1)
            idxes_f = idxes.unsqueeze(1).repeat(1, output.size(1)).flatten(0,1)
            outputs_f=torch.nn.functional.softmax(output, dim=2)[:,:,1].flatten(0,1)  # [B, N*T, 2]
            count_f = (outputs_f > 0.5)

            y_pd_f.extend(count_f)
            y_outputs_f.extend(outputs_f)
            y_labels_f.extend(labels_f)
            y_idxes_f.extend(idxes_f)


    gather_y_outputs = gather_tensor(y_outputs, args.world_size, to_numpy=False)
    gather_y_labels  = gather_tensor(y_labels, args.world_size, to_numpy=False)
    gather_y_idxes  = gather_tensor(y_idxes, args.world_size, to_numpy=False)
    gather_y_pd  = gather_tensor(y_pd, args.world_size, to_numpy=False)


    gather_y_outputs_f = gather_tensor(y_outputs_f, args.world_size, to_numpy=False)
    gather_y_labels_f  = gather_tensor(y_labels_f, args.world_size, to_numpy=False)
    gather_y_idxes_f  = gather_tensor(y_idxes_f, args.world_size, to_numpy=False)
    gather_y_pd_f  = gather_tensor(y_pd_f, args.world_size, to_numpy=False)

    test_result_list = []
    for i, idx in enumerate(gather_y_idxes):
        video_name = dataloader.dataset.all_list[idx][0][0]
        video_name_tmp = video_name.split("/")
        video_name = video_name[2:].replace('/'+video_name_tmp[-2]+'/'+video_name_tmp[-1], "").replace('/'+video_name_tmp[1]+"/", '')
        video_label = gather_y_labels[i].cpu().item()
        video_predict = gather_y_outputs[i].cpu().item()
        frame_predict = gather_y_pd[i].cpu().item()
        if video_label >= 2:
            video_label = video_label-2
            video_predict = 0.5
        test_result_list.append([video_name, video_label, video_predict, frame_predict])
    test_result_list = sorted(test_result_list, key=(lambda x:x[0]))
    result_dir = args.exam_dir
    result_dir = os.path.join(result_dir, test_label)
    os.makedirs(result_dir, exist_ok=True)
    predict_file = result_dir+"/"+args.final_test.dataset.params.method+".csv"
    pd.DataFrame(test_result_list, columns=["video", "label", "predict", "pred"]).to_csv(predict_file, index=False)
    print('saved as ', predict_file)
    config_file = args.config
    config_file_name = os.path.basename(config_file)
    copyfile(config_file, os.path.join(result_dir, config_file_name))
    auc, acc = video_evaluation.final_scores_video(test_result_list, result_file=predict_file)


    test_result_list_f = []
    for i, idx in enumerate(gather_y_idxes_f):
        video_name = dataloader.dataset.all_list[idx][0][0]
        video_name_tmp = video_name.split("/")
        video_name = video_name[2:].replace('/'+video_name_tmp[-2]+'/'+video_name_tmp[-1], "").replace('/'+video_name_tmp[1]+"/", '')
        video_label = gather_y_labels_f[i].cpu().item()
        video_predict = gather_y_outputs_f[i].cpu().item()
        frame_predict = gather_y_pd_f[i].cpu().item()
        if video_label >= 2:
            video_label = video_label-2
            video_predict = 0.5
        test_result_list_f.append([video_name, video_label, video_predict, frame_predict])
    test_result_list_f = sorted(test_result_list_f, key=(lambda x:x[0]))
    result_dir = args.exam_dir
    result_dir = os.path.join(result_dir, test_label)
    os.makedirs(result_dir, exist_ok=True)
    predict_file = result_dir+"/"+args.final_test.dataset.params.method+"_frame.csv"
    pd.DataFrame(test_result_list_f, columns=["video", "label", "predict", "pred"]).to_csv(predict_file, index=False)
    print('saved as ', predict_file)
    config_file = args.config
    config_file_name = os.path.basename(config_file)
    copyfile(config_file, os.path.join(result_dir, config_file_name))
    aucf, accf = video_evaluation.final_scores_video(test_result_list_f, result_file=predict_file)

    return auc, acc, aucf, accf




if __name__ == "__main__":
    main()
