import numpy as np
import cv2, torch, os
import matplotlib.pyplot as plt
from pylab import *
from PIL import Image,ImageFont,ImageDraw
import matplotlib.font_manager as fm # to create font
import json
import pandas as pd
import math
from glob import glob
import random
seed_value = 1234 
random.seed(seed_value)
REAL_LABLE = 0
FAKE_LABEL = 1
dataset_root = {
    'FaceForensics': 'FF++/',
    'Celeb-DF': 'Celeb-DF/',
    'DFDC': 'DFDC/',
    'df40-fomm': 'DF40/fomm/ff_processed',
    'df40-mobileswap': 'DF40/mobileswap/ff_processed',
    'df40-sadtalker': 'DF40/sadtalker/ff_processed',
    'df40-tpsm': 'DF40/tpsm/ff_processed',
    'df40-real': '',

    'df-40-v2-facedancer':'DF40/facedancer',
    'df-40-v2-fsgan':'DF40/fsgan',
    'df-40-v2-uniface':'DF40/uniface',
    'df-40-v2-simswap':'DF40/simswap',
    'df-40-v2-inswap':'DF40/inswap',

    'df-40-v2-danet':'DF40/danet',
    'df-40-v2-facevid2vid':'DF40/facevid2vid',
    'df-40-v2-mcnet':'DF40/mcnet',
    'df-40-v2-lia':'DF40/lia',
    'df-40-v2-hyperreenact':'DF40/hyperreenact',

    'FaceForensics_c23_train': 'FF++/',
    'Fsh': 'FF++/',

    'FFIW':'/home/guozonghui/data1/project/01-FFD/datasets/ffd-video-data-at/test/FFIW',
    
}
def get_subfolders(path):
    subfolders = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subfolder_path = os.path.join(root, dir)
            subfolders.append(subfolder_path)
    return subfolders

def get_data_list(dataset_name, base_root, split, only_real=False):
    dataset_info = []
    if "FaceForensics" in dataset_name:
        compress = dataset_name.split("_")[1]
        if split == 'train':
            root = os.path.join(base_root, dataset_root[dataset_name+"_train"])
        else:
            split = 'test'
            root = os.path.join(base_root, dataset_root[dataset_name.split("_")[0]])
        dataset_info = get_FF_list(root, split, compress=compress, only_real=only_real)
    elif dataset_name == 'Fsh':
        split = 'test'
        compress = 'c23'
        root = os.path.join(base_root, dataset_root[dataset_name.split("_")[0]])
        dataset_info = get_FF_list_Fs(root, split, compress=compress, only_real=only_real)


    elif dataset_name == 'Celeb-DF':
        root = os.path.join(base_root, dataset_root[dataset_name])
        video_list_txt = os.path.join(root, 'List_of_testing_videos.txt')
        with open(video_list_txt) as f:
            for data in f:
                line=data.split()
                dataset_info.append((line[1][:-4],FAKE_LABEL-int(line[0])))
    elif dataset_name == 'DFDC' and split == 'test' :
        root = os.path.join(base_root, dataset_root[dataset_name])
        label=pd.read_csv(root+'labels.csv',delimiter=',')
        dataset_info = [(video_name[:-4], label) for video_name, label in zip(label['filename'].tolist(), label['label'].tolist())]
        root = root+'test_videos/'
    elif dataset_name == 'DFDC' and split=='val':
        root = os.path.join(base_root, dataset_root[dataset_name])
        label=pd.read_csv(root+'labels.csv',delimiter=',')
        dataset_info = [(video_name[:-4], label) for video_name, label in zip(label['filename'].tolist(), label['label'].tolist())]
        # dataset_info = random.Random().sample(dataset_info, 500)
        dataset_info = dataset_info[:500]
        root = root+'test_videos/'    
    elif dataset_name == 'FFIW' and split=='test':
        root = ''
        real_root = os.path.join(dataset_root[dataset_name],'source/frame')
        fake_root = os.path.join(dataset_root[dataset_name],'target/frame')
        real_path = glob(real_root + '/*', recursive=True)
        fake_path = glob(fake_root + '/*', recursive=True)
        for i,path in enumerate(real_path):
            dataset_info.append((path, REAL_LABLE))
        for i,path in enumerate(fake_path):
            dataset_info.append((path, FAKE_LABEL))
    elif "df40" in dataset_name :
        split = 'test'
        root = os.path.join(base_root, dataset_root['FaceForensics'])
        dataset_info = get_FF_list(root, split, compress='c23', only_real=True)
        print(len(dataset_info))
        fake_root = os.path.join(base_root, dataset_root[dataset_name])
        fake_path = glob(fake_root + '/*', recursive=True)
        for i,path in enumerate(fake_path):
            dataset_info.append((path.replace(base_root,''), FAKE_LABEL))

    elif "DFDCP" in dataset_name :
        root = ''
        dataset_info = []
        fake_root = '/data4-16T/liuyingjie/datasets/DFDCP/frames/fake'
        fake_path = glob(fake_root + '/*', recursive=True)
        for i,path in enumerate(fake_path):
            dataset_info.append((path.replace(base_root,''), FAKE_LABEL))

        real_root = '/data4-16T/liuyingjie/datasets/DFDCP/frames/real'
        real_path = glob(real_root + '/*', recursive=True)
        for i,path in enumerate(real_path):
            dataset_info.append((path.replace(base_root,''), REAL_LABLE))

    elif "UAVDF" in dataset_name :
        root = ''
        dataset_info = []
        fake_root = '/home/guozonghui/data1/project/01-FFD/datasets/UAVDF/frame/fake'
        fake_path = glob(fake_root + '/*', recursive=True)
        for i,path in enumerate(fake_path):
            dataset_info.append((path.replace(base_root,''), FAKE_LABEL))

        real_root = '/home/guozonghui/data1/project/01-FFD/datasets/UAVDF/frame/real'
        real_path = glob(real_root + '/*', recursive=True)
        for i,path in enumerate(real_path):
            dataset_info.append((path.replace(base_root,''), REAL_LABLE))


    elif "WDF" in dataset_name :
        root = ''

        real_list = []
        fake_list = []
        real_path = '/data4-16T/liuyingjie/datasets/deepfake_in_the_wild/frame/real_test'
        fake_path = '/data4-16T/liuyingjie/datasets/deepfake_in_the_wild/frame/fake_test'
        result = get_subfolders(real_path)+get_subfolders(fake_path)
        for folder in result:
            file_list = []
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isfile(item_path):
                    file_list.append(item_path)
            if len(file_list) > 0 and 'real' in folder:
                real_list.append(folder)
            elif len(file_list) > 0 and 'fake' in folder:
                fake_list.append(folder)

        for i,path in enumerate(real_list):
            dataset_info.append((real_list[i], REAL_LABLE))
        for i,path in enumerate(fake_list):
            dataset_info.append((fake_list[i], FAKE_LABEL))

    elif "df-40-v2" in dataset_name:
        split = 'test'
        root = os.path.join(base_root, dataset_root['FaceForensics'])
        dataset_info = get_FF_list(root, split, compress='c23', only_real=True)
        fake_root = os.path.join(base_root, dataset_root[dataset_name])
        fake_path = glob(fake_root + '/frame/*', recursive=True)
        for i,path in enumerate(fake_path):
            dataset_info.append((path.replace(base_root,''), FAKE_LABEL))

    else:
        print('not support!', dataset_name)
        assert 0
    return dataset_info, root


def get_FF_list(root, split, compress='c23', only_real=False):
    print(root)
    split_json_path = os.path.join(root, 'splits', f'{split}.json')
    json_data = json.load(open(split_json_path, 'r'))
    if only_real:
        real_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
        real_video_dir = os.path.join('original_sequences', 'youtube', compress, 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), REAL_LABLE] for x in real_names]
    else:
        real_names = []
        fake_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
            fake_names.extend([f'{item[0]}_{item[1]}', f'{item[1]}_{item[0]}'])
        real_video_dir = os.path.join('original_sequences', 'youtube', compress, 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), 0] for x in real_names]
        ff_fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        for method in ff_fake_types:
            fake_video_dir = os.path.join('manipulated_sequences', method, compress, 'videos')
            for x in fake_names:
                dataset_info.append((os.path.join(fake_video_dir,x),FAKE_LABEL))
    return dataset_info



def get_FF_list_Fs(root, split, compress='c23', only_real=False):
    print(root)
    split_json_path = os.path.join(root, 'splits', f'{split}.json')
    json_data = json.load(open(split_json_path, 'r'))
    if only_real:
        real_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
        real_video_dir = os.path.join('original_sequences', 'youtube', compress, 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), REAL_LABLE] for x in real_names]
    else:
        real_names = []
        fake_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
            fake_names.extend([f'{item[0]}_{item[1]}', f'{item[1]}_{item[0]}'])
        real_video_dir = os.path.join('original_sequences', 'youtube', compress, 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), 0] for x in real_names]
        ff_fake_types = ['Fsh']
        fake_video_dir = '/data4-16T/liuyingjie/datasets/Fsh/test/frames'
        for x in fake_names:
            dataset_info.append((os.path.join(fake_video_dir,x),FAKE_LABEL))
    return dataset_info


def check_frame_len(video_len, num_segments):
    inner_index = list(range(video_len))
    pad_length = math.ceil((num_segments-video_len)/2)
    post_module = inner_index[1:-1][::-1] + inner_index
    l_post = len(post_module)
    post_module = post_module * (pad_length // l_post + 1)
    post_module = post_module[:pad_length]
    assert len(post_module) == pad_length
    pre_module = inner_index + inner_index[1:-1][::-1]
    l_pre = len(post_module)
    pre_module = pre_module * (pad_length // l_pre + 1)
    pre_module = pre_module[-pad_length:]
    assert len(pre_module) == pad_length

    sampled_clip_idxs = pre_module + inner_index + post_module
    sampled_clip_idxs = sampled_clip_idxs[:num_segments]
    return sampled_clip_idxs

if __name__=='__main__':
    # FaceForensics_c23 '../../datasets_processed/FF++/videos'
    data_list = get_data_list('FaceForensics_c23', '../../datasets_processed/', 'train')
    # Celeb-DF 
    # data_list = get_data_list('Celeb-DF', '../../datasets_processed/', 'test')
    # data_list = get_data_list('DFDC', '../../datasets_processed/', 'test')
    
    print(len(data_list))