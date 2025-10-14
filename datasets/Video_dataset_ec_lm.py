import os
import numpy as np
import cv2
import json
import torch
import torch.utils.data as data
import pickle
import random
from collections import OrderedDict
import pandas as pd
from decord import VideoReader, cpu
from common.utils import map_util
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import math
from datasets.utils import dataloader_util
from datasets.utils.mask_generator import MaskingGenerator
from tqdm import tqdm
class Video_dataset_ec_withlm(data.Dataset):
    def __init__(self,
                 root,
                 file_list=None,
                 method='Deepfakes',
                 compression='c23',
                 split='train',
                 num_segments=4,
                 clip_num=16,
                 transform=None,
                 transform2=None,
                 sparse_span=150,
                 dense_sample=0,
                 test_margin=1.3,
                 is_aligned=False,
                 cutout=False,
                 is_sbi=False,
                 align_type='cm',
                 image_size=224,
                 is_mask=False):
        super().__init__()
        self.root = root
        self.dataset_info = []
        self.file_list = file_list
        self.method = method
        self.split = split
        self.num_segments = num_segments
        self.clip_num = clip_num
        self.transform = transform
        self.sparse_span = sparse_span
        self.dense_sample = dense_sample
        self.test_margin = test_margin
        self.is_aligned = is_aligned
        self.align_type = align_type
        self.image_size = image_size
        self.is_cutout = cutout
        self.is_sbi = is_sbi
        self.is_mask = is_mask
        self.parse_dataset_info()
        self.crop_align_func = FasterCropAlignXRay(image_size)
        self.cutout = map_util.Cutout()
        self.masked_position_generator = MaskingGenerator(
        14, num_masking_patches=50,
        max_num_patches=None,
        min_num_patches=16,
        )
        self.frame_list = {}
        self.lm5 = self.get_five(np.load('datasets/defalut.npy')[0])

        
    def get_sampled_idx(self, file_path, video_label):
        if 'df40' in self.method and video_label ==1:
            frame_path = file_path
        elif 'df-40' in self.method and video_label ==1:
            frame_path = file_path
        elif 'Fsh' in self.method and video_label ==1:
            frame_path = file_path
        elif 'DFDCP' in self.method:
            frame_path = file_path    
        elif 'UAVDF' in self.method:
            frame_path = file_path      
        elif 'WDF' in self.method:
            frame_path = file_path   
        elif not 'FFIW' in file_path:
            frame_path = os.path.join(file_path, 'frame')
        else:
            frame_path = file_path
        file_ext = '.png'
        file_names = [f for f in os.listdir(frame_path) if f.endswith(file_ext)]
        all_frame_idxs = np.array([int(f[:-len(file_ext)]) for f in file_names])
        
        if len(all_frame_idxs) < self.num_segments:
            num_repeats = self.num_segments // len(all_frame_idxs) + 1
            sampled_frame_idxs = np.tile(all_frame_idxs, num_repeats)[:self.num_segments]
            sampled_frame_idxs = []
            idxs = dataloader_util.check_frame_len(len(all_frame_idxs), self.num_segments)
            all_frame_idxs = all_frame_idxs[idxs]
        all_frame_idxs.sort()
        sampled_frame_idxs = all_frame_idxs
        return sampled_frame_idxs
    
    def parse_dataset_info(self):
        dataset_list, data_root = dataloader_util.get_data_list(self.method, self.root, self.split, only_real=False)
        print('video_len:', len(dataset_list))
        self.all_list = []
        error = 0
        for i, file_info in enumerate(tqdm(dataset_list, desc="parse_dataset_info", unit="file")):    
            file_path, video_label = file_info[0], file_info[1]
            if self.method == 'Fsh':
                if video_label == 0:
                    file_path = data_root + file_path
                else:
                    file_path = file_path
            else:
                file_path = data_root + file_path
            # print(file_path, video_label)

            if 'df40' in self.method and video_label ==1: file_path= file_path.replace('FF++/','')
            if 'df-40' in self.method and video_label ==1: file_path= file_path.replace('FF++/','')
            if os.path.isdir(os.path.join(file_path, 'frame')) or ('FFIW' in file_path and os.path.isdir(file_path)) or ('DF40' in file_path and os.path.isdir(file_path))or ('Fsh' in file_path and os.path.isdir(file_path))or ('DFDCP' in file_path and os.path.isdir(file_path))or ('UAVDF' in file_path and os.path.isdir(file_path))or ('in_the_wild' in file_path and os.path.isdir(file_path)):
                sampled_frame_idx = self.get_sampled_idx(file_path, video_label)
                i_list = []
                for j,frame_idx in enumerate(sampled_frame_idx):
                    if 'df40' in self.method and video_label ==1:
                        filename_frame = os.path.join(file_path, f"{frame_idx}.png").replace('FF/','')
                    elif 'df-40' in self.method and video_label ==1:
                        filename_frame = os.path.join(file_path, f"{frame_idx}.png").replace('FF/','')
                    elif 'FFIW' in file_path:
                        filename_frame = os.path.join(file_path, f"{frame_idx}.png")
                    elif 'Fsh' in file_path and video_label ==1:
                        filename_frame = os.path.join(file_path, f"{frame_idx}.png") 
                    elif 'DFDCP' in file_path:
                        filename_frame = os.path.join(file_path, f"{frame_idx}.png")                    
                    elif 'UAVDF' in file_path:
                        filename_frame = os.path.join(file_path, f"{frame_idx}.png")  
                    elif 'in_the_wild' in file_path:
                        filename_frame = os.path.join(file_path, f"{frame_idx}.png")                   
                    else:
                        filename_frame = os.path.join(file_path, 'frame', f"{frame_idx}.png") 
                    video_labels = torch.tensor(video_label)
                    i_list.append((filename_frame, video_labels))       
                self.all_list.append(i_list) 
            else:
                print('error',file_path)
                error = error+1
                # assert 0
        print('success,video:', len(self.all_list))
        print('error', str(error))
        random.shuffle(self.all_list) 
    def get_five(self, ldm81):
        groups = [list(range(36, 42)), list(range(42, 48)), [30], [48], [54]]
        points = []
        for group in groups:
            group_landmarks = [ldm81[i] for i in group]
            mean_point = np.mean(group_landmarks, axis=0)
            points.append(mean_point)
        return np.array(points)  
    def get_lm(self, file_info, i):
        if 'df40' in self.method and file_info[0][1]==1:
            landmark_name = file_info[i][0].replace('png', 'npy')
            directory, filename = os.path.split(landmark_name)
            landmark_name = os.path.join(directory, 'landmarks', filename)
        else:
            landmark_name = file_info[i][0].replace('frame', 'landmarks').replace('png', 'npy')
            # print(landmark_name)
        
        if os.path.exists(landmark_name):
            ld_81 = np.load(landmark_name)[0]
            ld_5 = self.get_five(ld_81)
            # print('correct')
        else:
            # print('error')
            if i-1 >=0:
                ld_5 = self.get_lm(file_info, i-1)
            else:
                ld_5=self.lm5
        
        return torch.tensor(ld_5, dtype=torch.float32)
           
    def __getitem__(self, index):
        flag = True
        while flag:
            try: 
                file_info = self.all_list[index]
                video_labels = file_info[0][1]
                all_frames_list=[]
                all_landmark_list = []
                process_imgs_list = []
                # if self.split=='train':
                #     start_index = np.random.randint(0, len(file_info)- self.num_segments) if len(file_info)!= self.num_segments else 0
                # else:
                clip_num = self.clip_num
                start_index = 0
                interval = (len(file_info) - self.num_segments) // (clip_num-1)
                id_list = []
                all_lm5 = []
                for j in range(clip_num):
                    all_frames = []
                    all_landmark = []
                    for i in range(start_index+j*interval, start_index+j*interval+self.num_segments):
                        filename_frame = file_info[i][0]
                        frame = Image.open(filename_frame)
                        landmark5 = self.get_lm(file_info, i)
                        id_list.append(filename_frame)
                        frame = np.asarray(frame)
                        all_frames.append(frame)
                        all_landmark.append(landmark5)
                        self.lm5=landmark5
                    all_landmark = torch.stack(all_landmark)
                    all_frames_list.append(all_frames)
                    all_lm5.append(all_landmark)
                all_lm5 = torch.stack(all_lm5)
                if self.transform is not None:
                    for j in range(clip_num):
                        all_frames = all_frames_list[j]
                        additional_targets = {}
                        tmp_imgs = {"image": all_frames[0]}
                        for i in range(1, len(all_frames)):
                            additional_targets[f"image{i}"] = "image"
                            tmp_imgs[f"image{i}"] = all_frames[i]
                        self.transform.add_targets(additional_targets)
                        all_frames = self.transform(**tmp_imgs)
                        all_frames = OrderedDict(sorted(all_frames.items(), key=lambda x: x[0]))
                        all_frames = list(all_frames.values())
                        if self.is_cutout:
                            all_frames = torch.stack(all_frames)
                            process_imgs = self.cutout(all_frames).squeeze()
                        else:
                            process_imgs = torch.stack(all_frames)
                        process_imgs_list.append(process_imgs)
                    process_imgs_list = torch.stack(process_imgs_list)
                    flag=False
            except Exception as e:
                print(e)
                index=torch.randint(low=0,high=self.__len__(),size=(1,)).item()
                
        return {"images":process_imgs_list, "labels":video_labels, "video_path":filename_frame, 'landmark':all_lm5, 'index':index, 'id_list':all_frames_list}    
    
    def __len__(self):
        return len(self.all_list)



    
if __name__=='__main__':
    from utils import face_utils
    from utils import map_util
    from lib.ct.utils import get_crop_box
    from lib.ct.faster_crop_align_xray import FasterCropAlignXRay
    import albumentations as alb
    from albumentations.pytorch.transforms import ToTensorV2
    # from ..common.data import base_transform
    seed=1234
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_segments = 8
    additional_targets = {}
    for i in range(1, num_segments):
        additional_targets[f'image{i}'] = 'image'
    base_transform = alb.Compose([
            alb.Resize(224, 224),
            alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], additional_targets=additional_targets)

    image_dataset=Video_dataset_ec_withlm('../../../datasets/rawdata/',
                 '../../../datasets/rawdata/dataset_list_v1/',
                 method='ALL',
                 compression='c23',
                 split='train',
                 num_segments=8,
                 transform=base_transform,
                 sparse_span=150,
                 dense_sample=0,
                 test_margin=1.3,
                 is_aligned=True,
                 align_type='cm',
                 cutout=False,
                 is_sbi=False,)
    batch_size=1
    dataloader = torch.utils.data.DataLoader(image_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    # collate_fn=image_dataset.collate_fn,
                    num_workers=1,
                    # worker_init_fn=image_dataset.worker_init_fn
                    )
    data_iter=iter(dataloader)
    data=next(data_iter)
    for idx, datas in enumerate(dataloader):
        print(datas['images'].shape)
else:
    from common.utils import map_util
    from common.utils import face_utils
    from datasets.lib.ct.utils import get_crop_box
    from datasets.lib.ct.faster_crop_align_xray import FasterCropAlignXRay        

