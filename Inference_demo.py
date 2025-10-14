import os
import cv2
import numpy as np
from utils.real_face_process_v2 import process
from utils import Landmark_extract
import dlib
import albumentations as alb
import torch
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from models.TFCU import TFCU_Model
import csv
from glob import glob
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_nums = np.arange(total_frames)
    print(len(frame_nums))
    frames = []
    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        frames.append(frame)
        if ret:
            output_path = video_path.replace('.mp4',f'/{frame_num}.png').replace('videos','frames')
            os.makedirs(video_path.replace('.mp4','/').replace('videos','frames'), exist_ok=True)
            cv2.imwrite(output_path, frame)
    cap.release()
    return frames

def get_sampled_idx(file_path):
    file_ext = '.png'
    file_names = [f for f in os.listdir(file_path) if f.endswith(file_ext)]
    all_frame_idxs = np.array([int(f[:-len(file_ext)]) for f in file_names])
    all_frame_idxs.sort()
    sampled_frame_idxs = all_frame_idxs
    return sampled_frame_idxs

def get_five(ldm81):
    groups = [list(range(36, 42)), list(range(42, 48)), [30], [48], [54]]
    points = []
    for group in groups:
        group_landmarks = [ldm81[i] for i in group]
        mean_point = np.mean(group_landmarks, axis=0)
        points.append(mean_point)
    return np.array(points)  
def get_lm(root_path, idx):
    landmark_name = os.path.join(root_path, str(idx))
    if idx<0:
        ld_81 = np.load('defalut.npy')[0]
        ld_5 = get_five(ld_81)
        return torch.tensor(ld_5, dtype=torch.float32).cuda()
    try:
        ld_81 = np.load(landmark_name+'.npy')[0]
        ld_5 = get_five(ld_81)
    except:
        ld_5 = get_lm(root_path, idx-1)
    return torch.tensor(ld_5, dtype=torch.float32).cuda()

def load_tensor(root_path, idxs, transfrom):
    video_tensor = []
    landmark_tensor = []
    big_boxes = []
    for i in range(len(idxs)//4):
        clip = []
        clip_lm = []
        for j in range(4):
            path = os.path.join(root_path, str(i*4+j))
            if (os.path.exists(path+'.png')):
                img = Image.open(path+'.png')
            else:
                path = os.path.join(root_path, str(0))
                img = Image.open(path+'.png')

            img = np.asarray(img)
            tmp_imgs = {"image": img}
            input_tensor = transfrom(**tmp_imgs)
            input_tensor = input_tensor['image'].cuda()
            lm5_tensor = get_lm(root_path, i*4+j)

            clip.append(input_tensor)
            clip_lm.append(lm5_tensor)
            clip_tensor = torch.stack(clip)
            clip_lm_tensor = torch.stack(clip_lm)
            # bbox = np.load()
        video_tensor.append(clip_tensor)
        landmark_tensor.append(clip_lm_tensor)
    video_tensor = torch.stack(video_tensor)
    landmark_tensor = torch.stack(landmark_tensor)
    print(video_tensor.shape, landmark_tensor.shape)
    return video_tensor, landmark_tensor


def get_npy(root_path,num):
    path = os.path.join(root_path, str(num))
    if (os.path.exists(path+'.npy')):
        box = np.load(path+'.npy')
    else:
        box = get_npy(root_path, int(num)-1)
    return box


def load_box(root_path, idxs):
    big_boxes = []
    for i in range(len(idxs)//4):
        for j in range(4):
            num = i*4+j
            box = get_npy(root_path, num)
            big_boxes.append(box)
    return big_boxes

def get_model():
    model = TFCU_Model()
    ckpt_load_path = 'checkpoints/Final_TFCU_Model/ckpt/Final.tar'
    checkpoint = torch.load(ckpt_load_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
    else:
        sd = checkpoint
    new_state_dict = {}    
    for k, v in sd.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
            new_state_dict[k] = v
    msg = model.load_state_dict(new_state_dict,strict=False)
    print('sdload', msg)
    return model

import sys
from utils.supply_writer import SupplyWriter
if __name__ == "__main__":
    # input_argument = sys.argv[1]

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'lib/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    additional_targets = {}
    base_transform = alb.Compose([
        alb.Resize(224, 224),
        alb.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets=additional_targets)

    model = get_model()
    model = model.cuda()
    model.eval()


    all_videos = ['demo/videos/cmmueaopze.mp4']
    print(all_videos)
    # assert 0
    for input_video in all_videos:
        with torch.no_grad():
            input_frames = input_video.replace('videos', 'frames').replace('.mp4', '')
            aligned_frames = input_frames.replace('frames', 'frames_align')
            frames = extract_frames(input_video)
            all_idx = get_sampled_idx(input_frames)

            process(input_frames)
            Landmark_extract.extract(aligned_frames,face_detector,face_predictor)
            video_tensor, landmark_tensor = load_tensor(aligned_frames, all_idx,base_transform)

            output = model(video_tensor.unsqueeze(0), landmark_tensor.unsqueeze(0), eval=True)
            print(output)
            outputs=torch.nn.functional.softmax(output, dim=2)[:,:,1]  # [B, N*T, 2]
            outputs_mean = torch.mean(outputs, dim=1)
            print(outputs)
            csv_file = 'inference_results.csv'
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([input_video, outputs_mean.cpu().detach().numpy()[0]])
            out_file = input_video.replace('videos', 'result')
            optimal_threshold = 0.5
            all_box = load_box(input_frames, all_idx)
            SupplyWriter(input_video, out_file, optimal_threshold, rgb_input=False).run(frames, outputs[0], all_box)