from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
import dlib
from imutils import face_utils


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


from PIL import Image
def extract(org_path,face_detector,face_predictor):
    frames = glob(org_path+'/*.png')
    for cnt_frame in range(len(frames)):
        filename_frame = frames[cnt_frame]
        frame = cv2.imread(filename_frame)
        faces = face_detector(frame, 0)
        if len(faces)==0:
            print('noface',filename_frame)
            continue
        face_s_max=-1
        landmarks=[]
        size_list=[]
        for face_idx in range(len(faces)):
            landmark = face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0,y0=landmark[:,0].min(),landmark[:,1].min()
            x1,y1=landmark[:,0].max(),landmark[:,1].max()
            face_s=(x1-x0)*(y1-y0)
            size_list.append(face_s)
            landmarks.append(landmark)
        landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
        landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]
        land_path = filename_frame.replace('.png', '')
        # print(land_path)
        np.save(land_path, landmarks)

def reorder_landmark(landmark):
    landmark_add=np.zeros((13,2))
    for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
        landmark_add[idx]=landmark[idx_l]
    landmark[68:]=landmark_add
    return landmark

