import numpy as np
from lib.ct.detection import FaceDetector
import cv2
from lib.utils import flatten,partition
from tqdm import tqdm
from imutils import face_utils
import os
import face_utils
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
detector = FaceDetector(0)

def split_array(array, segment_size):
    segmented_array = []
    for i in range(0, len(array), segment_size):
        segment = array[i:i+segment_size]
        segmented_array.append(segment)
    return segmented_array


def get_image_files(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_files.append(os.path.join(root, file))
    return image_files

def process(root):
    data_list = get_image_files(root)
    frames = []
    data_list = sorted(data_list)
    data_list_split = split_array(data_list, 1)
    image_size = 224
    aligned = []
    all_boxes = []
    for clip in tqdm(data_list_split):
        frames = []
        for frame_name in clip:
            print(frame_name)
            frame=cv2.imread(os.path.join(frame_name))
            frames.append(frame)
        if len(frames) > 0:
            detect_res = flatten(
                [detector.detect(item) for item in partition(frames, 1)]
            )
            detect_res = get_valid_faces(detect_res, thres=0.5)
            for faces, frame, frame_name in zip(detect_res, frames, clip):
                if len(faces) > 0:
                    bbox, lm5, score = faces[0]
                    np.save(frame_name.replace('png','npy'), bbox)
                    frame, landmark, bbox=face_utils.crop_aligned(frame,lm5,landmarks_68=None,bboxes=bbox,aligned_image_size=image_size)
                    bbox = np.array([[bbox[0],bbox[1]],[bbox[2],bbox[3]]])
                    frame_croped = crop_face_sbi(frame, bbox=bbox, margin=False)
                    frame_croped = cv2.resize(frame_croped, (224, 224))

                    frame_name = frame_name.replace('frames/', 'frames_align/')
                    directory_path = os.path.dirname(frame_name)

                    print(frame_name, os.path.dirname(frame_name))
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                    try:
                        cv2.imwrite(frame_name, frame_croped)
                        aligned.append(frame_croped)
                        np.save(frame_name.replace('png','npy'), bbox)

                    except Exception as e:
                        print("An error occurred:", str(e))
def get_valid_faces(detect_results, max_count=10, thres=0.5, at_least=False):
    new_results = []
    for i, faces in enumerate(detect_results):
        if len(faces) > max_count:
            faces = faces[:max_count]
        l = []
        for j, face in enumerate(faces):
            if face[-1] < thres and not (j == 0 and at_least):
                continue
            box, lm, score = face
            box = box.astype(np.float)
            lm = lm.astype(np.float)
            l.append((box, lm, score))
        new_results.append(l)
    return new_results


def crop_face_sbi(img,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
    assert phase in ['train','val','test']
    H,W=len(img),len(img[0])
    if crop_by_bbox:
        x0,y0=bbox[0]
        x1,y1=bbox[1]
        w=x1-x0
        h=y1-y0
        w0_margin=w/4#0#np.random.rand()*(w/8)
        w1_margin=w/4
        h0_margin=h/4#0#np.random.rand()*(h/5)
        h1_margin=h/4
    if margin:
        w0_margin*=4
        w1_margin*=4
        h0_margin*=2
        h1_margin*=2
    elif phase=='train':
        w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()	
    else:
        w0_margin*=0.5
        w1_margin*=0.5
        h0_margin*=0.5
        h1_margin*=0.5
            
    y0_new=max(0,int(y0-h0_margin))
    y1_new=min(H,int(y1+h1_margin)+1)
    x0_new=max(0,int(x0-w0_margin))
    x1_new=min(W,int(x1+w1_margin)+1)

    img_cropped=img[y0_new:y1_new,x0_new:x1_new]

    return img_cropped


