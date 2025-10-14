import numpy as np
import cv2, torch, os
import matplotlib.pyplot as plt
from pylab import *
from PIL import Image,ImageFont,ImageDraw
import matplotlib.font_manager as fm # to create font

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

def visualize_feature_map(save_dir, img_batch):
    # img_batch = img_batch.permute(0,2,3,1)
    feature_map = np.squeeze(img_batch, axis=0).cpu().numpy()
    print(feature_map.shape)
 
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
 
    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
    #     plt.subplot(row, col, i + 1)
    #     plt.imshow(feature_map_split)
    #     axis('off')
    #     # title('feature_map_{}'.format(i))
 
    # plt.savefig(save_dir+'/feature_map.png')
    # plt.show()
 
    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig(save_dir+"/feature_map_sum.png")

def tensor2im(input_image, mean=[0,0,0], std=[1,1,1], imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB

            image_numpy = np.tile(image_numpy, (3, 1, 1))

            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        else:
            # image_numpy = np.transpose(image_numpy, (1, 2, 0))
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * std + mean
        image_numpy = image_numpy*255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = np.clip(image_numpy, 0,255)
    return image_numpy.astype(imtype)


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def save_image(image_numpy, image_path, aspect_ratio=1.0):

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path,quality=75) #added by Mia (quality)


def save_frames_video(image, idx,save_dir):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    t = image.size(1)
    image = image.transpose(1,0)
    # font=ImageFont.truetype('arial.ttf',10)
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    result = Image.new('RGB', (t*one_image_w-padding_num*2, one_image_w+20),"white")
            # print(image.size())
    for i in range(t):
        image_pil = Image.fromarray(tensor2im(image[i])).resize([one_image,one_image], Image.BICUBIC)
        result.paste(image_pil, box=(i*one_image_w, 0))
        drawer=ImageDraw.Draw(result)
        # drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w),fill=(0,0,0),text=str(idx[i].item()),font=font)
    img_path = os.path.join(save_dir+'.jpg')
    result.save(img_path,  quality=75)

from PIL import Image,ImageDraw,ImageFont
import matplotlib.font_manager as fm # to create font

def save_sbi_train_images(video_path,imgs_r, imgs_f, masks,mean, std,sampled_idxs, save_dir=None, save_name=None, return_np=False):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    t = imgs_r.size(1)
    result = Image.new('RGB', (t*one_image_w-padding_num*2, one_image_w*2-padding_num+40),"white")
    for i in range(t):
        img_r = Image.fromarray(tensor2im(imgs_r[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        img_f = Image.fromarray(tensor2im(imgs_f[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        # mask = Image.fromarray(tensor2im(masks[:,i,:,:,:])).resize([one_image,one_image], Image.BICUBIC)
        
        result.paste(img_r, box=(i*one_image_w, 0))
        result.paste(img_f, box=(i*one_image_w, one_image_w))
        # result.paste(mask, box=(i*one_image_w, one_image_w*2))
        # drawer=ImageDraw.Draw(result)
        # drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w*2),fill=(0,0,0),text=str(sampled_idxs[0][i]),font=font)
    # video_name = video_path[0].split("/")[-1]
    video_name_arr = video_path[0].split("/")
    video_name = video_name_arr[4]+"_"+video_name_arr[-1]
    if return_np:
        return np.array(result)
    else:
        img_path = os.path.join(save_dir+'/'+save_name+'-'+video_name+'.jpg')
        result.save(img_path,  quality=75)

def save_sbi_train_images2(video_path,imgs_r, imgs_f, imgs_pr, imgs_pf, masks,mean, std,sampled_idxs, save_dir=None, save_name=None, return_np=False):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    t = imgs_r.size(1)
    result = Image.new('RGB', (t*one_image_w-padding_num*2, one_image_w*4-padding_num+40),"white")
    for i in range(t):
        img_r = Image.fromarray(tensor2im(imgs_r[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        img_f = Image.fromarray(tensor2im(imgs_f[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        img_pr = Image.fromarray(tensor2im(imgs_pr[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        img_pf = Image.fromarray(tensor2im(imgs_pf[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        # mask = Image.fromarray(tensor2im(masks[:,i,:,:,:])).resize([one_image,one_image], Image.BICUBIC)
        
        result.paste(img_r, box=(i*one_image_w, 0))
        result.paste(img_f, box=(i*one_image_w, one_image_w))
        result.paste(img_pr, box=(i*one_image_w, 2*one_image_w))
        result.paste(img_pf, box=(i*one_image_w, 3*one_image_w))
        
        # result.paste(mask, box=(i*one_image_w, one_image_w*2))
        # drawer=ImageDraw.Draw(result)
        # drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w*2),fill=(0,0,0),text=str(sampled_idxs[0][i]),font=font)
    # video_name = video_path[0].split("/")[-1]
    video_name_arr = video_path[0].split("/")
    video_name = video_name_arr[4]+"_"+video_name_arr[-1]
    if return_np:
        return np.array(result)
    else:
        img_path = os.path.join(save_dir+'/'+save_name+'-'+video_name+'.jpg')
        result.save(img_path,  quality=75)




def save_images_loader(video_path,imgs, labels, mean, std, save_dir=None, save_name=None, return_np=False):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    print(imgs.shape)
    t = 8
    imgs = imgs.view((-1,t,)+imgs.size()[1:])
    labels = labels.view((-1,t,)+labels.size()[1:])
    result = Image.new('RGB', (t*one_image_w-padding_num*2, one_image_w-padding_num+40),"white")
    for i in range(t):
        img_r = Image.fromarray(tensor2im(imgs[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        result.paste(img_r, box=(i*one_image_w, 0))
        drawer=ImageDraw.Draw(result)
        drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w),fill=(0,0,0),text=str(labels[0][i].item()),font=font)
        # drawer.text(xy=(0,one_image_w),fill=(255,0,0),text=str(labels[0].item()),font=font)
    # video_name = video_path[0].split("/")[-1][:-4]
    
    if return_np:
        return np.array(result)
    else:
        img_path = os.path.join(save_dir+'/'+save_name+'.jpg')
        result.save(img_path,  quality=75)

def save_real_data_images(video_path,imgs, labels, mean, std,sampled_idxs, save_dir=None, save_name=None, return_np=False):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    t = imgs.size(1)
    result = Image.new('RGB', (t*one_image_w-padding_num*2, one_image_w-padding_num+40),"white")
    for i in range(t):
        img_r = Image.fromarray(tensor2im(imgs[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
        result.paste(img_r, box=(i*one_image_w, 0))
        drawer=ImageDraw.Draw(result)
        # drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w),fill=(0,0,0),text=str(sampled_idxs[0][i].item()),font=font)
        # print('dim:', sampled_idxs.dim())
        # drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w),fill=(0,0,0),text=str(sampled_idxs[0][i]),font=font)
        
        # if sampled_idxs.dim() == 1:
        #     drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w),fill=(0,0,0),text=str(sampled_idxs[i]),font=font)
        # else:
        #     drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w),fill=(0,0,0),text=str(sampled_idxs[0][i]),font=font)
        drawer.text(xy=(0,one_image_w),fill=(255,0,0),text=str(labels[0].item()),font=font)
    # video_name = video_path[0].split("/")[-1][:-4]
    
    video_name_arr = video_path[0].split("/")
    video_name = video_name_arr[4]+"_"+video_name_arr[-1]
    
    if return_np:
        return np.array(result)
    else:
        img_path = os.path.join(save_dir+'/'+video_name+'-'+save_name+'.jpg')
        result.save(img_path,  quality=75)


def save_real_data_images_withrebuild(video_path,outputreal, outputfake, imgs, imgs_f, rebuild_r, rebuild_f, labels, mean, std, save_dir=None, save_name=None, return_np=False):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    t_r = imgs.size(1)
    result = Image.new('RGB', (t_r*one_image_w-padding_num*2, 4*one_image_w-padding_num+40),"white")
    # mean = 0
    # std = 1
    if imgs.size(0) !=0:
        for i in range(t_r):
            img_r = Image.fromarray(tensor2im(imgs[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_r, box=(i*one_image_w, 0))
            drawer=ImageDraw.Draw(result)
            img_rp = Image.fromarray(tensor2im(rebuild_r[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_rp, box=(i*one_image_w, one_image_w))
            drawer=ImageDraw.Draw(result)
    t_f = imgs_f.size(1)
    if imgs_f.size(0) !=0:   
        for i in range(t_f):
            # print(imgs_f.shape)
            img_f = Image.fromarray(tensor2im(imgs_f[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_f, box=(i*one_image_w, one_image_w*2))
            drawer=ImageDraw.Draw(result)
            img_fp = Image.fromarray(tensor2im(rebuild_f[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_fp, box=(i*one_image_w, one_image_w*3))
            drawer=ImageDraw.Draw(result)         
           
            
    video_name_arr = video_path[0].split("/")
    video_name = video_name_arr[4]+"_"+video_name_arr[-1]
    
    if return_np:
        return np.array(result)
    else:
        img_path = os.path.join(save_dir+'/'+save_name+'-'+video_name+'_'+'.jpg')
        result.save(img_path,  quality=75)

import gc
from matplotlib.patches import Rectangle
import torchvision.transforms as transforms
def save_attn_img(input_fake, attn_fake, save_dir, save_name):
    n=input_fake.size(1)
    merged_image = torch.zeros(3, 224, 224 * n)
    mean=[0.485, 0.456, 0.406]
    std =[0.229, 0.224, 0.225]
    for i in range(n):
        frame = input_fake[0][i]
        frame = frame.cpu().permute(1, 2, 0).numpy()
        frame = std * frame + mean
        frame = np.clip(frame, 0, 1)
        fig, ax = plt.subplots()
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])
        for j in range(14):
            for k in range(14):
                x = k * 16
                y = j * 16
                token_index = j * 14 + k
                if token_index in attn_fake[0]:
                    rect = Rectangle((x, y), 16, 16, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
    img_path = os.path.join(save_dir+'/'+save_name+'_'+'.jpg')
    plt.savefig(img_path)
    plt.clf()
    plt.close()
    del  frame, fig, ax, rect
    gc.collect()
    return merged_image

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms


def save_maerebuild(savedir, rebuild, savename):
    img, bool_masked_pos, outputs = rebuild
    patch_size = (16,16)
    tmp = np.arange(0,8, 1) 
    frame_id_list = tmp.tolist()
    img = img.to(img.device, non_blocking=True)
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(img.device)[None, :, None, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(img.device)[None, :, None, None, None]
    ori_img = img * std + mean  # in [0, 1]
    imgs = [ToPILImage()(ori_img[0,:,vid,:,:].cpu()) for vid, _ in enumerate(frame_id_list)  ]
    img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size[0], p2=patch_size[0])
    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    img_patch[bool_masked_pos] = outputs.flatten(0,1)
    #make mask
    mask = torch.ones_like(img_patch)
    mask[bool_masked_pos] = 0
    mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
    mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
    #save reconstruction video
    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
    # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
    rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
    rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=patch_size[0], p2=patch_size[1], h=14, w=14)
    imgs2 = [ ToPILImage()(rec_img[0, :, vid, :, :].cpu().clamp(0,0.996)) for vid, _ in enumerate(frame_id_list)  ]
    for id, im in enumerate(imgs2):
        im.save(f"test/rec_img{id}.jpg")
    #save masked video 
    img_mask = rec_img * mask
    imgs3 = [ToPILImage()(img_mask[0, :, vid, :, :].cpu()) for vid, _ in enumerate(frame_id_list)]
    combined_image = Image.new('RGB', (3 * 224, len(frame_id_list) * 224))
    for i in range(len(frame_id_list)):
        combined_image.paste(imgs[i], (0, i * 224))
        combined_image.paste(imgs2[i], (224, i * 224))
        combined_image.paste(imgs3[i], (2 * 224, i * 224))

    combined_image = Image.new('RGB', (len(frame_id_list) * 224,3 * 224 ))
    for i in range(len(frame_id_list)):
        combined_image.paste(imgs[i], ( i * 224,0))
        combined_image.paste(imgs2[i], (i * 224,224))
        combined_image.paste(imgs3[i], ( i * 224,2 * 224))

    img_path = os.path.join(savedir+'/'+savename+'_rebuild.jpg')
    combined_image.save(img_path)











def save_real_data_images_withrebuild2(video_path,outputreal, outputfake, imgs, imgs_f, rebuild_r, rebuild_f, rebuild_r2, rebuild_f2, labels, mean, std,sampled_idxs, save_dir=None, save_name=None, return_np=False):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    t_r = imgs.size(1)
    result = Image.new('RGB', (t_r*one_image_w-padding_num*2, 6*one_image_w-padding_num+40),"white")
    # mean = 0
    # std = 1
    if imgs.size(0) !=0:
        for i in range(t_r):
            img_r = Image.fromarray(tensor2im(imgs[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_r, box=(i*one_image_w, 0))
            drawer=ImageDraw.Draw(result)
            img_rp = Image.fromarray(tensor2im(rebuild_r[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_rp, box=(i*one_image_w, one_image_w))
            drawer=ImageDraw.Draw(result)
            img_rp = Image.fromarray(tensor2im(rebuild_r2[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_rp, box=(i*one_image_w, one_image_w*2))
            drawer=ImageDraw.Draw(result)            
        drawer.text(xy=(one_image_w+int(one_image_w/2)-20,one_image_w),fill=(0,0,0),text=str(outputreal[0][0].item()),font=font)
        drawer.text(xy=(one_image_w+int(one_image_w/2)-20,one_image_w+20),fill=(0,0,0),text=str(outputreal[0][1].item()),font=font)
    t_f = imgs_f.size(1)
    if imgs_f.size(0) !=0:   
        for i in range(t_f):
            img_f = Image.fromarray(tensor2im(imgs_f[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_f, box=(i*one_image_w, one_image_w*3))
            drawer=ImageDraw.Draw(result)
            img_fp = Image.fromarray(tensor2im(rebuild_f[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_fp, box=(i*one_image_w, one_image_w*4))
            drawer=ImageDraw.Draw(result)
            img_fp = Image.fromarray(tensor2im(rebuild_f2[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_fp, box=(i*one_image_w, one_image_w*5))
            drawer=ImageDraw.Draw(result)            
                     
        drawer.text(xy=(one_image_w+int(one_image_w/2)-20,one_image_w*2),fill=(0,0,0),text=str(outputfake[0][0].item()),font=font)
        drawer.text(xy=(one_image_w+int(one_image_w/2)-20,one_image_w*2+20),fill=(0,0,0),text=str(outputfake[0][1].item()),font=font)
        drawer.text(xy=(0,one_image_w),fill=(255,0,0),text=str(labels[0].item()),font=font)
    # predinfo = 'new'
    if imgs.size(0) !=0 and imgs_f.size(0) ==0:
        if outputreal[0][0].item()<0.5:
            predinfo = 'Real->Fake'
        else:
            predinfo = 'allcroeet' 
    elif imgs.size(0) ==0 and imgs_f.size(0) !=0:        
        if outputfake[0][1].item()<0.5:
            predinfo = 'Fake->Real'
        else:
            predinfo = 'allcroeet' 
    else:
        if outputreal[0][0].item()==0.5 or outputfake[0][1].item()==0.5:
            predinfo = 'noframe'         
        elif outputreal[0][0].item()>0.5 and outputfake[0][1].item()>0.5:
            predinfo = 'allcroeet'    
        elif outputreal[0][0].item()<0.5 and outputfake[0][1].item()<0.5:
            predinfo = 'Fake->Real and Real->Fake'
        elif outputreal[0][0].item()<0.5:
            predinfo = 'Real->Fake'        
        elif outputfake[0][1].item()<0.5:
            predinfo = 'Fake->Real'                        
            
    video_name_arr = video_path[0].split("/")
    video_name = video_name_arr[4]+"_"+video_name_arr[-1]
    
    if return_np:
        return np.array(result)
    else:
        img_path = os.path.join(save_dir+'/'+save_name+'-'+video_name+'_'+predinfo+'.jpg')
        result.save(img_path,  quality=75)


def save_real_data_images_withrebuild3(video_path,outputreal, outputfake, imgs, imgs_f, rebuild_r, rebuild_f, rebuild_r2, rebuild_f2, rebuild_r3, rebuild_f3,labels, mean, std,sampled_idxs, save_dir=None, save_name=None, return_np=False):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    t_r = imgs.size(1)
    result = Image.new('RGB', (t_r*one_image_w-padding_num*2, 8*one_image_w-padding_num+40),"white")
    # mean = 0
    # std = 1
    if imgs.size(0) !=0:
        for i in range(t_r):
            img_r = Image.fromarray(tensor2im(imgs[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_r, box=(i*one_image_w, 0))
            drawer=ImageDraw.Draw(result)
            img_rp = Image.fromarray(tensor2im(rebuild_r[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_rp, box=(i*one_image_w, one_image_w))
            drawer=ImageDraw.Draw(result)
            img_rp = Image.fromarray(tensor2im(rebuild_r2[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_rp, box=(i*one_image_w, one_image_w*2))
            drawer=ImageDraw.Draw(result)
            img_rp = Image.fromarray(tensor2im(rebuild_r3[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_rp, box=(i*one_image_w, one_image_w*3))
            drawer=ImageDraw.Draw(result)            
            
                        
        drawer.text(xy=(one_image_w+int(one_image_w/2)-20,one_image_w),fill=(0,0,0),text=str(outputreal[0][0].item()),font=font)
        drawer.text(xy=(one_image_w+int(one_image_w/2)-20,one_image_w+20),fill=(0,0,0),text=str(outputreal[0][1].item()),font=font)
    t_f = imgs_f.size(1)
    if imgs_f.size(0) !=0:   
        for i in range(t_f):
            img_f = Image.fromarray(tensor2im(imgs_f[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_f, box=(i*one_image_w, one_image_w*4))
            drawer=ImageDraw.Draw(result)
            img_fp = Image.fromarray(tensor2im(rebuild_f[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_fp, box=(i*one_image_w, one_image_w*5))
            drawer=ImageDraw.Draw(result)
            img_fp = Image.fromarray(tensor2im(rebuild_f2[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_fp, box=(i*one_image_w, one_image_w*6))
            drawer=ImageDraw.Draw(result)
            img_fp = Image.fromarray(tensor2im(rebuild_f3[:,i,:,:,:],mean, std)).resize([one_image,one_image], Image.BICUBIC)
            result.paste(img_fp, box=(i*one_image_w, one_image_w*7))
            drawer=ImageDraw.Draw(result)                        
                     
        drawer.text(xy=(one_image_w+int(one_image_w/2)-20,one_image_w*2),fill=(0,0,0),text=str(outputfake[0][0].item()),font=font)
        drawer.text(xy=(one_image_w+int(one_image_w/2)-20,one_image_w*2+20),fill=(0,0,0),text=str(outputfake[0][1].item()),font=font)
        drawer.text(xy=(0,one_image_w),fill=(255,0,0),text=str(labels[0].item()),font=font)
    # predinfo = 'new'
    if imgs.size(0) !=0 and imgs_f.size(0) ==0:
        if outputreal[0][0].item()<0.5:
            predinfo = 'Real->Fake'
        else:
            predinfo = 'allcroeet' 
    elif imgs.size(0) ==0 and imgs_f.size(0) !=0:        
        if outputfake[0][1].item()<0.5:
            predinfo = 'Fake->Real'
        else:
            predinfo = 'allcroeet' 
    else:
        if outputreal[0][0].item()==0.5 or outputfake[0][1].item()==0.5:
            predinfo = 'noframe'         
        elif outputreal[0][0].item()>0.5 and outputfake[0][1].item()>0.5:
            predinfo = 'allcroeet'    
        elif outputreal[0][0].item()<0.5 and outputfake[0][1].item()<0.5:
            predinfo = 'Fake->Real and Real->Fake'
        elif outputreal[0][0].item()<0.5:
            predinfo = 'Real->Fake'        
        elif outputfake[0][1].item()<0.5:
            predinfo = 'Fake->Real'                        
            
    video_name_arr = video_path[0].split("/")
    video_name = video_name_arr[4]+"_"+video_name_arr[-1]
    
    if return_np:
        return np.array(result)
    else:
        img_path = os.path.join(save_dir+'/'+save_name+'-'+video_name+'_'+predinfo+'.jpg')
        result.save(img_path,  quality=75)





def save_bi_images(video_name,imgs_r, imgs_f, sampled_idxs,landmarks_68,landmarks_5,bboxes,masks=None, file='bi'):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    t = len(imgs_f)
    result = Image.new('RGB', (t*one_image_w-padding_num*2, one_image_w*3-padding_num+40),"white")
    for i in range(t):
        img_r = visualize_landmark(imgs_r[i], landmarks_68[i],landmarks_5[i],bboxes[i])
        img_r = img_r.resize([one_image,one_image], Image.BICUBIC)
        img_f = visualize_landmark(imgs_f[i], landmarks_68[i],landmarks_5[i],bboxes[i])
        img_f = img_f.resize([one_image,one_image], Image.BICUBIC)
        if len(masks[i].shape) > 2:
            masks[i] = masks[i][:,:,0]
        mask = Image.fromarray(np.uint8(masks[i]*255)).convert('RGB').resize([one_image,one_image], Image.BICUBIC)

        result.paste(img_r, box=(i*one_image_w, 0))
        result.paste(img_f, box=(i*one_image_w, one_image_w))
        result.paste(mask, box=(i*one_image_w, one_image_w*2))
        drawer=ImageDraw.Draw(result)
        drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w*3),fill=(0,0,0),text=str(sampled_idxs[i]),font=font)
    img_path = 'images/'+str(video_name)+'_'+file+'.jpg'
    result.save(img_path,  quality=75)
    
    
def visualize_landmark(image_array, landmarks,landmarks_5,bboxes):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    if landmarks is not None:
        landmarks = np.array(landmarks)
        for landmark in landmarks:
            # print(landmark.shape)   #81,2
            draw.point((landmark[0],landmark[1]),fill = (255, 255, 0))
    if landmarks_5 is not None:
        for landmark in landmarks_5:
            draw.point((landmark[0],landmark[1]),fill = (255, 255, 255))
    # line = 5
    # x, y = 10, 10
    # width, height = 100, 50
    # for i in range(1, line + 1):
    #     draw.rectangle((x + (line - i), y + (line - i), x + width + i, y + height + i), outline='red')
    draw.rectangle((bboxes[0][0],bboxes[0][1],bboxes[1][0],bboxes[1][1]), outline='red',width=2)

    # imshow(origin_img)
    return origin_img 
def save_multi_image(images,sampled_idxs, save_dir, mode='RGB'):
    one_image  = 224
    padding_num = 4
    one_image_w = one_image+2*padding_num
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')),20)
    result = Image.new(mode, (len(images)*one_image_w-padding_num*2, one_image_w+20),"white")
    for i, (image,idx) in enumerate(zip(images, sampled_idxs)):
        # image_pil = Image.fromarray(tensor2im(image[i])).resize([one_image,one_image], Image.BICUBIC)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = image.resize([one_image,one_image], Image.BICUBIC)
        result.paste(image, box=(i*one_image_w, 0))
        drawer=ImageDraw.Draw(result)
        drawer.text(xy=(i*one_image_w+int(one_image_w/2),one_image_w),fill=(0,0,0),text=str(idx),font=font)
    img_path = os.path.join(save_dir+'.jpg')
    result.save(img_path,  quality=75)



import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=0, length=0):
        n_holes = n_holes
        length = length

    def __call__(self, imgs):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = imgs[0].size(1)
        w = imgs[0].size(2)
        n_holes = np.random.randint(1,4)
        length = np.random.randint(1,30)
        length = int(h*length*0.01)*2
        mask = np.ones((h, w), np.float32)

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(imgs)
        imgs = imgs * mask
        return imgs
class Cutout_custom(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=0, length=0):
        n_holes = n_holes
        length = length

    def __call__(self, imgs):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = imgs[0].size(1)
        w = imgs[0].size(2)
        n_holes = np.random.randint(1,4)
        length = np.random.randint(1,30)
        length = int(h*length*0.01)*2
        mask = np.ones((h, w), np.float32)

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(imgs)
        imgs = imgs * mask
        return imgs,mask
class Cutout_ori(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=0, length=0):
        n_holes = n_holes
        length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        n_holes = np.random.randint(1,4)
        length = np.random.randint(1,30)
        length = int(h*length*0.01)*2
        mask = np.ones((h, w), np.float32)

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

import pickle
from collections import OrderedDict
def extract_data_list(file_url, is_sbi=False,method='ALL'):
    dataset_list = pickle.load(open(file_url, 'rb'))
    files = OrderedDict()
    files['common'] = []
    files['ForgeryNet'] = []
    if method == "ALL":
        common_dataset = ['FaceForensics_c23','Celeb-DF','DFDC']
    else:
        common_dataset = [method]
    fake_count = 0
    real_count = 0
    for i, datasets_name in enumerate(dataset_list):
        if datasets_name in common_dataset:
            for dir in dataset_list[datasets_name]['list']:
                for file_list in dataset_list[datasets_name]['list'][dir]:
                    tmp_url = os.path.join(dataset_list[datasets_name]['dataset_path'], file_list[0])
                    if 'cache_f' not in tmp_url and '/fr' != tmp_url[-3:]:
                        if is_sbi:
                            if file_list[1]==0:
                                tmp_sbi = ('sbi/'+tmp_url, file_list[1], file_list[2])
                                files['common'].append(tmp_sbi)
                        else:
                            tmp = (tmp_url, file_list[1], file_list[2])
                            files['common'].append(tmp)
    selected_datasets = []
    for (file_path, video_label, file_type)  in files['common']:
        if 'FaceShifter/' not in file_path:
            selected_datasets.append((file_path, video_label, file_type))
            if video_label==0:
                real_count = real_count+1
            else:
                fake_count = fake_count+1
    files['common'] = selected_datasets
    print(len(files['common'])+len(files['ForgeryNet']))
    print('real length='+str(real_count))
    print('fake length='+str(fake_count))
    return files

import json
import pandas as pd
def extract_file_from_ouc(dataset_name, root, split, only_real=False):
    dataset_info = []
    if dataset_name == 'FF-ALL':
        return extrace_FF_from_ouc(root,split,only_real)
    elif dataset_name == 'Celeb-DF' and split=='test':
        video_list_txt = os.path.join(root, 'List_of_testing_videos.txt')
        with open(video_list_txt) as f:
            for data in f:
                line=data.split()
                dataset_info.append((line[1][:-4],1-int(line[0])))
    elif dataset_name == 'DFDC' and split=='test':
        label=pd.read_csv(root+'labels.csv',delimiter=',')
        dataset_info = [(video_name[:-4], label) for video_name, label in zip(label['filename'].tolist(), label['label'].tolist())]
    return dataset_info

def extrace_FF_from_ouc(root, split, only_real=False):
    split_json_path = os.path.join(root, 'splits', f'{split}.json')
    # print(split_json_path, type(split_json_path))
    split_json_path = split_json_path.replace("videos/","")
    json_data = json.load(open(split_json_path, 'r'))
    if only_real:
        real_names = []
        for item in json_data:
            real_names.extend([item[0], item[1]])
        real_video_dir = os.path.join('original_sequences', 'youtube', 'c23', 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), 0] for x in real_names]
    else:
        real_names = []
        fake_names = []
        # i = 0
        for item in json_data:
            real_names.extend([item[0], item[1]])
            fake_names.extend([f'{item[0]}_{item[1]}', f'{item[1]}_{item[0]}'])
            # i += 1
            # if i>3:
            #     break
        real_video_dir = os.path.join('original_sequences', 'youtube', 'c23', 'videos')
        dataset_info = [[os.path.join(real_video_dir,x), 0] for x in real_names]
        ff_fake_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        for method in ff_fake_types:
            fake_video_dir = os.path.join('manipulated_sequences', method, 'c23', 'videos')
            for x in fake_names:
                dataset_info.append((os.path.join(fake_video_dir,x),1))
    return dataset_info

def load_face_from_ouc(root, file_info, is_sbi=False):
    if is_sbi:
        pkl_path = os.path.join(root, 'face_v2',file_info+'.pkl')
    else:
        if 'DFDC' in root:
            pkl_path = os.path.join(root, 'ftcn_v2', 'test_videos', file_info + '.pkl')
        else:
            pkl_path = os.path.join(root, 'ftcn_v2',file_info+'.pkl')
    return pkl_path

def load_face_from_a100(file_info, is_sbi=False):
    if is_sbi:
        file_info = file_info.replace('/face_v1/', '/face_v2/')
    if '/c40/' in file_info:
        pkl_path = file_info.replace('/c40/', '/c23/')
        pkl_path = pkl_path +".pkl"
    else:
        pkl_path = file_info +".pkl"
    
    # else:
    #     if 'DFDC' in root:
    #         pkl_path = os.path.join(root, 'ftcn_v2', 'test_videos', file_info + '.pkl')
    #     else:
    #         pkl_path = os.path.join(root, 'ftcn_v2',file_info+'.pkl')
    return pkl_path

if __name__ == '__main__':
    extract_data_list(os.path.join("../../../../datasets/rawdata/dataset_list_v1/test.pkl"))