# -*- coding: utf-8 -*-
"""

Utility methods for handling the ImageNet data:
    get_imagenet_data(net, preprocess)
    get_imagenet_classnames()
    
"""

import numpy as np
import os
import PIL
from torchvision import transforms
import matplotlib.pyplot as plt

path_data = "./data"

  
    
def get_image_data(img_size = 224):

    # get a list of all the images (note that we use networks trained on ImageNet data)
    img_list = os.listdir(path_data)

    # throw away files that are not in the allowed format (png or jpg)
    for img_file in img_list[:]:
        if not (img_file.endswith(".png") or img_file.endswith(".jpg")):
            img_list.remove(img_file)
        
    # fill up data matrix
    img_dim = (img_size, img_size)
    transform1 = transforms.Compose([                  
            transforms.Resize(256),             
            transforms.CenterCrop(img_dim)])

    transform2 = transforms.Compose([                
            transforms.Resize(256),            
            transforms.CenterCrop(img_dim),        
            transforms.ToTensor(),                    
            transforms.Normalize(                     
            mean=[0.485, 0.456, 0.406],               
            std=[0.229, 0.224, 0.225]                 
            )])

    X_filenames = []
    X = []
    X_im = []
    for i in range(len(img_list)):
        img = PIL.Image.open('{}/{}'.format(path_data, img_list[i]))
        if np.array(img).shape[0] >= img_dim[0] and np.array(img).shape[1] >= img_dim[1]:
            img_im = np.array(transform1(img))
            img_t = transform2(img).data.numpy()
            X_im.append(img_im)
            X.append(img_t)
            X_filenames.append(img_list[i].replace(".",""))
            
        else:
            print("Skipped ",img_list[i],", image dimensions were too small.")

    # cast to image values that can be displayed directly with plt.imshow()
    X_im = np.uint8(np.array(X_im))
    
    X = np.array(X)
    
    return X, X_im, X_filenames


def get_imagenet_classnames():
    return np.loadtxt(open(path_data+'/ilsvrc_2012_labels.txt'), dtype=object, delimiter='\n')

def IoU(seg1, seg2):
    seg1 = np.array(seg1.clip(min = 0), dtype=bool)
    seg2 = np.array(seg2.clip(min = 0), dtype=bool)
    overlap = seg1*seg2 # Logical AND
    union = seg1 + seg2 # Logical OR
    IOU = overlap.sum()/float(union.sum())
    return IOU
