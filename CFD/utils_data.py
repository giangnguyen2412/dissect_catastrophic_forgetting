# -*- coding: utf-8 -*-
import numpy as np
import os
import PIL
from torchvision import transforms
import matplotlib.pyplot as plt

path_data = "./data"
  
def get_image_data(img_size = 224):

    img_list = os.listdir(path_data)

    for img_file in img_list[:]:
        if not (img_file.endswith(".png") or img_file.endswith(".jpg")):
            img_list.remove(img_file)
        
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
            std=[0.229, 0.224, 0.225])])

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

    X_im = np.uint8(np.array(X_im))
    
    X = np.array(X)
    
    return X, X_im, X_filenames
