# -*- coding: utf-8 -*-
"""
This script is a modified version of Prediction Difference Analysiss (PDA), see this paper:
"Visualizing Deep Neural Network Decisions: Prediction Difference Analysis" - Luisa M Zintgraf, Taco S Cohen, Tameem Adel, Max Welling

Part of the codes are copied from PDA's tool box, see
https://github.com/lmzintgraf/DeepVis-PredDiff 

In CFD, this is made to tell the user which part of the model is forgetting the most after continuous learning.
@Author: Shuan Chen
"""

import matplotlib 
matplotlib.use('Agg')   

import numpy as np
import time
import glob
import os

# PDA, original coed from Luisa M Zintgraf
from prediction_difference_analysis import PredDiffAnalyser

# Import utilities 
import utils_classifiers as utlC
import utils_data as utlD
import utils_sampling as utlS
import Calculate_IOUs as CI


def PDA(mynets, basenet, img_size):
    # Basic settings of PDA
    win_size = 5 
    overlapping = False
    num_samples = 10
    padding_size = 2            
    batch_size = 12
    image_dims = (img_size, img_size)

    # Get the image data
    X_test, X_test_im, X_filenames = utlD.get_image_data()
    test_indices = [i for i in range(X_test.shape[0])]
    
    # Make folder for saving the results        
    IOU_savepath = './IOU_results/' 
    if not os.path.exists(IOU_savepath):
        os.makedirs(IOU_savepath)  

    npz_savepath = './npz_results/' 
    if not os.path.exists(npz_savepath):
        os.makedirs(npz_savepath)   
    
    # ------------------------ EXPERIMENTS ------------------------
    
    for test_idx in test_indices:
        npz_dict = {}
        for mynet_name in mynets:
            mynet = utlC.get_pytorchnet(mynet_name)
            x_test = X_test[test_idx]
                
            target_func = lambda x: utlC.forward_pass(mynet, x, img_size)                               
            print ("doing test ...",  "file :", X_filenames[test_idx], ", net:", mynet_name, ", win_size:", win_size)
       
            start_time = time.time()
            sampler = utlS.cond_sampler_imagenet(win_size=win_size, padding_size=padding_size, image_dims=image_dims, netname=mynet_name) 
            pda = PredDiffAnalyser(x_test, target_func, sampler, num_samples=num_samples, batch_size=batch_size)
            pred_diff = pda.get_rel_vect(win_size=win_size, overlap=overlapping)        
            print ("--- Total computation took {:.4f} minutes ---".format((time.time() - start_time)/60))
            
            np.savez(npz_savepath + mynet_name + '_' + X_filenames[test_idx] + '.npz', *pred_diff)      
             # Load the  previous made npz files instead doing PDA again
#            loaded_npz = np.load(npz_savepath + mynet_name + '_' + X_filenames[test_idx] + '.npz', mmap_mode='r')            
#            pred_diff= [loaded_npz['arr_0'], loaded_npz['arr_1'], loaded_npz['arr_2'], loaded_npz['arr_3'], loaded_npz['arr_4']]
            npz_dict[mynet_name] = pred_diff
        # Calculate the IoU between model vision and reference vision
        CI.IoU_calculation(mynets, X_filenames[test_idx][:-3], npz_dict, basenet)
    CI.conclude_reports()
    return

# Reference model (before forgetting)
basenet = 'M19'
# THe directory where your models locate
model_paths = glob.glob('./Pytorch_Models/*.ckpt')

# The image size input to the model
img_size = 224

models = [model_path.split('/')[-1].split('.')[0] for model_path in model_paths]
print ('Loaded models:', models, 'basenet:', basenet)
PDA(models, basenet, img_size)

