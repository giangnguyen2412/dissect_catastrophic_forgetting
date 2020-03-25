"""
From this script, experiments for ImageNet pictures can be started.
See "configuration" below for the different possible settings.
The results are saved automatically to the folder ./results

It is recommended to run caffe in gpu mode when overlapping is set
to True, otherwise the calculation will take a very long time.

@author: Luisa M Zintgraf
"""

# the following is needed to avoid some error that can be thrown when 
# using matplotlib.pyplot in a linux shell
import matplotlib 
matplotlib.use('Agg')   

# standard imports
import numpy as np
import time
import glob
import os

# most important script - relevance estimator
from prediction_difference_analysis import PredDiffAnalyser

# utilities
import utils_classifiers as utlC
import utils_data as utlD
import utils_sampling as utlS
import Calculate_IOUs as CI


def PDA(mynets, basenet, img_size):
    test_indices = None
    win_size = 5 
    overlapping = False
    num_samples = 10
    padding_size = 2            
    batch_size = 32
    image_dims = (img_size, img_size)

    # get the data
    X_test, X_test_im, X_filenames = utlD.get_image_data()
    
    if not test_indices:
        test_indices = [i for i in range(X_test.shape[0])]
    
    # make folder for saving the results if it doesn't exist       
    IOU_savepath = './IOU_results/' 
    if not os.path.exists(IOU_savepath):
        os.makedirs(IOU_savepath)  
		
    forgetting_report = './IOU_results/forgetting_report.txt'
    fp = open(forgetting_report , 'w')
    fp.write('Model\tInput_img\tForgetting_layer\n')
    fp.close()
	
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
            
#            loaded_npz = np.load(npz_savepath + mynet_name + '_' + X_filenames[test_idx] + '.npz', mmap_mode='r')            
#            pred_diff= [loaded_npz['arr_0'], loaded_npz['arr_1'], loaded_npz['arr_2'], loaded_npz['arr_3'], loaded_npz['arr_4']]
            npz_dict[mynet_name] = pred_diff

        CI.IoU_calculation(mynets, X_filenames[test_idx][:-3], npz_dict, basenet)
    return
    
basenet = 'M19'
model_paths = glob.glob('./Pytorch_Models/*.ckpt')
models = [model_path.split('/')[-1].split('.')[0] for model_path in model_paths]
img_size = 224
print ('Loaded models:', models, 'basenet:', basenet)
PDA(models, basenet, img_size)

        
