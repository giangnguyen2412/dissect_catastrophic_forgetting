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
import os

# most important script - relevance estimator
from prediction_difference_analysis import PredDiffAnalyser

# utilities
import utils_classifiers as utlC
import utils_data as utlD
import utils_sampling as utlS
import utils_visualise as utlV
import sensitivity_analysis_caffe as SA
import Calculate_IOUs as CI


# ------------------------ CONFIGURATION ------------------------
# -------------------> CHANGE SETTINGS HERE <--------------------

# pick neural network to run experiment for (alexnet, googlenet, vgg)
# k in alg 1 (see paper)
#Change to False to reduce the calculation time
# settings for sampling 



# ------------------------ SET-UP ------------------------

def PDA(mynets, netname = 'alexnet'):
    test_indices = None
    win_size = 5  
    overlapping = False
    sampl_style = 'conditional' 
    num_samples = 10
    padding_size = 2            
    batch_size = 128
    
    utlC.set_caffe_mode(gpu=False)
    
    net = utlC.get_caffenet(netname)
    # mynet = utlC.get_caffenet(netname)
    
    # get the data
    X_test, X_test_im, X_filenames = utlD.get_imagenet_data(net=net)
    # get the label names of the 1000 ImageNet classes
    classnames = utlD.get_imagenet_classnames()
    
    if not test_indices:
        test_indices = [i for i in range(X_test.shape[0])]      
    
    # make folder for saving the results if it doesn't exist       
    IOU_results = './results_IOU/' 
    if not os.path.exists(IOU_results):
        os.makedirs(IOU_results)              
    # ------------------------ EXPERIMENTS ------------------------
    
    # change the batch size of the network to the given value
    net.blobs['data'].reshape(batch_size, X_test.shape[1], X_test.shape[2], X_test.shape[3])
    
    # target function (mapping input features to output probabilities)

    
    # for the given test indices, do the prediction difference analysis
    for test_idx in test_indices:
        npz_dict = {}
        predicted_labels = {}
        for mynet_name in mynets:
            mynet = utlC.get_pytorchnet(mynet_name)
            target_func = lambda x: utlC.forward_pass(net, mynet, x)
            # get the specific image (preprocessed, can be used as input to the target function)
            x_test = X_test[test_idx]
            # get the image for plotting (not preprocessed)
            # prediction of the network
            y_pred = np.argmax(utlC.forward_pass(net, mynet, x_test)[-1])
            y_pred_label = classnames[y_pred]
            if mynet_name == 'M19':
                base_img_id = y_pred
                
            print (y_pred_label)                     
            print ("doing test...", "file :", X_filenames[test_idx], ", net:", netname, ", win_size:", win_size, ", sampling: ", sampl_style)
       
            start_time = time.time()
            
            if sampl_style == 'conditional':
                sampler = utlS.cond_sampler_imagenet(win_size=win_size, padding_size=padding_size, image_dims=net.crop_dims, netname=netname)
            elif sampl_style == 'marginal':
                sampler = utlS.marg_sampler_imagenet(X_test, net)
                
            pda = PredDiffAnalyser(x_test, target_func, sampler, num_samples=num_samples, batch_size=batch_size)
            pred_diff = pda.get_rel_vect(win_size=win_size, overlap=overlapping)
        
            print ("--- Total computation took {:.4f} minutes ---".format((time.time() - start_time)/60))
            npz_dict[mynet_name] = pred_diff
            predicted_labels[mynet_name] = y_pred_label
        CI.IoU_calculation(X_filenames[test_idx][:-3], predicted_labels, base_img_id, npz_dict)
    return
    
        
PDA(['M19', 'M20', 'M24'])

        
