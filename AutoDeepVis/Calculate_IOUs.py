import numpy as np
import glob
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def IoU(seg1, seg2):
    seg1 = np.array(seg1.clip(min = 0), dtype=bool)
    seg2 = np.array(seg2.clip(min = 0), dtype=bool)
    overlap = seg1*seg2 # Logical AND
    union = seg1 + seg2 # Logical OR
    IOU = overlap.sum()/float(union.sum())
    return IOU

def get_block_output(npz, block):
    #block = 'arr_' + str(block)
    return npz[block]

def match_best_IoU(base_map, maps):
    best_iou = 0
    for m in maps.T:
        iou = IoU(base_map, m)
        if iou > best_iou:
            best_iou = iou
            best_m = m          
    return best_iou, best_m
    
def comapre_mask(model, npz_dict, mask):
    Maps_IOU = {}
    Maps_image = {}     
    for block in range(5):
        maps = get_block_output(npz_dict[model], block)
        best_iou, best_m = match_best_IoU(mask, maps)
        Maps_IOU[block] = best_iou 
        Maps_image[block] = best_m
    
    print ('Finished comparing Ground truth and %s' % model)
    return Maps_IOU, Maps_image

def compare_basenet(basenet, model, npz_dict, Images_basenet):
    Maps_IOU = {}
    Maps_image = {}     
    for block in range(5):
        maps = get_block_output(npz_dict[model], block)
        best_iou, best_m = match_best_IoU(Images_basenet[block], maps)
        Maps_IOU[block] = best_iou 
        Maps_image[block] = best_m
    
    print ('Finished comparing %s and %s' % (basenet, model))
    return Maps_IOU, Maps_image

    
def draw_vis(IOUs_dict, Images_dict, output_name, img_cat):
    fig, axs = plt.subplots()
    for image in Images_dict:
        p = Images_dict[image].reshape(224, 224)    
        axs.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')
        axs.set_title('IoU: %.3f' % (IOUs_dict[image]))
        axs.axis('off')
        plt.savefig('./IOU_results/%s_%s_block%s .jpg' % (img_cat, output_name, (image+1)))
    return 

def write_forgetting_layer(IOUs_dict, model_name, img_cat):
    forgetting_report = './IOU_results/forgetting_report.txt'
    slopes = []
    for layer in IOUs_dict.keys():
        if int(layer) != 4 and int(layer) != 5:
            slope = IOUs_dict[layer+1] - IOUs_dict[layer]
            slopes.append(slope)
        else:
            slopes.append(0)
    fp = open(forgetting_report , 'a')
    fp.write('%s\t%s\t%s\n' %(model_name, img_cat, (np.argmin(slopes)+2)))
    fp.close()
    return
    

def output_result(basenet, model_name, npz_dict, mask, img_cat, Images_basenet = None):
    
    IOUs_dict, Images_dict = comapre_mask(model_name, npz_dict, mask)
    draw_vis(IOUs_dict, Images_dict, model_name + '_GT', img_cat)
    if model_name != basenet:
        IOUs_dict, Images_dict = compare_basenet(basenet, model_name, npz_dict, Images_basenet)
        draw_vis(IOUs_dict, Images_dict, model_name + '_%s' % basenet, img_cat)
        write_forgetting_layer(IOUs_dict, model_name, img_cat)
        return 
    else:
        return Images_dict
    
def IoU_calculation(mynets, img_cat, npz_dict, basenet):
    mask = np.load('./data/%s.npy' % img_cat).reshape(224*224)
    Images_basenet = output_result(basenet, basenet, npz_dict, mask, img_cat)
    testnets = mynets.copy()
    testnets.remove(basenet)
    for mynet in testnets:
        output_result(basenet, mynet, npz_dict, mask, img_cat, Images_basenet)
    return

