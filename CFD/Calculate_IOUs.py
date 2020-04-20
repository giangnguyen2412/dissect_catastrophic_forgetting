import numpy as np
import glob
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

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


def compare(model, npz_dict, mask):
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
    forgetting_report = './IOU_results/forgetting_report_%s.txt' % model_name
    slopes = []
    for layer in IOUs_dict.keys():
        if int(layer) != 4 and int(layer) != 5:
            slope = IOUs_dict[layer+1] - IOUs_dict[layer]
            slopes.append(slope)
        else:
            slopes.append(0)

    def find_min(slopes):
        min_idx = list(np.argwhere(slopes == np.min(slopes))[:, 0])
        return min_idx

    fp = open(forgetting_report , 'a')
    forget_blocks = find_min(slopes)
    for block in forget_blocks:
        fp.write('%s\t%s\n' %(img_cat, block + 2))
    fp.close()
    return
    

def output_result(basenet, model_name, npz_dict, mask, img_cat, Images_basenet = None):
    
    IOUs_dict, Images_dict = compare(model_name, npz_dict, mask)
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

def conclude_reports():
    reports = glob.glob('./IOU_results/forgetting_report_*.txt')
    def find_max(blocks):
        blocks_count = [blocks.count(n) for n in set(blocks)]  
        max_idx = list(np.argwhere(blocks_count == np.max(blocks_count))[:, 0])
        forget_block = []
        for idx in max_idx:
            forget_block.append(list(set(blocks))[idx])
        return forget_block

    for report in reports:
        blocks = []
        fp = open(report, 'r')
        for line in fp.readlines():
            blocks.append(line.split('\t')[-1])
        fp.close()
        fp = open(report, 'a')
        forget_blocks = find_max(blocks)
        for block in forget_blocks:
            fp.write('%s\t%s\n' %('The most forgetting block', block))
        fp.close()
    return 




