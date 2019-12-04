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
    block = 'arr_' + str(block)
    return npz[block]

def match_best_IoU(base_map, maps):
    best_iou = 0
    for m in maps.T:
        iou = IoU(base_map, m)
        if iou > best_iou:
            best_iou = iou
            best_m = m          
    return best_iou, best_m
    
def comapre_mask(model, npz_dict, mask, img_id):
    Maps_IOU = {}
    Maps_image = {}     
    for block in [0, 1, 2, 3, 4]:
        maps = get_block_output(npz_dict[model], block)
        best_iou, best_m = match_best_IoU(mask, maps)
        Maps_IOU[block] = best_iou 
        Maps_image[block] = best_m
    
    Maps_IOU[5] = IoU(mask, get_block_output(npz_dict[model], 5)[:, img_id]) 
    Maps_image[5] = get_block_output(npz_dict[model], 5)[:, img_id]
    print ('Finished comparing Ground truth and %s' % model)
    return Maps_IOU, Maps_image

def compare_basenet(basenet, model, npz_dict, Images_M19, img_id):
    Maps_IOU = {}
    Maps_image = {}     
    for block in [0, 1, 2, 3, 4]:
        maps = get_block_output(npz_dict[model], block)
        best_iou, best_m = match_best_IoU(Images_M19[block], maps)
        Maps_IOU[block] = best_iou 
        Maps_image[block] = best_m
        
    Maps_IOU[5] = IoU(Images_M19[5], get_block_output(npz_dict[model], 5)[:, img_id]) 
    Maps_image[5] = get_block_output(npz_dict[model], 5)[:, img_id]
    
    print ('Finished comparing %s and %s' % (basenet, model))
    return Maps_IOU, Maps_image

    
def draw_vis(IOUs_dict, Images_dict, output_name, img_cat, predicted_label):
    fig, axs = plt.subplots(2, 3)
    for image in Images_dict:
        p = Images_dict[image].reshape(224, 224)    
        axs[int(image/3), image%3].imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')
        axs[int(image/3), image%3].set_title('Layer %s IoU: %.3f' % (image, IOUs_dict[image]))
        axs[int(image/3), image%3].axis('off')
    plt.savefig('./IOU_results/%s_%s_%s.jpg' % (img_cat, output_name, predicted_label))
    return 

def output_result(basenet, model_name, npz_dict, mask, img_cat, img_id, predicted_labels, Images_basenet = None):
    predicted_label = predicted_labels[model_name]
    IOUs_dict, Images_dict = comapre_mask(model_name, npz_dict, mask, img_id)
    draw_vis(IOUs_dict, Images_dict, model_name + '_GT', img_cat, predicted_label)
    if model_name != basenet:
        IOUs_dict, Images_dict = compare_basenet(basenet, model_name, npz_dict, Images_basenet, img_id)
        draw_vis(IOUs_dict, Images_dict, model_name + '_%s' % basenet, img_cat, predicted_label)
        return 
    else:
        return Images_dict
    
def IoU_calculation(mynets, img_cat, predicted_labels, img_id, npz_dict, basenet):
    mask = np.load('./data/%s.npy' % img_cat).reshape(224*224)
    Images_basenet = output_result(basenet, basenet, npz_dict, mask, img_cat, img_id, predicted_labels)
    mynets.remove(basenet)
    for mynet in mynets:
        output_result(basenet, mynet, npz_dict, mask, img_cat, img_id, predicted_labels, Images_basenet)
    return

