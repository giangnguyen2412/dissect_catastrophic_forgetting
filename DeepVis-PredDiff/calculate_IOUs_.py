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

def compare_M19(model, Images_M19, npz_dict):
    Maps_IOU = {}
    Maps_image = {}     
    for block in [0, 1, 2, 3, 4, 5]:
        maps = get_block_output(npz_dict[model], block)
        best_iou, best_m = match_best_IoU(Images_M19[block], maps)
        Maps_IOU[block] = best_iou 
        Maps_image[block] = best_m
    print ('Finished comparing M19 and %s' % model)
    return Maps_IOU, Maps_image

    
def draw_vis(IOUs_dict, Images_dict, output_name, img_cat, predicted_label):
    fig, axs = plt.subplots(2, 3)
    for image in Images_dict:
        p = Images_dict[image].reshape(224, 224)    
        axs[int(image/3), image%3].imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')
        axs[int(image/3), image%3].set_title('Layer %s IoU: %.3f' % (image, IOUs_dict[image]))
        axs[int(image/3), image%3].axis('off')
    plt.savefig('./results_IOU/%s_%s_%s.jpg' % (img_cat, output_name, predicted_label))
    return 
    
def output_result(model_name, npz_dict, mask, img_cat, img_id, predicted_labels, Images_M19 = None):
    predicted_label = predicted_labels[model_name]
    IOUs_dict, Images_dict = comapre_mask(model_name, npz_dict, mask, img_id)
    draw_vis(IOUs_dict, Images_dict, model_name + '_GT', img_cat, predicted_label)
    if model_name != 'M19':
        IOUs_dict, Images_dict = compare_M19(model_name, Images_M19, npz_dict)
        draw_vis(IOUs_dict, Images_dict, model_name + '_M19', img_cat, predicted_label)
        return 
    else:
        return Images_dict
    
def IoU_calculation(img_cat, predicted_labels, img_id, npz_dict):
        
    mask = np.load('./data/%s.npy' % img_cat).reshape(224*224)
    Images_M19 = output_result('M19', npz_dict, mask, img_cat, img_id, predicted_labels)
    output_result('M20', npz_dict, mask, img_cat, img_id, predicted_labels, Images_M19)
    output_result('M24', npz_dict, mask, img_cat, img_id, predicted_labels, Images_M19)
    return

img_cat = 'bus'
img_id = 734
predicted_labels = {'M19': 'train', 'M20': 'train', 'M24': 'train'}
npz_dict = {}
npzs = glob.glob('results/%s*.npz' % img_cat)


for npz in npzs:
    model_name = npz.split('.npz')[0].split('_')[-1]
    npz_dict[model_name] = np.load(npz)
    
IoU_calculation(img_cat, predicted_labels, img_id, npz_dict)
