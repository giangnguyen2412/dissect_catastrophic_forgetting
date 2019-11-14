# Visualizing Deep Neural Network Decisions

This code implements the method from the paper

"Visualizing Deep Neural Network Decisions: Prediction Difference Analysis" - Luisa M Zintgraf, Taco S Cohen, Tameem Adel, Max Welling

which was accepted at ICLR2017, see

https://openreview.net/forum?id=BJ5UeU9xx

Note that we are only publishing the code for the ImageNet experiments, since we cannot publish the MRI scans. 
If you are interested in the MRI implementation, please contact me (lmzintgraf@gmail.com).

## ImageNet Experiments

### Dependencies:

If you want to run the IMAGENET experiments using one of the predefined models (our code supports the alexnet, googlenet and vgg) you need to install caffe, see

http://caffe.berkeleyvision.org/

You will also need to download the respective caffemodel files (they're quite large). Please see the readme file in the "./Caffe_Models" folder for further instructions.

### Running Experiments:

The experiments can be run by executing "./IMAGENET Experiments/experiments_imagenet.py". 
Different settings can be adjusted here, please see the file for further information.

### Data:

The above script will use images from the "./data" folder. Only RGB images in format .png and .jpg of a minimum size of 227x227 pixels will be considered. If the image is larger, it will be cut off at the sides. Note that there should be enough images in this folder, since the samplers need them (see paper for further information).

The "./data" folder also contains a text file with the ImageNet class labels.


### Pytorch implementation

This script is rewrote for pytorhc implementation.

1. Download alexnet.caffenet from https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet and upload it to the floder ./Caffe_Models/bvlc_alexnet/
2. Move your model to "Caffe_Models" file with the '.ckpt' file to visualized your model and change the "mynet_name" to your model name in "run_DeepVis.py".
3. If you want to chage the visualized layer, change the "visualizing_layer" to the layer you are interested in.
4. Put the figures you want to visualize in folder ./data

P.S. The default setting of overlapping is Flase to accelerate the running time.
