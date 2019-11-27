# -*- coding: utf-8 -*-
"""

Utility methods for handling the classifiers:
    set_caffe_mode(gpu)
    get_caffenet(netname)
    forward_pass(net, x, blobnames='prob', start='data')

"""

# this is to supress some unnecessary output of caffe in the linux console
import os
os.environ['GLOG_minloglevel'] = '2'
         
import numpy as np
import caffe 
import torch
import torch.nn as nn
import torchvision.models as models
import torch

import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import copy


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        for i, layer in enumerate(modules):
            print ('layer', i, layer)
        conv1 = list(resnet.children())[:4]
        conv2 = list(resnet.children())[:5]
        conv3 = list(resnet.children())[:6]
        conv4 = list(resnet.children())[:7]
        conv5 = list(resnet.children())[:8]

        self.last_layer = list(resnet.children())[-1]    # input features to this layer
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)
        self.conv3 = nn.Sequential(*conv3)
        self.conv4 = nn.Sequential(*conv4)
        self.conv5 = nn.Sequential(*conv5)
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, 256)
        self.bn = nn.BatchNorm1d(256, momentum = 0.01)       
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        feat1 = self.conv1(images)
        feat2 = self.conv2(images)
        feat3 = self.conv3(images)
        feat4 = self.conv4(images)
        feat5 = self.conv5(images)
        features = features.reshape(features.size(0), -1)
        last_feat = self.last_layer(features)
        return feat1, feat2, feat3, feat4, feat5, last_feat
    


def get_pytorchnet(netname):
    model = EncoderCNN()
    PATH = './Caffe_Models/' + netname + '.ckpt'
    model.load_state_dict(torch.load(PATH), strict = False)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model = model.float()
    model.eval()
    return model


def set_caffe_mode(gpu):
    ''' Set whether caffe runs in gpu or not, input is boolean '''
    if gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()    
        
     
def get_caffenet(netname):
    
    if netname=='googlenet':
     
        # caffemodel paths
        model_path = './Caffe_Models/googlenet/'
        net_fn   = model_path + 'deploy.prototxt'
        param_fn = model_path + 'bvlc_googlenet.caffemodel'
        
        # get the mean (googlenet doesn't do this per feature, but per channel, see train_val.prototxt)
        mean = np.float32([104.0, 117.0, 123.0]) 
        
        # define the neural network classifier
        net = caffe.Classifier(net_fn, param_fn, caffe.TEST, channel_swap = (2,1,0), mean = mean)

    elif netname=='alexnet':
            
        # caffemodel paths
        model_path = './Caffe_Models/bvlc_alexnet/'
        net_fn   = model_path + 'deploy.prototxt'
        param_fn = model_path + 'bvlc_alexnet.caffemodel'
        
        # get the mean
        mean = np.load('./Caffe_Models/ilsvrc_2012_mean.npy')
        # crop mean
        image_dims = (224, 224) # see deploy.prototxt file
        excess_h = mean.shape[1] - image_dims[0]
        excess_w = mean.shape[2] - image_dims[1]
        mean = mean[:, excess_h:(excess_h+image_dims[0]), excess_w:(excess_w+image_dims[1])]
        
        # define the neural network classifier
        net = caffe.Classifier(net_fn, param_fn, caffe.TEST, channel_swap = (2,1,0), mean = mean)
        
    elif netname == 'vgg':
    
        # caffemodel paths
        model_path = './Caffe_Models/vgg network/'
        net_fn   = model_path + 'VGG_ILSVRC_16_layers_deploy.prototxt'
        param_fn = model_path + 'VGG_ILSVRC_16_layers.caffemodel'
        
        mean = np.float32([103.939, 116.779, 123.68])    
        
        # define the neural network classifier    
        net = caffe.Classifier(net_fn, param_fn, caffe.TEST, channel_swap = (2,1,0), mean = mean)
        
    else:
        
        print ('Provided netname unknown. Returning None.')
        net = None
    
    return net  
     


def forward_pass(net, mynet, x, blobnames=['prob'], start='data'):
    
    # get input into right shape
    if np.ndim(x)==3:
        x = x[np.newaxis]  
    if np.ndim(x)<4:
        input_shape = net.blobs[start].data.shape
        x = x.reshape([x.shape[0]]+list(input_shape)[1:])

    # reshape net so it fits the batchsize (implicitly given by x)
    if net.blobs['data'].data.shape[0] != x.shape[0]:
        net.blobs['data'].reshape(*(x.shape))

        
    # feed forward the batch through the next
    x = torch.from_numpy(x)
    
#     y = mynet(x.float())
#     y = torch.nn.functional.softmax(y, dim = 1).data.numpy()
    
    y = mynet(x.float())
    y1 = y[0].data.numpy()
    y2 = y[1].data.numpy()
    y3 = y[2].data.numpy()
    y4 = y[3].data.numpy()
    y5 = y[4].data.numpy()

    y6 = y[-1]
    y6 = torch.nn.functional.softmax(y6, dim = 1).data.numpy()
    print (np.max(y6))
    returnVals = [y1, y2, y3, y4, y5, y6]
     
    return returnVals

