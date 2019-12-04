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
    PATH = './Pytorch_Models/' + netname + '.ckpt'
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

     


def forward_pass(mynet, x, img_size = 224):
    img_shape = (3, img_size, img_size)
    # get input into right shape
    if np.ndim(x)==3:
        x = x[np.newaxis]
    if np.ndim(x)<4:
        x = x.reshape([x.shape[0]]+list(img_shape)[:])

    x = torch.from_numpy(x)
    
    y = mynet(x.float())
    y1 = y[0].data.numpy()
    y2 = y[1].data.numpy()
    y3 = y[2].data.numpy()
    y4 = y[3].data.numpy()
    y5 = y[4].data.numpy()

    y6 = y[-1]
    y6 = torch.nn.functional.softmax(y6, dim = 1).data.numpy()
    print (np.max(y6[0]))
    returnVals = [y1, y2, y3, y4, y5, y6]
     
    return returnVals

