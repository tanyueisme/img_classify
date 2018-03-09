#!/usr/bin/env python
#coding=utf-8
import shutil
import mxnet as mx
# sym,arg_params,aux_params = mx.model.load_checkpoint('n02084071',40)
# sym,arg_params,aux_params = mx.model.load_checkpoint('net',40)
# sym,arg_params,aux_params = mx.model.load_checkpoint('imagenet1k-resnet-18',40)
sym,arg_params,aux_params = mx.model.load_checkpoint('imagenet1k-resnet-18-finetune-0',19)
mod = mx.mod.Module(symbol=sym,context=mx.cpu(),data_names=['data'],label_names=['softmax_label'])
mod.bind(for_training=False,data_shapes=[('data',(1,3,224,224))])
mod.set_params(arg_params,aux_params)
with open('synset_905.txt','r') as f:
    labels = [l.rstrip() for l in f]
# %matplotlib inline
# import matplotlib.pyplot as plt
import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(url, show=False):
    #url:图片路径
    #show:是否显示图片
    img = cv2.cvtColor(cv2.imread(url), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    if show:
         # plt.imshow(img)
         # plt.axis('off')
    # convert into format (batch, RGB, width, height)
	img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def predict(url):
    img = get_image(url, show=True)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    # print len(mod.get_outputs()) #是1
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    prob = np.argsort(prob)[::-1]
    top1=prob[0]#取概率最高的一类
    print url
    print top1  #输入类别,0为狗,1为猫
    urlist = url.split('/')
    url1 = '/'.join(urlist[:-2])+'/train_pos_1/'+urlist[-1]
    url0 = '/'.join(urlist[:-2])+'/train_pos_0/'+urlist[-1]
    # print url1
    # print url0
    # if top1==1:
    #     shutil.copyfile(url,url1)
    # else:
    #     shutil.copyfile(url,url0)
    # print labels[top1] #输出标签

#批量测试
# path = '/mxnet/tools/train-cat/2'
path = '/Users/tanyue/work_ty/imtag/train_pos'
import os 
for lists in os.listdir(path):
    image = os.path.join(path,lists)
    predict(image)
