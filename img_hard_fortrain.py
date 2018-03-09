# -*- coding: utf-8 -*
import numpy as np
import os

nclasses=905
threshold_min=0.01
rate_precrecall_max=1.5
precision_target=0.8

synset_list=[]
synset_highprec_set=set()
tagid_dict=dict()
# here1
tagid100=[]
# here2

def mapcal(score_file, gt_file, vis=True, prt=False):
    gt_mat=loadGT(gt_file)
    all_scores = np.loadtxt(score_file)
    ap_array=np.zeros((nclasses,), dtype=np.float32)
    threshold_array=np.zeros((nclasses,), dtype=np.float32)
    prec_array=np.zeros((nclasses,), dtype=np.float32)
    rec_array=np.zeros((nclasses,), dtype=np.float32)
    gt_sum=np.zeros((nclasses,), dtype=np.int32)
    
    for cls in range(0,nclasses):
        if cls==129:
            apcal(gt_mat, cls, all_scores[:,cls])
	

def loadGT(gt_file):
    with open(gt_file) as fid:
        lines=fid.readlines()
    gt_mat=np.zeros(len(lines))
    count=0
    for line in lines:
        cls_list=line.strip().split('\t')[1]
        gt_mat[count]=cls_list
        count=count+1
    return gt_mat

def apcal(gt, cls, scores):
    # scores=np.loadtxt(score_file,usecols=(cls,))
    if sum(gt)==0:
        return 0,0,0,0

    # 对scores进行从小到大的排序，输出为排序后的索引序列
    # [::-1]为逆排序
    # si中存的是当前被检索出来认为是当前类的图片
    f=open('/home/tanyue/train/hard.txt','w')
    # print "===",len(scores)
    si=scores.argsort()[::-1]
    for idx in si[0:50000]:
        if gt[idx] != 129:
            # print idx,"is not dog",scores[idx]
            f.write(str(idx)+'\n')
        if gt[idx] == 129:
            print idx,"is dog",scores[idx]

if __name__ == '__main__':
    mapcal('resulttrain.txt','imListTrain.txt')