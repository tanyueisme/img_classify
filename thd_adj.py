# -*- coding: utf-8 -*
import numpy as np
import os

nclasses=-1
threshold_min=0.01
rate_precrecall_max=1.5
precision_target=0.8

synset_list=[]
synset_highprec_set=set()
tagid_dict=dict()

id100=[]

def mapcal(score_file, gt_file, result_file,vis=True, prt=True):
    gt_mat=loadGT(gt_file)
    all_scores = np.loadtxt(score_file)
    ap_array=np.zeros((nclasses,), dtype=np.float32)
    threshold_array=np.zeros((nclasses,), dtype=np.float32)
    prec_array=np.zeros((nclasses,), dtype=np.float32)
    rec_array=np.zeros((nclasses,), dtype=np.float32)
    gt_sum=np.zeros((nclasses,), dtype=np.int32)

    # 定义array
    p1229=np.zeros((nclasses,), dtype=np.float32)
    p1123=np.zeros((nclasses,), dtype=np.float32)
    r1229=np.zeros((nclasses,), dtype=np.float32)
    r1123=np.zeros((nclasses,), dtype=np.float32)
    th1229=np.zeros((nclasses,), dtype=np.float32)
    # print type(p1229)
    # print nclasses
    for line in open('relatedata.txt'):
        linesplit=line.split('\r')
    for line in linesplit[1:]:
        linelist=line.split('\t')
        p1229[int(linelist[0])]=linelist[3]
        p1123[int(linelist[0])]=linelist[4]
        r1229[int(linelist[0])]=linelist[5]
        r1123[int(linelist[0])]=linelist[6]
        th1229[int(linelist[0])]=linelist[7]
    # print p1229[515]
    # print th1229[287]
    # 临界阈值下，1229_add的准确率、召回率，max：p1229<p1123的最大临界值；min：p1229>p1123的最小临界值
    p1229_max=np.zeros((nclasses,),dtype=np.float32)
    p1229_min=np.zeros((nclasses,),dtype=np.float32)
    p1229_new=np.zeros((nclasses,), dtype=np.float32)
    
    r1229_max=np.zeros((nclasses,),dtype=np.float32)
    r1229_min=np.zeros((nclasses,),dtype=np.float32)
    r1229_new=np.zeros((nclasses,), dtype=np.float32)
    
    th1229_min=np.zeros((nclasses,),dtype=np.float32)
    print "len(th1229_min)::::",len(th1229_min)
    th1229_max=np.zeros((nclasses,), dtype=np.float32)
    th1229_new=np.zeros((nclasses,), dtype=np.float32)

    for cls in range(0,nclasses):
        if p1229[cls]<p1123[cls]:
            (th1229_max[cls],th1229_min[cls])=findthm(gt_mat[:,cls],cls,all_scores[:,cls],p1229[cls],p1123[cls],th1229[cls])
            print "cls:",cls,"th1229_max:",th1229_max[cls],"th1229_min:",th1229_min[cls]
            (p1229_max[cls],r1229_max[cls])=get_rp(gt_mat[:,cls],cls,all_scores[:,cls],th1229_max[cls])
            print "p1229_max:",p1229_max[cls],"r1229_max:",r1229_max[cls]
            (p1229_min[cls],r1229_min[cls])=get_rp(gt_mat[:,cls],cls,all_scores[:,cls],th1229_min[cls])
            print "p1229_min:",p1229_min[cls],"r1229_min:",r1229_min[cls]

            if (p1229_min[cls]-p1123[cls]) <= 0.05: # p上升小于0.05
                th1229_new[cls]=th1229_min[cls]
                p1229_new[cls]=p1229_min[cls]
                r1229_new[cls]=r1229_min[cls]
            elif (p1229_min[cls]-p1123[cls]) <= 0.1 and (r1123[cls]-r1229_min[cls]) <= 0.1: # p上升小于0.1且r下降小于0.1
                th1229_new[cls]=th1229_min[cls]
                p1229_new[cls]=p1229_min[cls]
                r1229_new[cls]=r1229_min[cls]
            else:
                th1229_new[cls]=th1229_max[cls]
                p1229_new[cls]=p1229_max[cls]
                r1229_new[cls]=r1229_max[cls]
            (ap_array[cls],threshold_array[cls],prec_array[cls],rec_array[cls])=apcal(gt_mat[:,cls], cls, all_scores[:,cls])
        else:
            th1229_new[cls]=th1229[cls]
            p1229_new[cls]=p1229[cls]
            r1229_new[cls]=r1229[cls]
	
    file_th=open('newdata.txt','w')
    for cls in range(0,nclasses):
        file_th.write('%s\t%.6f\t%.3f\t%.3f\n'% (tagid_dict[synset_list[cls]],th1229_new[cls],p1229_new[cls],r1229_new[cls]))
    file_th.close()

    gt_sum[cls]=sum(gt_mat[:,cls])
    
    meanap=ap_array.mean()
    if vis:
        print 'map = {:f}'.format(meanap)

    if prt:
        fid=open(result_file,'w')
        # fid.write('synset\tagid\n_img\tap\tthresh\tprec\trec\n')
        for cls in range(nclasses):
            if str(cls) in id100:
                fid.write('%s\t%s\t%d\t%.3f\t%.6f\t%.3f\t%.3f\n'% (synset_list[cls],tagid_dict[synset_list[cls]],gt_sum[cls],ap_array[cls],threshold_array[cls],prec_array[cls],rec_array[cls]))
        fid.close()
    return meanap
    
def loadGT(gt_file):
    with open(gt_file) as fid:
        lines=fid.readlines()
    gt_mat=np.zeros((len(lines),nclasses),dtype=bool)
    count=0
    for line in lines:
        cls_list=line.strip().split('\t')[2:]
        for cls in cls_list:
            gt_mat[count,int(cls)]=True
        count=count+1
    return gt_mat

def findthm(gt,cls,scores,p1229,p1123,th1229):
    p_tmp=p1229
    th_tmp=float(th1229)
    while(p_tmp<p1123):
        th_tmp=th_tmp+0.005
        (p_tmp,r_tmp)=get_rp(gt,cls,scores,th_tmp)
    th1229_min=th_tmp
    
    # 改之前
    # th_tmp=th1229_min
    # while(th_tmp>th1229):
    #     th_tmp=th_tmp-0.001
    # th1229_max=th_tmp

    # 改之后
    while(p_tmp>p1123):
        th_tmp=th_tmp-0.001
        (p_tmp.r_tmp)=get_rp(gt,cls,scores,th_tmp)
    th1229_max=th_tmp
    
    return th1229_max,th1229_min

def get_rp(gt, cls, scores,th_tmp):
    # scores=np.loadtxt(score_file,usecols=(cls,))
    if sum(gt)==0:
        return 0,0

    # 对scores进行从小到大的排序，输出为排序后的索引序列
    # [::-1]为逆排序
    # si中存的是当前被检索出来认为是当前类的图片
    si=scores.argsort()[::-1]
    flag=1
    i=0
    # print "th_tmp===",th_tmp
    while(flag):
        if scores[si[i]]<th_tmp:
            flag=0
        else:
            i=i+1
    
    tp=(gt[si[0:i]]>0).astype('Float32')
    fp=(gt[si[0:i]]<=0).astype('Float32')
    
    fp=np.sum(fp)
    tp=np.sum(tp)
    # rec和prec是两个数组，rec小prec大，rec大prec小
    rec=tp/np.sum(gt>0)
    prec=tp/(fp+tp)
    return (prec,rec)
    

def apcal(gt, cls, scores):
    # scores=np.loadtxt(score_file,usecols=(cls,))
    if sum(gt)==0:
	return 0,0,0,0

    if str(cls) not in id100:
        return 0,0,0,0

    # 对scores进行从小到大的排序，输出为排序后的索引序列
    # [::-1]为逆排序
    # si中存的是当前被检索出来认为是当前类的图片
    si=scores.argsort()[::-1]

    tp=(gt[si]>0).astype('Float32')
    fp=(gt[si]<=0).astype('Float32')
    
    # cumsum计算累加值
    fp=np.cumsum(fp)
    tp=np.cumsum(tp)
    # rec和prec是两个数组，rec小prec大，rec大prec小
    rec=tp/np.sum(gt>0)
    prec=tp/(fp+tp)

    ap=0
    for t in range(0,11,1):
        tt=t*0.1
        p_ind=prec[rec>=tt]
        if len(p_ind)>0:
            p=np.max(p_ind)
        else:
            p=0.0
        ap=ap+p/11

    scores.sort()

    #use the precision-recall curve cross with y=x line position
    bestind2=np.where(prec<rec)[0][0]
    if synset_list[cls] in synset_highprec_set:
        bestind2_temp=np.where(prec<precision_target)[0][0]
	if prec[bestind2_temp]>prec[bestind2]:
	    bestind2=bestind2_temp
	if prec[bestind2]>=rate_precrecall_max*rec[bestind2]:
	    bestind2=np.where(prec<rate_precrecall_max*rec)[0][0]
    threshold=scores[::-1][bestind2]
    bestprec=prec[bestind2]
    bestrec=rec[bestind2]

    return (ap,threshold,bestprec,bestrec)

if __name__ == '__main__':
    # for line in open('id_100.txt'):
    #     id100.append(line[:-1])
    # print id100

    for line in open('synset_905.txt'):
	synset_list.append(line.strip())
    nclasses=len(synset_list)
    for line in open('synset_highprec.txt'):
	synset_highprec_set.add(line.strip())
    for line in open('synset_tagid_905.txt'):
	linesplit=line.strip().split('\t')
	tagid_dict[linesplit[1]]=linesplit[0]
    mapcal('1229_add/result_test_24268.txt.1229_add','textdata/imListTest_24268.txt','1229_add/map_testval_24268.txt')
