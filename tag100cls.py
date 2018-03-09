#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import sys

cls=[]
clsid=[]

f1 = open('tags_905.txt','rb')
lines1 = f1.readlines()
for line in lines1:
	line1arrays = line.split('\r')

for line1array in line1arrays:
	line1array=line1array.split('\t')
	cls.append(line1array[0])
	clsid.append(line1array[2])
# print cls[904]
# print clsid[904]

cls2=[]
f2 = open('tags_100.txt','rb')
lines2 = f2.readlines()
for line in lines2:
	clsid2list = line.split('\r')
# print line2arrays[0:3]

cls2=[]
for clsid2 in clsid2list:
	cls2.append(cls[clsid.index(clsid2)])
# print clsid2list[50]
# print cls2[50]

f=open('id_100.txt','w')
for i in range(0,100):
	f.write(cls2[i]+'\n')