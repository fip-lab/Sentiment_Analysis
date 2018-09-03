# coding=utf-8
'''
The purpose of this code is to combine the lexical analysis results of coreNLP and Jieba.
'''
import numpy as np
from os.path import join
import pandas as pd

root_path = './data/....'

jieba = './cixing_jieba_xxxx.txt'
corenlp = './cixing_corenlp_xxxx.txt'
table = './binary_sequence_table.txt'
jieba = open(join(root_path,jieba),'r')
corenlp = open(join(root_path,corenlp),'r')
jieba = jieba.readlines()
corenlp = corenlp.readlines()
table_dict = {}
n = 0
for j,i in enumerate(jieba):
    a = i.split()
    for x in a:
	if x not in table_dict:
            table_dict[x] = n
            n += 1
	    print x,n
for i in corenlp:
    a = i.split()
    for x in a:
        if x not in table_dict:
            table_dict[x] = n
            n += 1
            print x,n
print table_dict
all_vec = []
for i,j in zip(jieba,corenlp):
    i = set(i.split())
    j = set(j.split())
    vec = [0]*93
    for x in i:
        vec[table_dict[x]] = 1
    for y in j:
        vec[table_dict[x]] = 1
    all_vec.append(vec)
df = pd.DataFrame(all_vec)
df.to_csv(join(root_path,'./c2vec_xxxx.csv'),encoding='utf-8',index=None,header=None)#save results of binarization sequnence
print 'end'
