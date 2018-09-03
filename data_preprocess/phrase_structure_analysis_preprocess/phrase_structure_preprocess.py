#coding=utf-8
'''
Use coreNLP to phrase structure analysis for short text
'''
import sys
reload(sys)  
sys.setdefaultencoding('utf8') 
import pandas as pd
reload(sys)
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'.../Downloads/stanford-corenlp-full-2018-01-31/', lang='zh')#zh indicates that the processed language is Chinese.

data = open('./data/data_fip.txt','r')#read text data
data = data.readlines()

f = open('./data/neu_fip_phrase_structure.txt','w')#output result of phrase analysis
for index,i in enumerate(data):
    print index
    s = nlp.parse(i)
    s = s.split()
    for j in s:
        f.write(j+' ')
    f.write('\n')
f.close()

nlp.close()
