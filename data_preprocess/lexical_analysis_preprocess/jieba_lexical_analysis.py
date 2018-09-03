# coding=utf-8
'''
Use Jieba to lexical analysis for short text
'''

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')   
import pandas as pd
import jieba
import jieba.posseg

pos=pd.read_excel('./data/pos_fip.xls',header=None,index=None,encoding='utf-8')
neg=pd.read_excel('./data/neg_fip.xls',header=None,index=None,encoding='utf-8')
neu=pd.read_excel('./data/neu_fip.xls',header=None,index=None,encoding='utf-8')
word = pd.concat([pos,neg,neu],axis=0)
word.columns = ['word']
print(word.head())
f = open('./data/jieba_lexical_analysis.txt','w') #save results of lexical analysis

for n,i in enumerate(word['word']):
    cut = jieba.cut(i)
    for j in cut:
        c = j.flag
        f.write(c+' ')
    f.write('\n')
    if n%20 == 0:
        print(n)
f.close()
nlp.close()
