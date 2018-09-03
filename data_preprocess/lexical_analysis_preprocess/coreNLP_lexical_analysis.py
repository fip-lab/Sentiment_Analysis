# coding=utf-8
'''
Use coreNLP to lexical analysis for short text
'''
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')   
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'/home/dl/Downloads/stanford-corenlp-full-2018-01-31/', lang='zh')
pos=pd.read_excel('./data/pos.xls',header=None,index=None,encoding='utf-8')
neg=pd.read_excel('./data/neg.xls',header=None,index=None,encoding='utf-8')
neu=pd.read_excel('./data/neu.xls',header=None,index=None,encoding='utf-8')
word = pd.concat([pos,neg,neu],axis=0)
word.columns = ['word']
print(word.head())
f = open('./data/cixing_corenlp_text.txt','w') #save results of lexical analysis

for n,i in enumerate(word['word']):
    cut0 = nlp.pos_tag(i)
    for j in cut0:
        c = j[1]
        f.write(c+' ')
    f.write('\n')
    if n%20 == 0:
        print(n)
f.close()
nlp.close()
