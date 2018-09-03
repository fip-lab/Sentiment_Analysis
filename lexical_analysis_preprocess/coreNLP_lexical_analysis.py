# coding=utf-8
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')   
import pandas as pd
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'/home/dl/Downloads/stanford-corenlp-full-2018-01-31/', lang='zh')
pos=pd.read_excel('/home/dl/main_code/Sentiment-Analysis-master/code/pre_process_data/data/long_text_chinese/fashion.xls',header=None,index=None,encoding='utf-8')
neg=pd.read_excel('/home/dl/main_code/Sentiment-Analysis-master/code/pre_process_data/data/long_text_chinese/financial.xls',header=None,index=None,encoding='utf-8')
mid=pd.read_excel('/home/dl/main_code/Sentiment-Analysis-master/code/pre_process_data/data/long_text_chinese/political.xls',header=None,index=None,encoding='utf-8')
word = pd.concat([pos,neg,mid],axis=0)
word.columns = ['word']
print(word.head())
f = open('/home/dl/main_code/Sentiment-Analysis-master/code/pre_process_data/data/long_text_chinese/93demension/cixing_corenlp_long_chinese_text.txt','w')

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
