#coding=utf-8
import sys  
reload(sys)  
sys.setdefaultencoding('utf8') 
import pandas as pd
reload(sys)
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'/home/dl/Downloads/stanford-corenlp-full-2018-01-31/', lang='zh')#默认处理英文，加上zh代表处理中文。

data = open('/home/dl/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/chinese/mid_fip.txt','r')#read text data
data = data.readlines()

f = open('/home/dl/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/syntax_tree/mid_fip_tree.txt','w')#output result of phrase analysis
for index,i in enumerate(data):
    print index
    s = nlp.parse(i)
    s = s.split()
    for j in s:
        f.write(j+' ')
    f.write('\n')
f.close()

nlp.close()
