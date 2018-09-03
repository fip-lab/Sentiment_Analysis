# coding=utf-8
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')   
import pandas as pd
import jieba
import jieba.posseg

pos=pd.read_excel('/home/dl/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/chinese/pos_fip.xls',header=None,index=None,encoding='utf-8')
neg=pd.read_excel('/home/dl/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/chinese/neg_fip.xls',header=None,index=None,encoding='utf-8')
mid=pd.read_excel('/home/dl/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/chinese/mid_fip.xls',header=None,index=None,encoding='utf-8')
word = pd.concat([pos,neg,mid],axis=0)
word.columns = ['word']
print(word.head())
f = open('/home/dl/main_code/reproduce/CNN/cnn-text-classification-tf-master/cnn-text-classification-tf-master/data/rt-polaritydata/segment.txt','w')

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
