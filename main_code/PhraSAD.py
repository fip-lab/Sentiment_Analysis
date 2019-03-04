# -*- coding: utf-8 -*-
'''
Sample code for 
Combining Phrase Structure and Attention Mechanism with Deep Neural Network for Chinese Short Text Sentiment Analysis
Much of the code is modified from
https://github.com/BUPTLdy/Sentiment-Analysis(for lstmNet classes)
'''
import os  
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import yaml
import sys
import tensorflow as tf
reload(sys)
sys.setdefaultencoding('utf8')
from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml
from keras.layers import Bidirectional, Convolution1D, MaxPool1D, Input, Flatten, concatenate, BatchNormalization, GRU, merge
from keras import regularizers
from keras.callbacks import TensorBoard, EarlyStopping
import keras
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras import optimizers	
from Attention_keras import Attention

np.random.seed(1337)  # For Reproducibility
import jieba
import pandas as pd
import sys
import xlrd
from pypinyin import pinyin, lazy_pinyin, Style
sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 200
maxlen = 100
n_iterations = 30  # ideally more..
n_exposures = 5
window_size = 5
batch_size = 128
n_epoch = 20
input_length = 100
cpu_count = multiprocessing.cpu_count()


#加载训练文件
def loadfile():
#origin
    pos=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/chinese/pos.xls',header=None,index=None)
    neg=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/chinese/neg.xls',header=None,index=None)
    neu=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/chinese/neu.xls',header=None,index=None)  #3-class

    combined=np.concatenate((pos[0], neg[0], neu[0]))     #word

    y = np_utils.to_categorical(np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int), np.ones(len(neu),dtype=int)*2)),num_classes=3)   

    return combined,y

def loadfile_2():
#93_demension
    pos_2=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/93demension/pos.xls' ,header=None,index=None)
    neg_2=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/93demension/neg.xls' ,header=None,index=None)
    neu_2=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/93demension/neu.xls' ,header=None,index=None)
    
    combined_2 = pd.concat([pos_2, neg_2, neu_2],axis=0)  #lexical analysis vector       
    print combined_2
    y_2 = np_utils.to_categorical(np.concatenate((np.ones(len(pos_2),dtype=int), np.zeros(len(neg_2),dtype=int), np.ones(len(neu_2),dtype=int)*2)),num_classes=3)   

    return combined_2,y_2


def loadfile_4():
#origin
    pos_4=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/syntax_tree/pos.xls',header=None,index=None)
    neg_4=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/syntax_tree/neg.xls',header=None,index=None)
    neu_4=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/syntax_tree/neu.xls',header=None,index=None)  

    combined_4 = np.concatenate((pos_4[0], neg_4[0], neu_4[0]))     #syntactic analysis vector
    print combined_4
    y_4 = np_utils.to_categorical(np.concatenate((np.ones(len(pos_4),dtype=int), np.zeros(len(neg_4),dtype=int), np.ones(len(neu_4),dtype=int)*2)),num_classes=3)   #3-class

    return combined_4, y_4

#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    #print text
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text



#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过5的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过5的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于5的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print 'No data provided...'


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引_4
def create_dictionaries_4(model_4=None,
                        combined_4=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined_4 is not None) and (model_4 is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model_4.wv.vocab.keys(),
                            allow_update=True)
        w2indx_4 = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec_4 = {word_4: model_4[word_4] for word_4 in w2indx_4.keys()}#所有频数超过10的词语的词向量

        def parse_dataset_4(combined_4):
            ''' Words become integers
            '''
            data_4=[]
            for sentence in combined_4:
                new_txt = []
                for word_4 in sentence:
                    try:
                        new_txt.append(w2indx_4[word_4])
                    except:
                        new_txt.append(0)
                data_4.append(new_txt)
            return data_4
        combined_4=parse_dataset_4(combined_4)
        combined_4= sequence.pad_sequences(combined_4, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx_4, w2vec_4,combined_4
    else:
        print 'No data provided...'



#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined,total_examples=model.corpus_count, epochs=model.epochs)
    model.save('/home/fip/main_code/Sentiment-Analysis-master/code/lstm_data/long_chinese/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引_4

def word2vec_train_4(combined_4):

    model_4 = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model_4.build_vocab(combined_4)
    model_4.train(combined_4,total_examples=model_4.corpus_count, epochs=model_4.epochs)
    model_4.save('/home/fip/main_code/Sentiment-Analysis-master/code/lstm_data/long_chinese/Word2vec_model_3.pkl')
    index_dict_4, word_vectors_4,combined_4 = create_dictionaries_4(model_4=model_4,combined_4=combined_4)
    return   index_dict_4, word_vectors_4,combined_4


def get_data(index_dict, index_dict_4, word_vectors, word_vectors_4, combined, combined_2, combined_4, y):

#origin
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于5的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    combined = pd.DataFrame(combined)
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    #print 'hhhhhhh',combined.shape,x_train.index
    print x_train.shape,y_train.shape,type(x_train),x_train

#93_demension
    x_train_2 = combined_2.iloc[x_train.index]
    x_test_2 = combined_2.iloc[x_test.index]
    x_train_2 = x_train_2.values
    x_train_2 = x_train_2.reshape(x_train_2.shape[0],1,93)
    x_test_2 = x_test_2.values
    x_test_2 = x_test_2.reshape(x_test_2.shape[0],1,93)


#syntax_tree
    n_symbols_4 = len(index_dict_4) + 1
    embedding_weights_4 = np.zeros((n_symbols_4,vocab_dim))
    for word_4, index in index_dict_4.items():
        embedding_weights_4[index, :] = word_vectors_4[word_4]
    combined_4 = pd.DataFrame(combined_4)
    #x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(combined_4, y, test_size=0.2)
    x_train_4 = combined_4.iloc[x_train.index]
    x_test_4 = combined_4.iloc[x_test.index]
    #print 'syntax_tree:', x_train_4.shape, x_test_4.shape,x_train_4,'end!'

#all
    return n_symbols, n_symbols_4, embedding_weights, embedding_weights_4, x_train, y_train, x_test, y_test, x_train_2, x_test_2,x_train_4, x_test_4


##定义网络结构
def train_lstm(n_symbols, n_symbols_4, embedding_weights, embedding_weights_4, x_train, y_train, x_test, y_test, x_train_2, x_test_2, x_train_4, x_test_4):
    print 'Defining a Simple Keras Model...'

#1Chinese
    main_input=Input(shape=(100,), dtype='float32')
    embed = Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=False,
                        weights=[embedding_weights],
                        input_length=input_length)(main_input)
    c1 = Attention(64,64)([embed, embed, embed])
    c1 = Convolution1D(64, 2, padding='same', strides = 2,activation='relu')(c1)
    #c1 = Attention(64,64)([c1,c1,c1])
    c1 = Dropout(0.5)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Bidirectional(LSTM(output_dim=64))(c1)



  
  
#2simple_syntax_analysis
    main_input_2 = Input(shape=(1,93),dtype='float32')
    #c2 = Attention(64,64)([main_input_2, main_input_2, main_input_2])
    c2 = Convolution1D(64,
                       1,
                       padding='same', 
                       strides = 1,
                       input_shape = (1,93))(main_input_2)
    c2 = Dropout(0.5)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Bidirectional(LSTM(output_dim=64))(c2)



#4syntax_tree

    main_input_4 = Input(shape=(100,), dtype='float32')
    embed_4 = Embedding(output_dim=vocab_dim,
                      input_dim=n_symbols_4,
                      mask_zero=False,
                      weights=[embedding_weights_4],
                      input_length=input_length)(main_input_4)
    #c4 = Attention(64,64)([embed_4, embed_4, embed_4])
    c4 = Convolution1D(64, 2, padding='same', strides = 2, activation='relu')(embed_4)
    #c4 = Attention(64,128)([c4,c4,c4])
    c4 = Dropout(0.5)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Bidirectional(LSTM(output_dim=64))(c4)


    #add = keras.layers.Add()([c1,c3])
    #add_2 = keras.layers.Add()([c2,c4])


    #dot = keras.layers.Dot(1, normalize=False)([c1,c2])
    dot_2 = keras.layers.Dot(1, normalize=False)([c2,c4])

    
    con = keras.layers.concatenate([c1,dot_2])#axis=-1
    main_output = Dense(3, activation='softmax')(con)
    model = Model(inputs = [main_input, main_input_2,  main_input_4], output = main_output)

    print 'Compiling the Model...'

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['mae', 'accuracy'])
    model.summary()

    print "Train..."
    meta_file = "metadata.tsv"
    tensorboard = TensorBoard(log_dir='/home/fip/main_code/tensorboard/log', 
                              histogram_freq=1,
                              write_graph=True,
                              embeddings_metadata = meta_file)
    sess = tf.InteractiveSession() 
    sess.run(tf.global_variables_initializer()) 
    saver = tf.train.Saver() 
    LOG_DIR = '/home/fip/main_code/tensorboard/log'
    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history=model.fit([x_train,x_train_2,x_train_4],
                     y_train, 
                     batch_size=batch_size,
                     nb_epoch=n_epoch,verbose=1,
                     validation_data=([x_test,x_test_2,x_test_4], y_test),
                     callbacks=[tensorboard]
                     )

    print "Evaluate..."
    score = model.evaluate([x_test, x_test_2, x_test_4], y_test,
                                batch_size=batch_size)
    print x_test, y_test


    yaml_string = model.to_yaml()
    with open('/home/fip/main_code/tensorboard/log/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('/home/fip/main_code/tensorboard/log/lstm.h5')
    print 'Test score:', score


#-----------------------------------------------------------------------------------------------------------------------------------------------------
    f = open('/home/fip/main_code/Sentiment-Analysis-master/backup/ACL/2-class_wrong_data/real_label','w')  #obtained manual real_label file
    for i in y_test:
        print>>f,"{0}".format(i)
    f.close()
#-----------------------------------------------------------------------------------------------------------------------------------------------------
    f = open('/home/fip/main_code/Sentiment-Analysis-master/backup/ACL/2-class_wrong_data/test_data','w')
    pos=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/chinese/pos.xls',header=None,index=None)
    neg=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/chinese/neg.xls',header=None,index=None)
    neu=pd.read_excel('/home/fip/main_code/Sentiment-Analysis-master/code/pre_process_data/data/fip/chinese/neu.xls',header=None,index=None)
    data = np.concatenate((pos[0], neg[0], neu[0]))
    for i in data[x_test.index]:
        print>>f,"{0}".format(i)
    f.close()
    result = model.predict([x_test, x_test_2, x_test_4], batch_size = batch_size)
#------------------------------------------------------------------------------------------------------------------------------------------------------
    f = open('/home/fip/main_code/Sentiment-Analysis-master/backup/ACL/2-class_wrong_data/test_data_index_label','w') #obtained test dataset file
    for i in x_test.index:
        print>>f,"{0}".format(i)
    f.close()
#-------------------------------------------------------------------------------------------------------------------------------------------------------

    f = open('/home/fip/main_code/Sentiment-Analysis-master/backup/ACL/2-class_wrong_data/test_data_label','w') #obtained program judge label file
    for i in result:
        print>>f,"{0}".format(i)
    f.close()

#---------------------------------------------------------------------------------------------------------------------------------------------------------

    yaml_string = model.to_yaml()
    with open('/home/fip/main_code/Sentiment-Analysis-master/code/lstm_data/fip/2-classicial/2classicial.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('/home/fip/main_code/Sentiment-Analysis-master/code/lstm_data/fip/2-classicial/2classicial.h5')

#----------------------------------------------------------------------------------------------------------------------------------------------------------

    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["acc"], color="g", label="Train")
    plt.plot(history.history["val_acc"], color="b", label="Test")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="g", label="Train")
    plt.plot(history.history["val_loss"], color="b", label="Test")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()

#训练模型，并保存
def train():
    print 'Loading Data...'
    combined,y=loadfile()
    combined_2,y_2=loadfile_2()
    combined_4,y_4=loadfile_4()
    print len(combined),len(y),len(combined_2),len(y_2),len(combined_4),len(y_4)
    print 'Tokenising...'
    combined = tokenizer(combined)
    print 'Training a Word2vec model...'
    index_dict, word_vectors,combined=word2vec_train(combined)
    print 'Training a Word2vec model_4...'
    combined_4 = [t.split() for t in combined_4]
    index_dict_4, word_vectors_4, combined_4 = word2vec_train_4(combined_4)
    print 'Setting up Arrays for Keras Embedding Layer...'
    n_symbols,n_symbols_4, embedding_weights, embedding_weights_4, x_train, y_train, x_test, y_test, x_train_2, x_test_2, x_train_4, x_test_4 = get_data(index_dict, index_dict_4, word_vectors, word_vectors_4, combined, combined_2, combined_4, y)
    #x_train_2,y_train_2,x_test_2,y_test_2=get_data_2(combined_2,y_2)
    #print x_train.shape,y_train.shape
    train_lstm(n_symbols, n_symbols_4, embedding_weights, embedding_weights_4, x_train, y_train, x_test, y_test, x_train_2, x_test_2, x_train_4, x_test_4)


def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('/home/fip/main_code/Sentiment-Analysis-master/code/lstm_data/long_chinese/Word2vec_model.pkl')
    model_3=Word2Vec.load('/home/fip/main_code/Sentiment-Analysis-master/code/lstm_data/long_chinese/Word2vec_model_3.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

def lstm_predict(string):
    #print 'loading model......'
    with open('/home/fip/main_code/Sentiment-Analysis-master/code/lstm_data/long_chinese/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    #print 'loading weights......'
    model.load_weights('/home/fip/main_code/Sentiment-Analysis-master/code/lstm_data/long_chinese/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    #print data
    result=model.predict_classes(data)
    #print result
'''
    if result[0]==1:
        print string,' positive'
    if result[0]==0:
        print string,' negative'
    if result[0]==2:
        print string,'natural'  #3-class
'''

if __name__=='__main__':
    train()
    '''
    f=open('/home/dl/main_code/Sentiment-Analysis-master/code/test_data/test.txt')
    while 1:
        string=f.readline()
        if not string:
            break
        lstm_predict(string)
    '''