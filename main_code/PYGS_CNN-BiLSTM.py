# -*- coding: utf-8 -*-
'''
Sample code for 
Combining PinYin and Grammatical Structure with Deep Neural Networks for Chinese Short Text Sentiment Analysis

Much of the code is modified from
https://github.com/BUPTLdy/Sentiment-Analysis(for lstmNet classes)
'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import yaml
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from sklearn.cross_validation import train_test_split
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
from keras.layers import Bidirectional, Convolution1D, MaxPool1D, Input, Flatten, concatenate, BatchNormalization, GRU, Merge
from keras import regularizers
from keras.callbacks import TensorBoard, EarlyStopping
import keras
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras import optimizers
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
n_iterations = 30
n_exposures = 5
window_size = 10
batch_size = 64
n_epoch = 30
input_length = 100
cpu_count = multiprocessing.cpu_count()

'''
Load file
These files are obtained after preprocess, include text, lexical code, phrase structure.
'''
def loadfile():
#text
    pos=pd.read_excel('./data/pos_fip.xls',header=None,index=None)
    neg=pd.read_excel('./data/neg_fip.xls',header=None,index=None)
    neu=pd.read_excel('./data/neu_fip.xls',header=None,index=None)

    combined=np.concatenate((pos[0], neg[0], neu[0]))
    y = np_utils.to_categorical(np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int), np.ones(len(neu),dtype=int)*2)),num_classes=3)   #3-class

    return combined,y

def loadfile_2():
#lexical_analysis
    pos_2=pd.read_excel('./data/pos_lexical(fip).xls' ,header=None,index=None)
    neg_2=pd.read_excel('./data/neg_lexical(fip).xls' ,header=None,index=None)
    neu_2=pd.read_excel('./data/neu_lexical(fip).xls' ,header=None,index=None)
    
    combined_2 = pd.concat([pos_2, neg_2, neu_2],axis=0)
    print combined_2
    y_2 = np_utils.to_categorical(np.concatenate((np.ones(len(pos_2),dtype=int), np.zeros(len(neg_2),dtype=int), np.ones(len(neu_2),dtype=int)*2)),num_classes=3)   #3-class

    return combined_2,y_2

def loadfile_3():
#Chinese_Pinyin
    pos_3=pd.read_excel('./data/pos_fip.xls',header=None,index=None)
    neg_3=pd.read_excel('./data/neg_fip.xls',header=None,index=None)
    neu_3=pd.read_excel('./data/neu_fip.xls',header=None,index=None)

    combined_3 = pd.concat([pos_3,neg_3,neu_3],axis=0)
    
    combined_3.columns = ['data']
    l = []
    for i in combined_3['data']:
        s =''
        c=pinyin(i) #Convert Chinese to Pinin
        for j in c:
            s += j[0]
	l.append(s)
    combined_3 = np.array(l)
    print(len(combined_3))
    print(combined_3)

    y_3 = np_utils.to_categorical(np.concatenate((np.ones(len(pos_3),dtype=int), np.zeros(len(neg_3),dtype=int), np.ones(len(neu_3),dtype=int)*2)),num_classes=3)   #3-class

    return combined_3,y_3

def loadfile_4():
#Phrase_structure analysis
    pos_4=pd.read_excel('./data/pos_phrase_structure(fip).xls',header=None,index=None)
    neg_4=pd.read_excel('./data/neg_phrase_structure(fip).xls',header=None,index=None)
    neu_4=pd.read_excel('./data/neu_phrase_structure(fip).xls',header=None,index=None)   #3-class

    combined_4 = np.concatenate((pos_4[0], neg_4[0], neu_4[0]))
    y_4 = np_utils.to_categorical(np.concatenate((np.ones(len(pos_4),dtype=int), np.zeros(len(neg_4),dtype=int), np.ones(len(neu_4),dtype=int)*2)),num_classes=3)   #3-class

    return combined_4, y_4

#Segmentation for sentence
def tokenizer(text):
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

# Create vocabulary
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
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

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
        combined= sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec,combined
    else:
        print 'No data provided...'

def create_dictionaries_3(model_3=None,
                        combined_3=None):

    if (combined_3 is not None) and (model_3 is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model_3.wv.vocab.keys(),
                            allow_update=True)
        w2indx_3 = {v: k+1 for k, v in gensim_dict.items()}#Index of all words with a frequency greater than 5.
        w2vec_3 = {word_3: model_3[word_3] for word_3 in w2indx_3.keys()}#All word vectors with a frequency greater than 5.

        def parse_dataset_3(combined_3):

            data_3=[]
            for sentence in combined_3:
                new_txt = []
                for word_3 in sentence:
                    try:
                        new_txt.append(w2indx_3[word_3])
                    except:
                        new_txt.append(0)
                data_3.append(new_txt)
            return data_3
        combined_3=parse_dataset_3(combined_3)
        combined_3= sequence.pad_sequences(combined_3, maxlen=maxlen)#Words with a frequency less than 5 in the sentence, index 0.
        return w2indx_3, w2vec_3,combined_3
    else:
        print 'No data provided...'

def create_dictionaries_4(model_4=None,
                        combined_4=None):

    if (combined_4 is not None) and (model_4 is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model_4.wv.vocab.keys(),
                            allow_update=True)
        w2indx_4 = {v: k+1 for k, v in gensim_dict.items()}
        w2vec_4 = {word_4: model_4[word_4] for word_4 in w2indx_4.keys()}

        def parse_dataset_4(combined_4):

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
        combined_4= sequence.pad_sequences(combined_4, maxlen=maxlen)
        return w2indx_4, w2vec_4,combined_4
    else:
        print 'No data provided...'

def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined,total_examples=model.corpus_count, epochs=model.epochs)
    model.save('./model/fip/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

def word2vec_train_3(combined_3):

    model_3 = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model_3.build_vocab(combined_3)
    model_3.train(combined_3,total_examples=model_3.corpus_count, epochs=model_3.epochs)
    model_3.save('./model/fip/Word2vec_model_3.pkl')
    index_dict_3, word_vectors_3,combined_3 = create_dictionaries_3(model_3=model_3,combined_3=combined_3)
    return   index_dict_3, word_vectors_3,combined_3

def word2vec_train_4(combined_4):

    model_4 = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model_4.build_vocab(combined_4)
    model_4.train(combined_4,total_examples=model_4.corpus_count, epochs=model_4.epochs)
    model_4.save('./model/fip/Word2vec_model_3.pkl')
    index_dict_4, word_vectors_4,combined_4 = create_dictionaries_4(model_4=model_4,combined_4=combined_4)
    return   index_dict_4, word_vectors_4,combined_4

def get_data(index_dict, index_dict_3, index_dict_4, word_vectors, word_vectors_3, word_vectors_4, combined, combined_2, combined_3, combined_4, y):
#text
    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():#Each word corresponds to its word vector
        embedding_weights[index, :] = word_vectors[word]
    combined = pd.DataFrame(combined)
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)  #80% data use to be training dataset, 20% data use to be test dataset.
    print x_train.shape,y_train.shape,type(x_train),x_train

#lexical analysis
    x_train_2 = combined_2.iloc[x_train.index]
    x_test_2 = combined_2.iloc[x_test.index]
    x_train_2 = x_train_2.values
    x_train_2 = x_train_2.reshape(x_train_2.shape[0],1,93)
    x_test_2 = x_test_2.values
    x_test_2 = x_test_2.reshape(x_test_2.shape[0],1,93)

#Chinese Pinyin
    n_symbols_3 = len(index_dict_3) + 1
    embedding_weights_3 = np.zeros((n_symbols_3,vocab_dim))
    for word_3, index in index_dict_3.items():
        embedding_weights_3[index, :] = word_vectors_3[word_3]
    combined_3 = pd.DataFrame(combined_3)
    x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(combined_3, y, test_size=0.2)
    x_train_3 = combined_3.iloc[x_train.index]
    x_test_3 = combined_3.iloc[x_test.index]
    print 'pinyin:', x_train_3.shape, x_test_3.shape,x_train_3,'end!'

#Phrase_structure analysis
    n_symbols_4 = len(index_dict_4) + 1
    embedding_weights_4 = np.zeros((n_symbols_4,vocab_dim))
    for word_4, index in index_dict_4.items():
        embedding_weights_4[index, :] = word_vectors_4[word_4]
    combined_4 = pd.DataFrame(combined_4)
    x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(combined_4, y, test_size=0.2)
    x_train_4 = combined_4.iloc[x_train.index]
    x_test_4 = combined_4.iloc[x_test.index]

#all
    return n_symbols, n_symbols_3, n_symbols_4, embedding_weights, embedding_weights_3, embedding_weights_4, x_train, y_train, x_test, y_test, x_train_2, x_test_2, x_train_3, x_test_3, x_train_4, x_test_4

##Defining the network structure
def train_lstm(n_symbols, n_symbols_3, n_symbols_4, embedding_weights, embedding_weights_3, embedding_weights_4, x_train, y_train, x_test, y_test, x_train_2, x_test_2, x_train_3, x_test_3, x_train_4, x_test_4):
    print 'Defining a Simple Keras Model...'

#branch of text
    main_input=Input(shape=(100,), dtype='float32')
    embed = Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=False,
                        weights=[embedding_weights],
                        input_length=input_length)(main_input)
    c1 = Convolution1D(64, 2, padding='same', strides = 2,activation='relu')(embed)
    c1 = Dropout(0.5)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Bidirectional(LSTM(output_dim=64))(c1)

#branch of lexical analysis
    main_input_2 = Input(shape=(1,93),dtype='float32')
    c2 = Convolution1D(64,
                       1,
                       padding='same', 
                       strides = 1,
                       input_shape = (1,93))(main_input_2)
    c2 = Dropout(0.5)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Bidirectional(LSTM(output_dim=64))(c2)

#branch of Chinese Pinyin
    main_input_3 = Input(shape=(100,), dtype='float32')
    
    embed_3 = Embedding(output_dim=vocab_dim,
                      input_dim=n_symbols_3,
                      mask_zero=False,
                      weights=[embedding_weights_3],
                      input_length=input_length)(main_input_3)
    
    c3 = Convolution1D(64, 5, padding='same', strides = 2, activation='relu')(embed_3)
    c3 = Dropout(0.5)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Bidirectional(LSTM(output_dim=64))(c3)

#branch of Phrase_structure analysis
    main_input_4 = Input(shape=(100,), dtype='float32')
    
    embed_4 = Embedding(output_dim=vocab_dim,
                      input_dim=n_symbols_4,
                      mask_zero=False,
                      weights=[embedding_weights_4],
                      input_length=input_length)(main_input_4)
    
    c4 = Convolution1D(64, 5, padding='same', strides = 2, activation='relu')(embed_4)
    c4 = Dropout(0.5)(c4)
    c4 = BatchNormalization()(c4)
    c4 = Bidirectional(LSTM(output_dim=64))(c4)

#Dot product
    dot_1 = keras.layers.Dot(1, normalize=False)([c1,c3])
    dot_2 = keras.layers.Dot(1, normalize=False)([c2,c4])

    con = keras.layers.concatenate([dot_1,dot_2])
    main_output = Dense(3, activation='softmax')(con)
    model = Model(inputs = [main_input, main_input_2, main_input_3, main_input_4], output = main_output)

    print 'Compiling the Model...'

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['mae', 'accuracy'])
    model.summary()

    print "Train..."
    tensorboard = TensorBoard(log_dir='/home/dl/main_code/tensorboard/log', histogram_freq=1,write_graph=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history=model.fit([x_train,x_train_2,x_train_3,x_train_4],
                     y_train, 
                     batch_size=batch_size,
                     nb_epoch=n_epoch,verbose=1,
                     validation_data=([x_test,x_test_2,x_test_3,x_test_4], y_test),
                     callbacks=[tensorboard]
                     )

    print "Evaluate..."
    score = model.evaluate([x_test, x_test_2, x_test_3, x_test_4], y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('./model/fip/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('./model/fip/lstm.h5')
    print 'Test score:', score

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

#training and save model
def train():
    print 'Loading Data...'
    combined,y=loadfile()
    combined_2,y_2=loadfile_2()
    combined_3,y_3=loadfile_3()
    combined_4,y_4=loadfile_4()
    print len(combined),len(y),len(combined_2),len(y_2),len(combined_3),len(y_3),len(combined_4),len(y_4)
    print 'Tokenising...'
    combined = tokenizer(combined)
    print 'Training a Word2vec model...'
    index_dict, word_vectors,combined=word2vec_train(combined)
    print 'Training a Word2vec model_3...'
    combined_3 = [t.split() for t in combined_3]
    index_dict_3, word_vectors_3, combined_3 = word2vec_train_3(combined_3)
    print 'Training a Word2vec model_4...'
    combined_4 = [t.split() for t in combined_4]
    index_dict_4, word_vectors_4, combined_4 = word2vec_train_4(combined_4)
    print 'Setting up Arrays for Keras Embedding Layer...'
    n_symbols, n_symbols_3, n_symbols_4, embedding_weights, embedding_weights_3, embedding_weights_4, x_train, y_train, x_test, y_test, x_train_2, x_test_2, x_train_3, x_test_3, x_train_4, x_test_4 = get_data(index_dict, index_dict_3, index_dict_4, word_vectors, word_vectors_3, word_vectors_4, combined, combined_2, combined_3, combined_4, y)
    train_lstm(n_symbols, n_symbols_3, n_symbols_4, embedding_weights, embedding_weights_3, embedding_weights_4, x_train, y_train, x_test, y_test, x_train_2, x_test_2, x_train_3, x_test_3, x_train_4, x_test_4)

def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('./model/fip/Word2vec_model.pkl')
    model_3=Word2Vec.load('./model/fip/Word2vec_model_3.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

def lstm_predict(string):
    #print 'loading model......'
    with open('lstm_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    #print 'loading weights......'
    model.load_weights('./model/fip/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    #print data
    result=model.predict_classes(data)
    #print result
    if result[0]==1:
        print string,' positive'
    if result[0]==0:
        print string,' negative'
    if result[0]==2:
        print string,' neutral'

if __name__=='__main__':
    train()