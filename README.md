# Sentiment_Analysis
Chinese financial short text sentiment analysis

## Requirements
* Unix/Linux operating System
* python2.7
* python package: keras, gensim, sklearn, jieba, pypinyin, h5py, numpy, pandas, Theano. Version details see requirements text.

## Data preprocessing
### lexical analysis
First, run:
**coreNLP_lexical_analysis.py** and **jieba_lexical_analysis.py**

Second, combined result of coreNLP lexical analysis and jieba lexical analysis to binarization through **binarization.py**.

### phrase structure

run **phrase_structure_preprocess.py**

## Train and Evaluation
run **./main_code/PYGS_CNN-BiLSTM.py**

80% data is training and 20% data use in test.

### Example output

Epoch 1/30
14525/14525 [==============================] - 58s 4ms/step - loss: 0.9198 - mean_absolute_error: 0.3688 - acc: 0.5497 - val_loss: 0.7305 - val_mean_absolute_error: 0.2787 - val_acc: 0.6820

Epoch 2/30
14525/14525 [==============================] - 58s 4ms/step - loss: 0.7009 - mean_absolute_error: 0.2770 - acc: 0.6956 - val_loss: 0.6097 - val_mean_absolute_error: 0.2319 - val_acc: 0.7525

Epoch 3/30
14525/14525 [==============================] - 58s 4ms/step - loss: 0.5910 - mean_absolute_error: 0.2305 - acc: 0.7541 - val_loss: 0.5633 - val_mean_absolute_error: 0.2028 - val_acc: 0.7781

