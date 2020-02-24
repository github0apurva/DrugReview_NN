# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:12:57 2020

@author: Apurva
"""

import numpy as np
np.random.seed(27)
import time as tp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from string import punctuation
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from sklearn.externals import joblib


from flask import Flask , jsonify
#import Untitled5.py
app = Flask(__name__)


@app.route("/")
def hello_world():
  return "Hello, How are you!"

@app.route("/model_train")
def hello():
    def change_y (rating, param_prep):
        r1 = np.where ( rating < 4 , 0 , rating )
        r2 = np.where ( r1 > 7 , 2 , r1 )
        r3 = np.where ( r2 > 2 , 1  , r2 )
        return r3
    def aux_data ( tr, param_prep ):
        tr_aux1 = tr[['usefulCount' ,  'date']]
        tr_ax = tr_aux1.drop(['date' ], axis = 1 )
        return tr_ax
    def filter_punc_stop_stem_words ( train_sentences, param_prep ):
        stop_words_sel = ( set(stopwords.words('english'))  - param_prep['stopwords_no'] )
        train_sentences2 = [ train_sentences[x].translate(str.maketrans('','', punctuation)) for x in range (len(train_sentences))]
        train_sentences3 = train_sentences2
        ps = PorterStemmer()
        for i, sen in enumerate(train_sentences2) :
            sent = [ ps.stem(w.lower()) for w in sen.split() if w.lower() not in stop_words_sel ]
            train_sentences3[i] = ' '.join(sent)
        return train_sentences3
    def data_prep_train (train_raw, param_prep ):
        train_text = [x for x in train_raw.review]
        train_text1 = filter_punc_stop_stem_words ( train_text , param_prep)
        tokenizer = Tokenizer(num_words = param_prep['vocab_size'] , oov_token = param_prep['oov_tok'] )
        tokenizer.fit_on_texts(train_text1)
        word_index = tokenizer.word_index
        seq = tokenizer.texts_to_sequences(train_text1)
        seq_padded = pad_sequences(seq, padding = 'post' , maxlen = param_prep['max_length'] , truncating = param_prep['trunc_type'] ) 
        label3 = change_y(train_raw['rating'], param_prep ) 
        train_aux = aux_data (train_raw , param_prep )
        return  seq_padded , train_aux , label3, tokenizer
    def train_model ():
        param_prep = {
            'vocab_size' : 10000 ,
            'max_length' : 100 ,
            'stopwords_no' : set({ 'no', 'not', 'but', "haven't", "weren't", "wasn't", "doesn't", 'off',
                          "hadn't", "mightn't", "couldn't", "shan't", "mustn't", "isn't", "won't",
                          "needn't", "don't", "didn't", "shouldn't", "wouldn't", "hasn't" }),
            'oov_tok' : "<OOV>" ,
            'padding_type' : 'post',
            'trunc_type' : 'post',
            'embedding_dim' : 16 ,
            'training_size' : 1024,
            'epochs' : 25,
            'pkl_filename' : "my_model.h5" ,
            'tok' : "tokenizer.pickle"
            }       
        train_raw = pd.read_csv("C:/Users/Apurva/Desktop/CaseStudy/Test/drugsComTrain_raw.tsv", sep = "\t" )
        print ( "Progress: LOADING DATA COMPLETE" )
        seq_padded , train_aux , label3, tokenizer = data_prep_train (train_raw, param_prep )
        print ( "Progress: DATA PREPARATION COMPLETE" )
        # model creation
        aux_input = tf.keras.Input(shape=(train_aux.shape[1],), name='aux')
        prim_input = tf.keras.Input( shape =(param_prep['max_length'] ) , name='prim')
        feat_embed = tf.keras.layers.Embedding(param_prep['vocab_size'] ,param_prep['embedding_dim'] , input_length = param_prep['max_length'] )(prim_input)
        feat_flat = tf.keras.layers.Flatten()(feat_embed)
        feat_con = tf.keras.layers.concatenate([feat_flat,aux_input])
        feat_dens = tf.keras.layers.Dense(10,activation = 'relu')(feat_con)
        feat_sftm = tf.keras.layers.Dense(3,activation = 'softmax')(feat_dens)
        model = tf.keras.Model(inputs=[prim_input, aux_input], outputs=[feat_sftm])
        model.summary()
        
        model.compile( loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        print ( "progress: BUILDING MODEL" )
        hist = model.fit({'prim': seq_padded ,'aux': train_aux } , label3, epochs = param_prep['epochs'] ,
                          verbose = 2, batch_size = param_prep['training_size'] )
        train_pred = model.predict( {'prim': seq_padded ,'aux': train_aux } )
        print ( "progress: EPOCHS COMPLETE" )
        print ( "selected model output : ", hist.epoch[-6] + 1 , "epochs and training accuracy of ", hist.history['acc'][-6] )
        
        # saving model
        model.save(param_prep['pkl_filename'])  # creates a HDF5 file 'my_model.h5'
        # saving tokenizer
        joblib.dump(tokenizer, param_prep['tok'])
        return hist.history['loss'][-6] , hist.history['acc'][-6], train_pred
    tr_loss , tr_acc, tr_pred =  train_model()
    t = int(tr_acc*10000)/100 
    return jsonify({'train accuracy %': t })


@app.route("/model_test")
def hellotest():
    def change_y (rating, param_prep):
        r1 = np.where ( rating < 4 , 0 , rating )
        r2 = np.where ( r1 > 7 , 2 , r1 )
        r3 = np.where ( r2 > 2 , 1  , r2 )
        return r3
    def aux_data ( tr, param_prep ):
        tr_aux1 = tr[['usefulCount' ,  'date']]
        tr_ax = tr_aux1.drop(['date' ], axis = 1 )
        return tr_ax
    def filter_punc_stop_stem_words ( train_sentences, param_prep ):
        stop_words_sel = ( set(stopwords.words('english'))  - param_prep['stopwords_no'] )
        train_sentences2 = [ train_sentences[x].translate(str.maketrans('','', punctuation)) for x in range (len(train_sentences))]
        train_sentences3 = train_sentences2
        ps = PorterStemmer()
        for i, sen in enumerate(train_sentences2) :
            sent = [ ps.stem(w.lower()) for w in sen.split() if w.lower() not in stop_words_sel ]
            train_sentences3[i] = ' '.join(sent)
        return train_sentences3
    def data_prep_test ( train_raw, tokenizer, param_prep):
        train_text = [x for x in train_raw.review]
        train_text1 = filter_punc_stop_stem_words ( train_text , param_prep)
        seq = tokenizer.texts_to_sequences(train_text1)
        seq_padded = pad_sequences(seq, padding = 'post' , maxlen = param_prep['max_length'] , truncating = param_prep['trunc_type'] ) 
        label3 = change_y(train_raw['rating'], param_prep) 
        train_aux = aux_data (train_raw, param_prep)
        return  seq_padded , train_aux , label3
    def pred_model ( ):
        param_prep = {
            'vocab_size' : 10000 ,
            'max_length' : 100 ,
            'stopwords_no' : set({ 'no', 'not', 'but', "haven't", "weren't", "wasn't", "doesn't", 'off',
                          "hadn't", "mightn't", "couldn't", "shan't", "mustn't", "isn't", "won't",
                          "needn't", "don't", "didn't", "shouldn't", "wouldn't", "hasn't" }),
            'oov_tok' : "<OOV>" ,
            'padding_type' : 'post',
            'trunc_type' : 'post',
            'embedding_dim' : 16 ,
            'training_size' : 1024,
            'epochs' : 25,
            'pkl_filename' : "my_model.h5" ,
            'tok' : "tokenizer.pickle"
            }  
        
        # loading model 
        pickle_model = load_model(param_prep['pkl_filename']) 
        # loading tokenizer
        tokenizer = joblib.load(param_prep['tok'])
        test_raw = pd.read_csv("C:/Users/Apurva/Desktop/CaseStudy/Test/drugsComTest_raw.tsv", sep = "\t" )
        print ( "Progress: LOADING DATA COMPLETE" )
        seq_padded_test , test_aux , label3_test = data_prep_test (test_raw, tokenizer, param_prep )
        print ( "Progress: DATA PREPARATION COMPLETE" )
        test_loss , test_acc = pickle_model.evaluate(x={'prim':  seq_padded_test , 'aux': test_aux }, y=label3_test , batch_size= param_prep['training_size'] , verbose=2)
        test_pred = pickle_model.predict( {'prim':  seq_padded_test , 'aux': test_aux } )
        print ( "progress: BUILDING MODEL" )
        print ( "selected model output : accuracy of ", test_acc  )
        return test_loss, test_acc, test_pred
    tt_loss , tt_acc, tt_pred =  pred_model()
    t = int(tt_acc*10000)/100 
    return jsonify({'test accuracy %': t })

if __name__ == '__main__':
    app.run()