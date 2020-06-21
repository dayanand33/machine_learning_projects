#!/usr/bin/env python

#on aws run below
# $import nltk
# $nltk.download()


#change pickle file names
#create methods wherever possible
#use tfidf vectorizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from sklearn.feature_extraction.text import CountVectorizer

from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
import time
import re
import os
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
from scipy.sparse import hstack
import plotly
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Layout
p_index = 15
recommend_cnt = 2

data = pd.read_pickle('pickels/after_dedup_stop_words_apparel_data')

class AllModelsInitializer():
    def __init__(self):
        self.bag_of_words = None
        self.tf_idf = None
        self.idf_model = None
        self.word_vec_model = None

    def initialize(self):
        self.bag_of_words = Bag_Of_Words_Model()
        print('Initialized Bag of words model')
        self.tf_idf = TfIdfModel()
        print('Initialized TfIdf model')
        self.idf_model = IdfModel()
        print('Initialized Idf model')
        self.word_vec_model = WordVecModel(self.idf_model)
        print('Initialized Word Vec model')



class Bag_Of_Words_Model():

    def __init__(self, num_results=recommend_cnt, data=data):
        self._num_results = num_results
        self._data = data
        title_vectorizer = CountVectorizer()
        self._title_features = title_vectorizer.fit_transform(data['title'])
        self._title_features.get_shape()  # a sparse 2d matrix - rows = titles, columns = dictionary of words in corpus

    def get_similar_products(self, doc_id):
        data = self._data
        # pairwise_dist will store the distance from given input title to all remaining titles
        # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
        pairwise_dist = pairwise_distances(self._title_features, self._title_features[doc_id])

        # np.argsort will return indices of the smallest distances
        indices = np.argsort(pairwise_dist.flatten())[0:self._num_results]
        #pdists will store the smallest distances. the first element will be title belonging to doc_id because
        # it is the closest
        pdists = np.sort(pairwise_dist.flatten())[0:self._num_results]

        #returns the rowindexes at those particular indices.
        df_indices = list(data.index[indices])

        title_url_lst = []
        for i in range(0,len(indices)):
            title_url_lst.append((data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]]))
        return title_url_lst

# In the output heat map each value represents the count value
# of the label word, the color represents the intersection 
# with inputs title.

# TF-IDF based product similarity



class TfIdfModel():
    def __init__(self, num_results=recommend_cnt, data=data):
        self._num_results = num_results
        self._data = data
        tfidf_title_vectorizer = TfidfVectorizer(min_df=0)
        self._tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])


    def get_similar_products(self, doc_id):
        # doc_id: apparel's id in given corpus
        pairwise_dist = pairwise_distances(self._tfidf_title_features, self._tfidf_title_features[doc_id])

        #returns the index of num_results smallest distances
        indices = np.argsort(pairwise_dist.flatten())[0:self._num_results]
        pdists  = np.sort(pairwise_dist.flatten())[0:self._num_results]
        df_indices = list(self._data.index[indices])
        title_url_lst = []

        for i in range(0,len(indices)):
            # we will pass 1. doc_id, 2. title1, 3. title2, url, model
            title_url_lst.append((self._data['title'].loc[df_indices[i]], self._data['medium_image_url'].loc[df_indices[i]]))
        return title_url_lst

import os.path
from os import path
# IDF based product similarity
class IdfModel():
    def __init__(self, num_results=recommend_cnt, data=data):
        self._num_results = num_results
        self._data = data

        if not path.exists('pickels/idf_title_features'):
            self._idf_title_vectorizer = CountVectorizer()
            self._idf_title_features = self._idf_title_vectorizer.fit_transform(data['title'])
            self._idf_title_features  = self._idf_title_features.astype(np.float)

            for i in self._idf_title_vectorizer.vocabulary_.keys():
                # for every word in whole corpus find the idf value
                idf_val = self.idf(i)

                # to calculate idf_title_features, replace the count values with the idf values of the word
                # idf_title_features[:, idf_title_vectorizer.vocabulary_[i]].nonzero()[0] will return all documents in which the word i present
                for j in self._idf_title_features[:, self._idf_title_vectorizer.vocabulary_[i]].nonzero()[0]:
                    # replace the count values of word i in document j with  idf_value of word i
                    # idf_title_features[doc_id, index_of_word_in_courpus] = idf value of word
                    self._idf_title_features[j, self._idf_title_vectorizer.vocabulary_[i]] = idf_val
            pickle.dump(self._idf_title_features, open('pickels/idf_title_features','wb'))
            pickle.dump(self._idf_title_vectorizer, open('pickels/idfcv', 'wb'))
        else:
            idftf = pickle.load(open('pickels/idf_title_features','rb'))
            self._idf_title_features = idftf
            idfcv = pickle.load(open('pickels/idfcv','rb'))
            self._idf_title_vectorizer = idfcv

    def n_containing(self, word):
        return sum(1 for blob in self._data['title'] if word in blob.split())

    def idf(self, word):
        # idf = log(#number of docs / #number of docs which had the given word)
        return math.log(self._data.shape[0] / (self.n_containing(word)))

    def get_vectorizer(self):
        return self._idf_title_vectorizer

    def get_idf_title_features(self):
        return self._idf_title_features

    def get_similar_products(self, doc_id):

        # pairwise_dist will store the distance from given input apparel to all remaining apparels
        # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
        pairwise_dist = pairwise_distances(self._idf_title_features, self._idf_title_features[doc_id])

        indices = np.argsort(pairwise_dist.flatten())[0:self._num_results]
        pdists  = np.sort(pairwise_dist.flatten())[0:self._num_results]

        df_indices = list(self._data.index[indices])
        title_url_lst = []

        for i in range(0,len(indices)):
            title_url_lst.append((self._data['title'].loc[df_indices[i]], self._data['medium_image_url'].loc[df_indices[i]]))
        return title_url_lst



class WordVecModel:

        #tododaya
        # in this project we are using a pretrained model by google
        # its 3.3G file, once you load this into your memory
        # it occupies ~9Gb, so please do this step only if you have >12G of ram
        # we will provide a pickle file wich contains a dict ,
        # and it contains all our courpus words as keys and  model[word] as values
        # To use this code-snippet, download "GoogleNews-vectors-negative300.bin"
        # from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
        # it's 1.9GB in size.

         #model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


    def __init__(self, idf_model, num_results=recommend_cnt, data=data):
        self._num_results = num_results
        self._data = data


        #tododaya if you do NOT have RAM >= 12GB, use the code below.
        with open('pickels/word2vec_model', 'rb') as handle:
            self._model = pickle.load(handle)
        self._vocab = self._model.keys()

        docid = 0
        self._w2v_title = []
        # for every title we build a avg vector representation
        for i in data['title']:
            self._w2v_title.append(self.build_avg_vec(i, 300, docid, 'avg', idf_model))
            docid += 1

        # w2v_title = np.array(# number of doc in courpus * 300), each row corresponds to a doc
        self._w2v_title = np.array(self._w2v_title)

        docid = 0
        self._w2v_title_weight = []
        # for every title build a weighted vector representation
        for i in data['title']:
            self._w2v_title_weight.append(self.build_avg_vec(i, 300, docid, 'weighted', idf_model))
            docid += 1
        # w2v_title = np.array(# number of doc in courpus * 300), each row corresponds to a doc
        self._w2v_title_weight = np.array(self._w2v_title_weight)

        # Weighted similarity using brand and color.

        self._data['brand'].fillna(value="Not given", inplace=True)

        # replace spaces with hypen
        brands = [x.replace(" ", "-") for x in data['brand'].values]
        types = [x.replace(" ", "-") for x in data['product_type_name'].values]
        colors = [x.replace(" ", "-") for x in data['color'].values]

        self._brand_vectorizer = CountVectorizer()
        self._brand_features = self._brand_vectorizer.fit_transform(brands)

        self._type_vectorizer = CountVectorizer()
        self._type_features = self._type_vectorizer.fit_transform(types)

        self._color_vectorizer = CountVectorizer()
        self._color_features = self._color_vectorizer.fit_transform(colors)

        # stack sparse matrices vertically (row wise)
        self._extra_features = hstack((self._brand_features, self._type_features, self._color_features)).tocsr()

    def get_word_vec(self, sentence, doc_id, m_name, idf_model):
        # sentence : title of the apparel
        # doc_id: document id in our corpus
        # m_name: model information it will take two values
            # if  m_name == 'avg', we will append the model[i], w2v representation of word i
            # if m_name == 'weighted', we will multiply each w2v[word] with the idf(word)
        vec = []
        idf_title_vectorizer = idf_model.get_vectorizer()
        idf_title_features = idf_model.get_idf_title_features()

        for i in sentence.split():
            if i in self._vocab:
                if m_name == 'weighted' and i in  idf_title_vectorizer.vocabulary_:
                    vec.append(idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[i]] * self._model[i])
                elif m_name == 'avg':
                    vec.append(self._model[i])
            else:
                # if the word is present in corpus but missing in google's word2vec then set the vector to all zeros
                vec.append(np.zeros(shape=(300,)))
        return  np.array(vec)

    def get_distance(self, vec1, vec2):
        final_dist = []
        # for each vector in vec1 caluclate the distance(euclidean) to all vectors in vec2
        for i in vec1:
            dist = []
            for j in vec2:
                # np.linalg.norm(i-j) will result the euclidean distance between vectors i, j
                dist.append(np.linalg.norm(i-j))
            final_dist.append(np.array(dist))
        # final_dist = np.array(#number of words in title1 * #number of words in title2)
        # final_dist[i,j] = euclidean distance between vectors i, j
        return np.array(final_dist)

    # vocab = stores all the words that are there in google w2v model
    # vocab = model.wv.vocab.keys() # if you are using Google word2Vec

    # this function will add the vectors of each word and returns the avg vector of given sentence
    def build_avg_vec(self, sentence, num_features, doc_id, m_name, idf_model):
        # sentence: its title of the apparel
        # num_features: the length of word2vec vector, its values = 300
        # m_name: model information it will take two values
            # if  m_name == 'avg', append the model[i], w2v representation of word i
            # if m_name == 'weighted', multiply each w2v[word] with the idf(word)

        featureVec = np.zeros((num_features,), dtype="float32")
        # intialize a vector of size 300 with all zeros
        # add each word2vec(word) to this featureVec
        nwords = 0
        idf_title_features = idf_model.get_idf_title_features()
        idf_title_vectorizer = idf_model.get_vectorizer()
        for word in sentence.split():
            nwords += 1
            if word in self._vocab:
                if m_name == 'weighted' and word in  idf_title_vectorizer.vocabulary_:
                    featureVec = np.add(featureVec, idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[word]] * self._model[word])
                elif m_name == 'avg':
                    featureVec = np.add(featureVec, self._model[word])
        if(nwords>0):
            featureVec = np.divide(featureVec, nwords)
        # returns the avg vector of given sentence, its of shape (1, 300)
        return featureVec


# Average Word2Vec product similarity.



    def avg_w2v_model(self, doc_id):
        # doc_id: apparel's id in given corpus

        # dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
        pairwise_dist = pairwise_distances(self._w2v_title, self._w2v_title[doc_id].reshape(1,-1))

        indices = np.argsort(pairwise_dist.flatten())[0:self._num_results]
        pdists  = np.sort(pairwise_dist.flatten())[0:self._num_results]

        df_indices = list(data.index[indices])

        title_url_lst = []

        for i in range(0, len(indices)):
            title_url_lst.append((self._data['title'].loc[df_indices[i]], self._data['medium_image_url'].loc[df_indices[i]]))

        return title_url_lst

    # IDF weighted Word2Vec for product similarity
    def weighted_w2v_model(self, doc_id):
        # doc_id: apparel's id in given corpus

        # pairwise_dist will store the distance from given input apparel to all remaining apparels
        # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
        # http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
        pairwise_dist = pairwise_distances(self._w2v_title_weight, self._w2v_title_weight[doc_id].reshape(1,-1))

        # np.argsort will return indices of 9 smallest distances
        indices = np.argsort(pairwise_dist.flatten())[0:self._num_results]
        #pdists will store the 9 smallest distances
        pdists  = np.sort(pairwise_dist.flatten())[0:self._num_results]

        #data frame indices of the 9 smallest distace's
        df_indices = list(data.index[indices])
        title_url_lst = []

        for i in range(0, len(indices)):
            title_url_lst.append((self._data['title'].loc[df_indices[i]], self._data['medium_image_url'].loc[df_indices[i]]))

        return title_url_lst


    def idf_w2v_brand(self, doc_id, w1, w2):
        # doc_id: apparel's id in given corpus
        # w1: weight for  w2v features
        # w2: weight for brand and color features

        # pairwise_dist will store the distance from given input apparel to all remaining apparels
        # the metric we used here is cosine, the cosine distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
        idf_w2v_dist  = pairwise_distances(self._w2v_title_weight, self._w2v_title_weight[doc_id].reshape(1,-1))
        ex_feat_dist = pairwise_distances(self._extra_features, self._extra_features[doc_id])
        pairwise_dist   = (w1 * idf_w2v_dist +  w2 * ex_feat_dist)/float(w1 + w2)

        indices = np.argsort(pairwise_dist.flatten())[0:self._num_results]
        pdists  = np.sort(pairwise_dist.flatten())[0:self._num_results]
        df_indices = list(data.index[indices])

        title_url_lst = []

        for i in range(0, len(indices)):
            title_url_lst.append((self._data['title'].loc[df_indices[i]], self._data['medium_image_url'].loc[df_indices[i]]))
        return title_url_lst
