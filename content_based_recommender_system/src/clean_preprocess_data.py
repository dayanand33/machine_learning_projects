
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

plotly.offline.init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
data = pd.read_json('tops_fashion_small.json')

print('Record count : ', data.shape[0], 'Features:', data.shape[1])

pd.set_option('display.max_rows', 300)

# data[data.title.str.contains('(3X)')]


# Out of the 19 features, only the following features will be retained
#     1. asin  ( Amazon standard identification number)
#     2. brand
#     3. color
#     4. product_type_name (type of the apperal, ex: SHIRT/TSHIRT )
#     5. medium_image_url
#     6. title
#     7. formatted_price

data = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name', 'title', 'formatted_price']]
print('Number of data points : ', data.shape[0], 'Number of features:', data.shape[1])
data.head()  # prints the top rows in the table.

print(data['product_type_name'].describe())

# A vast majority of the products are shirts,
print(data['product_type_name'].unique())

# Top 5 most frequent product_type_names.
product_type_count = Counter(list(data['product_type_name']))
product_type_count.most_common(5)

data['brand'].describe()

# very few missing values

brand_count = Counter(list(data['brand']))
brand_count.most_common(5)

data['color'].describe()
color_count = Counter(list(data['color']))
color_count.most_common(5)

# price info not available for a majority of the products
data['formatted_price'].describe()

price_count = Counter(list(data['formatted_price']))
price_count.most_common(5)

# short and informative and useful for our current purpose
data['title'].describe()

# pickling data before further processing
data.to_pickle('pickels/small_apparel_data')

data = data.loc[~data['formatted_price'].isnull()]

data = data.loc[~data['color'].isnull()]

data.to_pickle('pickels/very_small_apparel_data')

# Remove near duplicate titles
# duplicated titles have different color. Duplicates aren't useful when they come up in recommendations.

print(sum(data.duplicated('title')))

# Remove All products with very few words in title
data_sorted = data[data['title'].apply(lambda x: len(x.split()) > 4)]
print("After removal of products with short description:", data_sorted.shape[0])

data_sorted.sort_values('title', inplace=True, ascending=False)
data_sorted.head()

indices = []
# i here is the row index of the data frame
for i, row in data_sorted.iterrows():
    indices.append(i)

# for index, row in data.iterrows():  #index is not enumeration. It is the index of the row. So you can do something like  data[‘name_of_col’][index] = ‘overwrite this value’
# df.index[0] # returns the index of the first row of the df. Note that index value here is the index label.
# df.index[i] notation lets you think of data frame as an array.


import itertools

stage1_dedupe_asins = []
i = 0
j = 0
num_data_points = data_sorted.shape[0]
while i < num_data_points and j < num_data_points:

    previous_i = i

    # store the list of words of ith string
    a = data['title'].loc[indices[i]].split()

    # search for the similar products sequentially
    j = i + 1
    while j < num_data_points:

        # store the list of words of jth string in b
        b = data['title'].loc[indices[j]].split()

        # store the maximum length of two strings
        length = max(len(a), len(b))

        # count is used to store the number of words that matched in both strings
        count = 0

        # itertools.zip_longest(a,b): will map the corresponding words in both strings, it will appened None in case of unequal strings
        # example: a =['a', 'b', 'c', 'd']
        # b = ['a', 'b', 'd']
        # itertools.zip_longest(a,b): will give [('a','a'), ('b','b'), ('c','d'), ('d', None)]
        for k in itertools.zip_longest(a, b):
            if (k[0] == k[1]):
                count += 1

        # if the number of words in which both strings differ are > 2 , the two titles are regarded as being different
        if (length - count) > 2:  # number of words in which both sentences differ
            # if both strings are differ by more than 2 words we include the 1st string index
            stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[i]])

            # start searching for similar apperals corresponds 2nd string
            i = j
            break
        else:
            j += 1
    if previous_i == i:
        break

data = data.loc[data['asin'].isin(stage1_dedupe_asins)]

print('Number of data points : ', data.shape[0])

data.to_pickle('pickels/very_very_very_apperal_data')

# Previously, similar titles adjacent to another were removed.
# But there are some products whose titles are not adjacent but very similar.
# This code snippet takes significant amount of time.
# O(n^2) time.

indices = []
for i, row in data.iterrows():
    indices.append(i)

stage2_dedupe_asins = []
while len(indices) != 0:
    i = indices.pop()
    stage2_dedupe_asins.append(data['asin'].loc[i])
    # consider the first apperal's title
    a = data['title'].loc[i].split()
    # store the list of words of ith string in a
    for j in indices:

        b = data['title'].loc[j].split()
        # store the list of words of jth string in b

        length = max(len(a), len(b))

        # count is used to store the number of words that matched in both the titles
        count = 0

        # itertools.zip_longest(a,b): will map the corresponding words in both strings, it will appened None in case of unequal strings
        for k in itertools.zip_longest(a, b):
            if (k[0] == k[1]):
                count += 1

        # if the number of words in which both strings differ are < 3 , we are considering it as those two apperals are same, hence we are ignoring them
        if (length - count) < 3:
            indices.remove(j)

# In[ ]:


data = data.loc[data['asin'].isin(stage2_dedupe_asins)]

print('Number of data points after stage two of dedupe: ', data.shape[0])

stop_words = set(stopwords.words('english'))


def remove_stopwords(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum()))
            # Conver all letters to lower-case
            word = word.lower()
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        data[column][index] = string


start_time = time.clock()
for index, row in data.iterrows():
    remove_stopwords(row['title'], index, 'title')
print(time.clock() - start_time, "seconds")

data.to_pickle('pickels/after_dedup_stop_words_apparel_data')
