# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:07:47 2018

@author: kbhandari
"""

import pandas as pd
import numpy as np
import string
import sys
import re
from nltk.corpus import stopwords
#import nltk
#nltk.download('stopwords')

data = pd.read_csv("Search Queries 20180701-20180930.csv")

def data_preprocess(data, column, lower=True, no_ascii_chars=True, no_numbers=True, no_punctuation=True, remove_stopwords=True, custom_blank_text='non ascii symbols punctuations numbers'):
    #Lower case
    if lower == True:
        data['Query_Modified'] = data[column].str.lower()
    
    #Remove non-ascii characters
    if no_ascii_chars == True:                            
        data["Query_Modified"] = data["Query_Modified"].apply(lambda x: ''.join([" " if i not in string.printable else i for i in x]))
    
    #Remove numbers
    if no_numbers == True:
        data['Query_Modified'] = data['Query_Modified'].str.replace(r'\d', '')
    
    #Punctuation
    if no_punctuation == True:
        data['Query_Modified'] = data['Query_Modified'].str.replace(r'[^\w\s]+', ' ')
    
    #Remove stopwords
    if remove_stopwords == True:
        stop = stopwords.words('english')
        data['Query_Modified'] = data['Query_Modified'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    #Replacing blanks from ascii characters, punctuations and numbers with custom text
    data['Query_Modified'].replace(r'^\s*$', custom_blank_text, regex=True, inplace = True)
    
    #Extra Spaces
    data['Query_Modified'] = data['Query_Modified'].apply(lambda x: re.sub("\s\s+", " ", str(x.strip())))
    
    return data


data = data_preprocess(data, 'Search Query')


#Check length of query
data['Query_Modified'].str.split().str.len().describe(percentiles=[0.25,0.5,0.75,0.80,0.85,0.90,0.95])
len(data[data['Query_Modified'] == ''])


def get_non_glove_words(dataframe, column, model):

    # Unique Words
    counts = dataframe[column].str.split(expand=True).stack().value_counts(dropna=False).rename_axis('unique_words').reset_index(name='counts')
    
    # Extracting Glove Words and Non Glove Words
    non_glove_words = list()
    glove_words = list()
    for i in counts['unique_words']:
        try:
            model.get_vector(i)
        except KeyError:
            non_glove_words.append(i)
        else:
            glove_words.append(i)
    
    #Non-Glove words
    non_glove_words_df = pd.DataFrame({'unique_non_glove_words':non_glove_words})
    non_glove_words_df = pd.merge(non_glove_words_df,counts,how='left',left_on=['unique_non_glove_words'],right_on=['unique_words']).iloc[:,[0,2]]
    non_glove_words_df['cum_perc'] = round(100*non_glove_words_df["counts"].cumsum()/non_glove_words_df["counts"].sum(),2)

    return(non_glove_words_df)


from gensim.models import KeyedVectors
# Load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
    
non_glove_words_df = get_non_glove_words(dataframe = data, column = 'Query_Modified', model = model)    

#Spelling Mistakes and Abbreviations
replacements = {
      '(?:^|\W)microfleece(?:$|\W)': ' micro fleece ',
      '(?:^|\W)drinkware(?:$|\W)': ' drink ware ',
      '(?:^|\W)shopee(?:$|\W)': ' shop ',
      '(?:^|\W)trackid(?:$|\W)': ' track id ',
      '(?:^|\W)tensorflow(?:$|\W)': ' tensor flow ',
      '(?:^|\W)men39s(?:$|\W)': ' men ',
      '(?:^|\W)packable(?:$|\W)': ' pack able ',
      '(?:^|\W)tube(?:$|\W)': ' youtube ',
      '(?:^|\W)tub(?:$|\W)': ' youtube ',
      '(?:^|\W)merch(?:$|\W)': ' merchandise ',
      '(?:^|\W)tee(?:$|\W)': ' t-shirt ',
      '(?:^|\W)tees(?:$|\W)': ' t-shirts ',
      r'[a-z]*y[a-z]*[a-z]*t[a-z]*[a-z]*b[a-z]*[a-z]*e[a-z0-9]*': 'youtube',
      r'[a-z]*y[a-z]*[a-z]*o[a-z]*[a-z]*u[a-z]*[a-z]*t[a-z0-9]*': 'youtube',
      r'[a-z]*y[a-z]*[a-z]*o[a-z]*[a-z]*u[a-z]*[a-z]*b[a-z0-9]*': 'youtube ',
      r'[a-z]*u[a-z]*[a-z]*t[a-z]*[a-z]*b[a-z]*[a-z]*e[a-z0-9]*': 'youtube',
      r'[a-z]*g[a-z]*[a-z]*o[a-z]*[a-z]*l[a-z]*[a-z]*e[a-z0-9]*': 'google',
      r'[a-z]*g[a-z]*[a-z]*o[a-z]*[a-z]*o[a-z]*[a-z]*g[a-z0-9]*': 'google',
      r'[a-z]*s[a-z]*[a-z]*h[a-z]*[a-z]*o[a-z]*[a-z]*p[a-z0-9]*': 'shop',
      r'[a-z]*a[a-z]*[a-z]*n[a-z]*[a-z]*d[a-z]*[a-z]*r[a-z0-9]*[a-z]*d[a-z]*[a-z]*': 'android',
      r'[a-z]*s[a-z]*[a-z]*h[a-z]*[a-z]*r[a-z]*[a-z]*t[a-z0-9]*': 'shirt',
      r'[a-z]*w[a-z]*[a-z]*m[a-z]*[a-z]*e[a-z]*[a-z]*n[a-z0-9]*': 'women',
      r'[a-z]*a[a-z]*[a-z]*p[a-z]*[a-z]*r[a-z]*[a-z]*l[a-z0-9]*': 'apparel',
      r'[a-z]*e[a-z]*[a-z]*m[a-z]*[a-z]*o[a-z]*[a-z]*j[a-z]*[a-z]*i[a-z]*[a-z0-9]*': 'emotion',
      r'[a-z]*s[a-z]*[a-z]*t[a-z]*[a-z]*o[a-z]*[a-z]*r[a-z]*[a-z]*e[a-z]*[a-z0-9]*': 'store',
      r'[a-z]*a[a-z]*[a-z]*c[a-z]*[a-z]*s[a-z]*[a-z]*r[a-z]*[a-z]*s[a-z]*[a-z0-9]*': 'accessories',
      r'[a-z]*a[a-z]*[a-z]*c[a-z]*[a-z]*c[a-z]*[a-z]*r[a-z]*[a-z]*s[a-z]*[a-z0-9]*': 'accessories',
      r'[a-z]*m[a-z]*[a-z]*e[a-z]*[a-z]*r[a-z]*[a-z]*c[a-z]*[a-z]*d[a-z]*[a-z]*i[a-z]*[a-z0-9]*': 'merchandise',
      r'[a-z]*m[a-z]*[a-z]*e[a-z]*[a-z]*r[a-z]*[a-z]*c[a-z]*[a-z]*d[a-z]*[a-z]*e[a-z]*[a-z0-9]*': 'merchandise',
      r'[a-z]*m[a-z]*[a-z]*r[a-z]*[a-z]*c[a-z]*[a-z]*h[a-z]*[a-z]*d[a-z]*[a-z]*z[a-z]*[a-z0-9]*': 'merchandise',
      r'[a-z]*m[a-z]*[a-z]*e[a-z]*[a-z]*c[a-z]*[a-z]*h[a-z]*[a-z]*d[a-z]*[a-z]*i[a-z]*[a-z0-9]*': 'merchandise',
      r'[a-z]*t[a-z]*[a-z]*i[a-z]*[a-z]*m[a-z]*[a-z]*b[a-z]*[a-z]*u[a-z]*[a-z]*k[a-z]*[a-z0-9]*': 'timbuktu',
      r'[a-z]*w[a-z]*[a-z]*a[a-z]*[a-z]*t[a-z]*[a-z]*e[a-z]*[a-z]*r[a-z]*[a-z]*b[a-z]*[a-z]*t[a-z]*[a-z]*l[a-z]*[a-z0-9]': 'water bottle',
      r'[a-z]*b[a-z]*[a-z]*a[a-z]*[a-z]*g[a-z]*[a-z]*s[a-z]*[a-z]*t[a-z]*[a-z]*o[a-z]*[a-z]*r[a-z]*[a-z]*e[a-z]*[a-z0-9]': 'bag store',
}

data['Query_Modified'].replace(replacements, regex=True, inplace=True)

#Extra Spaces
data['Query_Modified'] = data['Query_Modified'].apply(lambda x: re.sub("\s\s+", " ", str(x.strip())))

non_glove_words_df = get_non_glove_words(dataframe = data, column = 'Query_Modified', model = model)

def replace_non_glove_words(data, non_glove_words_df, column):
    #Replacing Non Glove Words with Blanks
    j=0
    length = len(non_glove_words_df['unique_non_glove_words'])-1
    for i in non_glove_words_df['unique_non_glove_words']:
        data[column].replace(r'(\b)+%s+(\b)'%i, ' ', regex=True, inplace=True)
        if j==length:
            print('\rProgress:  100%', end='')
            sys.stdout.flush()   
        elif j%10==0:
            print('\rProgress: %d' % j, end='')
            sys.stdout.flush()
        j+=1
        
    #Extra Spaces
    data[column] = data[column].apply(lambda x: re.sub("\s\s+", " ", str(x.strip())))    
    return data

data = replace_non_glove_words(data, non_glove_words_df, 'Query_Modified')
    
#Blank rows
len(data[data['Query_Modified'] == ''])
blanks = data[data['Query_Modified'] == '']
data.loc[data['Query_Modified'] == '','Query_Modified'] = 'non ascii symbols punctuations numbers'   
data = data[data['Query_Modified'] != '']

#Length of query
data['Query_Modified'].str.split().str.len().describe(percentiles=[0.25,0.5,0.75,0.80,0.85,0.90,0.95])
len(data[data['Query_Modified'] == ''])


def approach(dataframe, column, method, n=3):
    if method=="first_n_words":
        #Approach: First n words
        #Add 'blank' to words less than n
        dataframe['Length_Glove_Words'] = dataframe[column].str.split().str.len()
        def blank_words (row, n):
           for i in range(1,n+1) :
               if row['Length_Glove_Words'] == i :
                   return ' blank' * (n-i)
        
        dataframe['Words'] = dataframe.apply(lambda row: blank_words(row,n),axis=1)
        dataframe['Top_Words'] = dataframe[column].fillna('') + dataframe['Words'].fillna('')
        dataframe.drop(['Length_Glove_Words','Words'], axis=1, inplace=True)
        
        #Select First n Words
        dataframe['Top_Words'] = dataframe['Top_Words'].str.split().str[0:n].str.join(' ')
        
        #Add Glove embeddings
        gloveFile = "glove.6B.100d.txt"
        Glovewords = pd.read_table(gloveFile, sep=" ", index_col=0, header=None, quoting=3)
        
        # Unique words
        unique = list(dataframe['Top_Words'].str.split(' ', expand=True).stack().unique())
        unique_word_vec=Glovewords.loc[unique].T.to_dict('list')
        del Glovewords,unique
        
        #Glove vectors for top 3 words        
        j=0
        length = len(dataframe)-1
        stack = list()
        for index, row in dataframe.iterrows():
            df = []
            for i in range(0,n):
                df = np.append(df,unique_word_vec[row.Top_Words.split(' ')[i]])
            stack.extend(np.vstack(df).T)
            if j==length:
               print('\rProgress:  100%', end='')
               sys.stdout.flush()
            elif j%100==0:
               print('\rProgress: %d' % j, end='')
               sys.stdout.flush()
            j+=1                        
        
        del unique_word_vec
        
        stack=pd.DataFrame(stack)
        
        cluster_dataset = dataframe[["Top_Words"]]
        cluster_dataset = pd.concat([cluster_dataset.reset_index(drop=True), stack], axis=1)
        del stack
        return cluster_dataset
    
    elif method == "sum_word_vectors":        
        #Approach: Sum of d word vectors for n words
        #Add Glove embeddings
        gloveFile = "glove.6B.100d.txt"
        Glovewords = pd.read_table(gloveFile, sep=" ", index_col=0, header=None, quoting=3)
        
        # Unique words
        unique = list(dataframe[column].str.split(' ', expand=True).stack().unique())
        unique_word_vec=Glovewords.loc[unique].T.to_dict('list')
        del Glovewords,unique
        
        #Sum of Glove vectors for n words
        from operator import add
        j=0
        length = len(dataframe)-1
        stack = list()
        for index, row in dataframe.iterrows():
            sum_word_vec = [0]*100
            for word in row[column].split(' '):
                word_vec = unique_word_vec[word]
                sum_word_vec = list(map(add, sum_word_vec, word_vec))
            stack.extend([sum_word_vec])
            if j==length:
               print('\rProgress:  100%', end='')
               sys.stdout.flush()
            elif j%100==0:
               print('\rProgress: %d' % j, end='')
               sys.stdout.flush()
            j+=1
        
        stack=pd.DataFrame(stack)
        
        cluster_dataset = dataframe[[column]]
        cluster_dataset = pd.concat([cluster_dataset.reset_index(drop=True), stack], axis=1)
        del stack
        return cluster_dataset


cluster_dataset = approach(data,'Query_Modified','first_n_words',n=3)
cluster_dataset = approach(data,'Query_Modified','sum_word_vectors')


#Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(cluster_dataset.iloc[:,1:])
del cluster_dataset

wcss = []
for i in range(1, 52, 5):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
    print('\rProgress: %d' % i, end='')
    sys.stdout.flush()

import matplotlib.pyplot as plt
plt.plot(range(1, 52, 5), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 8, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_train)

del X_train

data['CLUSTERS'] = kmeans.labels_
data.to_csv("cluster_sum_glove_vectors.csv",index=False)

#Summarize results
def summary(dataframe, cluster_column, original_column, modified_column, top_n, show_original=False):
    df = dataframe.groupby([cluster_column])[["Clicks", "Impressions"]].sum()
    df['CTR'] = (df['Clicks']/df['Impressions'])*100
    df['Weighted Average Position'] = dataframe.groupby([cluster_column]).agg({'Average Position': lambda x: np.average(x, weights=data.loc[x.index, "Impressions"]) })
    df['Counts'] = dataframe.groupby([cluster_column])[[original_column]].size()
    
    if show_original==True:
        original_keywords = list()
        for i, row in df.iterrows():
            kws = ",".join(pd.Series(dataframe.loc[dataframe[cluster_column] == i,[original_column]].values.flatten()).str.split(expand=True).stack().value_counts(dropna=False).rename_axis('unique_words').reset_index(name='counts').loc[0:top_n-1,'unique_words'].tolist())
            original_keywords.extend([kws])        
        original_keywords = pd.DataFrame(original_keywords, columns=['Top Original Keywords'])
        df = pd.concat([df, original_keywords.reset_index(drop=True)], axis=1)
    
    modified_keywords = list()
    for i, row in df.iterrows():
        kws = ",".join(pd.Series(dataframe.loc[dataframe[cluster_column] == i,[modified_column]].values.flatten()).str.split(expand=True).stack().value_counts(dropna=False).rename_axis('unique_words').reset_index(name='counts').loc[0:top_n-1,'unique_words'].tolist())
        modified_keywords.extend([kws])        
    modified_keywords = pd.DataFrame(modified_keywords, columns=['Top Modified Keywords'])
    df = pd.concat([df, modified_keywords.reset_index(drop=True)], axis=1)
    
    return df

cluster_summary = summary(data, 'CLUSTERS', 'Search Query', 'Query_Modified', 10, False)
cluster_summary.to_csv("cluster_summary_sum_glove_vectors.csv",index=False)


#Sub clusters
cluster_dataset = data.loc[data['CLUSTERS'] == 4]
cluster_dataset = approach(cluster_dataset,'Query_Modified','sum_word_vectors')

#Clustering
sc_X = StandardScaler()
X_train = sc_X.fit_transform(cluster_dataset.iloc[:,1:])
del cluster_dataset

wcss = []
for i in range(1, 15, 1):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
    print('\rProgress: %d' % i, end='')
    sys.stdout.flush()

import matplotlib.pyplot as plt
plt.plot(range(1, 15, 1), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_train)

del X_train

sub_cluster = data.loc[data['CLUSTERS'] == 4].copy()
sub_cluster['SUB_CLUSTERS'] = kmeans.labels_
sub_cluster.to_csv("sub_cluster_sum_glove_vectors.csv",index=False)

sub_cluster_summary = summary(sub_cluster, 'SUB_CLUSTERS', 'Search Query', 'Query_Modified', 10, False)
cluster_summary.to_csv("sub_cluster_summary_sum_glove_vectors.csv",index=False)






#cluster_dataset.drop('CLUSTERS', axis=1, inplace=True)