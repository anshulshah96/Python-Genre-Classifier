from pandas import HDFStore, DataFrame
from pandas import read_hdf
import pandas as pd

import numpy as np
import json

import sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

MFCC_FEATURES_N = 12
CHROMA_N = 12

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
          'pop', 'reggae', 'rock']
nb_classes = 10

def get_partition(df2):
	df_train = pd.DataFrame()
	df_cross = pd.DataFrame()
	df_test = pd.DataFrame()
	data_percentage = 100 # in integral value

	for name, group in df2:
	    file_names = group['file_name'].drop_duplicates()
	    file_names = file_names[:data_percentage]
	    file_train, file_test, _, _ = train_test_split(file_names, file_names, test_size=0.4, random_state=42)
	    file_test, file_cv, _, _ = train_test_split(file_test, file_test, test_size=0.5, random_state=42)
	    df_train = df_train.append(group[group['file_name'].isin(file_train)])
	    df_cross = df_cross.append(group[group['file_name'].isin(file_cv)])
	    df_test = df_test.append(group[group['file_name'].isin(file_test)])
    	# print name, len(file_train), len(file_cv), len(file_test)
	return df_train, df_cross, df_test

from numpy import zeros

def predict(lclist, df_cross, mm, covm):
    cls = len(mm)
    dfg = df_cross.groupby('file_name')
    pred_file_genre = pd.DataFrame(columns=('file_name', 'genre'))

    from scipy.stats import multivariate_normal
    for file_name, group in dfg:
        total = group['genre'].count()
        dist = zeros((total,cls))
        for i in range(cls):
            dist[:,i] = multivariate_normal.pdf(group[lclist], 
                        mean=mm[i], cov=covm[i],allow_singular=True)
        plist = np.sum(np.log(dist), axis =0)
        pred_file_genre = pred_file_genre.append(pd.DataFrame(data={
            'file_name':file_name,
            'genre':[np.argmax(plist)]
        }))
    return pred_file_genre
    
def gfit(chlist, df_train):
    dfg = df_train.groupby('genre')
    mm = list()
    covm = list()
    for name, group in dfg:
        mm.append(np.mean(group[chlist],axis=0))
        covm.append(np.cov(group[chlist], rowvar=False))
    return mm, covm

def train_MLE(df_train, df_cross):
    lclist = ["mfcc{}".format(i) for i in range(14)]
    
    ## Mean, standard deviation Scaling 
    scaler = preprocessing.StandardScaler().fit(df_train[lclist])
    df_train.loc[:,lclist] = scaler.transform(df_train[lclist])
    df_cross.loc[:,lclist] = scaler.transform(df_cross[lclist])
    
    mm, covm = gfit(lclist, df_train)
    pred_file_genre = predict(lclist, df_cross, mm, covm)
    
    pred_file_genre.sort_values('file_name', inplace=True)
#     print pred_file_genre.head(5)
    orig_file_genre = df_cross[['genre','file_name']].drop_duplicates().sort_values('file_name')
    
#     print pred_file_genre.head(100)
#     print orig_file_genre.head(100)
    
    print(metrics.classification_report(orig_file_genre['genre'], 
                                        pred_file_genre['genre']))

if __name__ == '__main__':
	df = read_hdf('data/features_dataset_TS0.3.h5')
	lclist = ["mfcc{}".format(i) for i in range(14)]
	df = df[np.isfinite(df['centroid'])]
	df2 = df.groupby('genre')
	df_train, df_cross, df_test = get_partition(df2)
	train_MLE(df_train, df_cross)
