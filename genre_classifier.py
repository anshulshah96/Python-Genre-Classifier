import os, glob, sys

import numpy as np
import pandas as pd

from pandas import HDFStore, DataFrame
from pandas import read_hdf
from sklearn.model_selection import train_test_split

from data_utils import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D



#old lib
from keras.layers import Convolution2D, MaxPooling2D

from sklearn import preprocessing

from sklearn import svm
from sklearn import metrics

MFCC_FEATURES_N = 12
LPC_N = 10
CHROMA_N = 12

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
          'pop', 'reggae', 'rock']

nb_classes = 10

def get_keras_model():
    model = Sequential()
    model.add(Dense(output_dim=35, input_dim=29))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    return model


def get_keras_model2():
    model = Sequential()
    model.add(Convolution1D(40, 3, border_mode='same', input_shape=(10, 29)))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(output_dim=25))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    return model

def get_keras_model3(np_array):
    model = Sequential()

    model.add(Conv2D(32, 3, 3 , input_shape=np_array.shape[1:] ))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, 3 ))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model

def get_keras_model4(np_array):
    model = Sequential()
    model.add(Convolution2D(16, 3, 3, activation='relu',input_shape=np_array.shape[1:] ))
    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
     
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes, activation='softmax'))
     
    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model

def transform(features):
    feature = []
    for array in features:
        index = np.argmax(array)
        array = [0] * (len(array))
        array[index] = 1
        feature.append(array)
    return feature


def train(df, predictor_var , outcome_var):
    batch_size = 10
    nb_epoch = 5
    x_train, x_test, y_train, y_test = train_test_split(
        df[predictor_var], df[outcome_var], test_size=0.3, random_state=42)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    model = get_model(predictor_var)
    model.fit(x_train_transformed, y_train, nb_epoch=nb_epoch, batch_size=batch_size)
    loss_and_metrics = model.evaluate(x_test_transformed, y_test, batch_size=batch_size)
    print loss_and_metrics
    return model


def train_on_df():
    # reads data
    df = json_to_df("feature_data.json")

    # Shuffles the data
    df = df.sample(frac =  1)

    predictor_var = ["centroid","rolloff","zcr","mean","variance"]
    for feature in ['mfcc{}'.format(feature_j) for feature_j in range(MFCC_FEATURES_N)]:
        predictor_var.append(feature)
    # for feature in ['lpc{}'.format(feature_j) for feature_j in range(LPC_N)]:
    #     predictor_var.append(feature)
    for feature in ['chroma{}'.format(feature_j) for feature_j in range(CHROMA_N)]:
        predictor_var.append(feature)
    print predictor_var

    outcome_var = "genre"

    keras_model = train(df,predictor_var,outcome_var)

    return keras_model


def train_on_np_array():
    df = json_to_np_array("feature_data.json") 
    print df.shape

    # shuffle the data
    np.random.shuffle(df)

    # split into input and corresponding labels
    x_data = df[:, 1:]

    y_data = df[:, 0]

    # convert class vectors to binary class matrices
    y_data = np_utils.to_categorical(y_data, nb_classes)

    print x_data.shape
    print y_data.shape
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    scaler = preprocessing.StandardScaler().fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model = get_keras_model()

    history = model.fit(x_train, y_train, nb_epoch=30, batch_size=32)
    predicted  = model.predict(x_test)

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)

    print loss_and_metrics
    # print np.argmax(predicted,axis = 1).shape
    print(metrics.classification_report(np.argmax(y_test,axis = 1), np.argmax(predicted,axis = 1) ))

    return model

def train_on_multiclass_svm():
    df = json_to_np_array("feature_data.json")
    
    # split into input and corresponding labels
    x_data = df[:, 1:]
    y_data = df[:, 0]
    print y_data
    # convert class vectors to binary class matrices
    # y_data = np_utils.to_categorical(y_data, nb_classes)

    print x_data.shape
    print y_data.shape
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    scaler = preprocessing.StandardScaler().fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    #plt.scatter(x_data,y_data)
    #plt.show()
    
    # fit SVM model
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    
    predicted = clf.predict(x_test)
    
    # summarize the fit of the model
    # print("expected: ",y_test," predicted: ",predicted)
    print(metrics.classification_report(y_test, predicted))
    
    return clf

def to_cat(data):
    y_data2 = np.zeros((data.shape[0],nb_classes))
    for i in range(data.shape[0]):
        # print y_data2[i,int(data[i])]
        y_data2[i,int(data[i])] = 1
    return y_data2

def train_on_spectra():
    from config import *

    df = np.load(np_data_path+'song_spectra_np.npy')
    np.random.shuffle(df)
    
    x_data = df[:, : , 1:]
    x_data = x_data / 95
    y_data = df[:, 1 , 0].reshape((df.shape[0]))

    # scaler = preprocessing.StandardScaler().fit(x_data)
    # x_data = scaler.transform(x_data)

    print(x_data.shape, y_data.shape)

    y_data = to_cat(y_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    x_train2 = x_train.reshape((x_train.shape[0], 1,x_train.shape[1],x_train.shape[2]))
    x_test2 = x_test.reshape((x_test.shape[0], 1,x_test.shape[1],x_test.shape[2]))

    model = get_keras_model4(x_train2)
    history = model.fit(x_train2, y_train, nb_epoch=5, batch_size=32)
    y_test_op = model.predict(x_test2,batch_size=32)
    print(y_test_op[:10,:])
    print(y_test[:10,:])
    loss_and_metrics = model.evaluate(x_test2, y_test, batch_size=32)

    print loss_and_metrics

    return model
  
def train_on_np_3d_array():
    df = json_to_np_array("feature_data.json")

    num_segments = 10

    # split into input and corresponding labels
    x_data = df[:, 1:]
    y_data = df[:, 0]

    scaler = preprocessing.StandardScaler().fit(x_data)
    x_data = scaler.transform(x_data)

    # convert class vectors to binary class matrices
    print y_data.shape
    y_data = to_cat(y_data)
    # y_data = np_utils.to_categorical(y_data, nb_classes)

    print x_data.shape, y_data.shape
    
    # Conversion to 3d
    x_data = np_array_fold(x_data, num_segments)
    y_data = np_array_fold(y_data, num_segments)

    # Shuffle the data
    # Merging Necessary because x_data must be linked with y_data
    data_merge = np.concatenate((y_data, x_data), axis = 2)
    print x_data.shape, y_data.shape, data_merge.shape
    np.random.shuffle(data_merge)

    y_data = data_merge[:,:,:nb_classes]
    x_data = data_merge[:,:,nb_classes:]
    y_data = np.array([f_element[0] for f_element in y_data])
    print x_data.shape, y_data.shape

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.8, random_state=42)
    
    # model = get_keras_model2()
    x_train2 = x_train.reshape((x_train.shape[0], 1,x_train.shape[1],x_train.shape[2]))
    x_test2 = x_test.reshape((x_test.shape[0], 1,x_test.shape[1],x_test.shape[2]))

    model = get_keras_model4(x_train2)

    history = model.fit(x_train2, y_train, nb_epoch=5, batch_size=32)
    y_test_op = model.predict(x_test2, batch_size=32)
    print(y_test_op[:30 , :])
    print(y_test[:30 , :])
    loss_and_metrics = model.evaluate(x_test2, y_test, batch_size=32)

    print loss_and_metrics

    return model


def save_model(keras_model):
    keras_model.save('data/keras_model_1.h5')

    keras_model.save_weights('data/keras_model_1_weights')
    
    json_string = keras_model.to_json()
    text_file = open("data/keras_model_1", "w")
    text_file.write(json_string)
    text_file.close()


if __name__=="__main__":
    # For training using dataframe
    # keras_model = train_on_df()

    # For training using numpy array

    # keras_model = train_on_np_array()
    # svm_model = train_on_multiclass_svm()
     # keras_model = train_on_np_3d_array()
    skeras_model = train_on_spectra()
    # save_model(keras_model)




