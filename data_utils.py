import os, glob, sys

from pymir import AudioFile

import numpy as np
from numpy import zeros, newaxis
import json
import pandas as pd

from pandas import HDFStore, DataFrame
from pandas import read_hdf

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
          'pop', 'reggae', 'rock']

def read_file(filename):
    with open(filename) as data_file:    
        data = json.load(data_file)
    return data


def read_files(SEGMENT_SIZE = 65536, SONGS_N = 100):
    genresNo = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    music_files = []
    for i in range(len(GENRES)):
        genre_files = []
        for j in range(0, 100):
            genre_files.append("data/" + GENRES[i] + "/" + GENRES[i] + "." + str(j).zfill(5) + ".wav")
        music_files.append(genre_files)
    x_train = []
    y_train = []
    song_features_dataset = pd.DataFrame()
    for genreid in range(0, len(GENRES)):
        for songid in range(0, SONGS_N):
            song_name = os.path.basename(music_files[genreid][songid]).split('.')
            song_genre = GENRES.index(song_name[0])
            song_idx = int(song_name[1])

            wav_data = AudioFile.open(music_files[genreid][songid])
            fixed_frames = wav_data.frames(SEGMENT_SIZE, np.hamming)
            fixed_frames = fixed_frames[:-1]
            SEGMENTS_N = len(fixed_frames)
            spectra = [frame.spectrum() for frame in fixed_frames]
            
            zcr = pd.Series([frame.zcr() for frame in fixed_frames])
            centroid = pd.Series([spectrum.centroid() for spectrum in spectra])
            rolloff = pd.Series([spectrum.rolloff() for spectrum in spectra])

            features = pd.DataFrame(data={
                'song_idx': pd.Series([song_idx] * SEGMENTS_N),
                'centroid': centroid,
                'zcr': zcr,
                'rolloff': rolloff,
                'genre': pd.Series([song_genre] * SEGMENTS_N)
            })
            
            MFCC_FEATURES_N = 32
            LPC_N = 10
            CHROMA_N = 12

            mfcc2 = pd.DataFrame([spectrum.mfcc2() for spectrum in spectra],
                         columns=['mfcc{}'.format(i) for i in range(MFCC_FEATURES_N)])
            features = features.join(mfcc2)
            
            chroma = pd.DataFrame([spectrum.chroma() for spectrum in spectra],
                          columns=['chroma{}'.format(i) for i in range(CHROMA_N)])
            features = features.join(chroma)

            song_features_dataset = song_features_dataset.append(
                features , ignore_index=True)
    return song_features_dataset


def json_to_df(filename):
    json_data = read_file(filename)
    song_features_dataset = pd.DataFrame()
    for song_data in json_data:
        data_dict = dict()   
        data_dict['song_idx'] = int(song_data['song_name'].split('.')[1])
        for feature in song_data:
            if isinstance(song_data[feature],list):
                values = song_data[feature]
                dims = len(values)
                for i in range(dims):
                    data_dict[((feature+"{}").format(i))] = [values[i]]
            else:
                data_dict[feature] = [song_data[feature]]
        genre_name = song_data['song_name'].split('.')[0]
        data_dict['genre'] = GENRES.index(genre_name)
        features = pd.DataFrame(data=data_dict)
        song_features_dataset = song_features_dataset.append(
                features , ignore_index=True)
    return song_features_dataset


def json_to_np_array(filename):
    json_data = read_file(filename)
    song_dataset = list()
    
    for song_data in json_data:
        data_dict = dict()
        del song_data['song_name']
        
        for feature in song_data:
            if isinstance(song_data[feature],list):
                values = song_data[feature]
                for i in range(len(values)):
                    data_dict[((feature+"{}").format(i))] = values[i]
            else:
                data_dict[feature] = song_data[feature]
    
        data_dict['genre'] = GENRES.index(data_dict['genre'])
        
        song_dataset.append(data_dict.values())
        
    return np.array(song_dataset)    


def json_to_np_3d_array(filename, num_segments, num_features):
    # We assume Y-Label is the first column

    json_data = read_file(filename)
    song_dataset = list()
    
    num_songs = len(json_data) / num_segments
    
    song_3d_data = zeros((num_songs,num_segments,num_features))

    for i,song_data in enumerate(json_data):
        data_dict = dict()
        del song_data['song_name']
        
        for feature in song_data:
            if isinstance(song_data[feature],list):
                values = song_data[feature]
                for i in range(len(values)):
                    data_dict[((feature+"{}").format(i))] = values[i]
            else:
                data_dict[feature] = song_data[feature]

        data_dict['genre'] = GENRES.index(data_dict['genre'])
        
        song_dataset.append(data_dict.values())
    

    for song_i in range(num_songs):
        for song_i_seg in range(num_segments):
            for song_i_feat in range(num_features):
                song_3d_data[song_i][song_i_seg][song_i_feat] = song_dataset[song_i*num_segments + song_i_seg][song_i_feat]

    return np.array(song_3d_data)

def np_array_fold(data, num_segments):
    # if rows % num_segments != 0:
    #     raise Exception("Number of rows must be a multiple of num_segments")

    if(len(data.shape) == 2):
        rows, num_features = data.shape
        num_songs = rows / num_segments
        song_data = zeros((num_songs,num_segments,num_features))

        for song_i in range(num_songs):
            for song_i_seg in range(num_segments):
                for song_i_feat in range(num_features):
                    song_data[song_i][song_i_seg][song_i_feat] = data[song_i*num_segments + song_i_seg, song_i_feat]

    else:
        rows = data.shape
        num_songs = rows / num_segments
        song_data = zeros((num_songs,num_segments))

        for song_i in range(num_songs):
            for song_i_seg in range(num_segments):
                for song_i_feat in range(num_features):
                    song_data[song_i][song_i_seg][song_i_feat] = data[song_i*num_segments + song_i_seg, song_i_feat]

    return song_data


def test_np3d_from_2d():
    np_arr3 = json_to_np_3d_array("feature_data.json", 10, 30)
    np_arr2 = json_to_np_array("feature_data.json")

    # print np_x.shape
    # print np_y.shape
    # print np_y[:10]

    song_i = 100
    seg_num = 8
    
    print np_arr3.shape
    
    print np_arr3[song_i,:,seg_num]
    print np_arr2[song_i*10 + seg_num,:]

    print np_arr3[:, 0, 0]
    
    # Tests for equality of 2d data and 3d data
    if np_arr3[song_i,:,seg_num].all() != np_arr2[song_i*10 + seg_num,:].all():
        assert False


if __name__ == '__main__':
    # df = read_files(SEGMENT_SIZE = 65536, SONGS_N = 1)
    # print len(df)
    # print df.columns.values.tolist()
    # df = json_to_df("feature_data.json")
    # print len(df)
    # print df.columns.values.tolist()

    test_np3d_from_2d()

    