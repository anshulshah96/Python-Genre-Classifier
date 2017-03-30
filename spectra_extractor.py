from __future__ import print_function, division

import os
import glob
import sys

import numpy as np
import pandas as pd
from pymir import AudioFile


GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
          'pop', 'reggae', 'rock']

N_SEGMENTS = 100
N_COL = 1000
# WINDOW_SIZE = 16384 # about 0.36 ms
# MAX_SEGMENTS = 40


def get_genre_songs_path(genre_name):
    return glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'data', genre_name, '*'))


def mfcc_extend(spectra, numFeatures = 13):
    mfcclst = []
    for i in range(numFeatures):
        mfcclst = mfcclst + [spectra.mfcc(i)]
    return mfcclst


def extract_features(song_path, N_SEGMENTS = 100):
    """Extract required spectral and temporal features from a particular song.
    Features extracted here include Zero Crossing Rate, Spectral Centroids,
    Rolloff, Chromas and MFCCs (Mel Frequency Cepstral Coefficients).
    Arguments
    ---------
    song_path: str
        path to the song file in ``data`` directory
    Returns
    -------
    pd.DataFrame
        Each row represents features from each frame / spectrum of a song.
    """
    file_name = os.path.basename(song_path)
    song_name = os.path.basename(song_path).split('.')
    song_genre = GENRES.index(song_name[0])
    song_idx = int(song_name[1])

    song_data = AudioFile.open(song_path)
    
    # By Time Duration
    fixed_frames = song_data.frames(len(song_data) // N_SEGMENTS, np.hamming)
    fixed_frames = fixed_frames[:N_SEGMENTS-1]
    
    # By Window Size
    # fixed_frames = song_data.frames( WINDOW_SIZE, np.hamming)
    # fixed_frames = fixed_frames[:MAX_SEGMENTS]
    
    spectra = [frame.spectrum() for frame in fixed_frames]

    ddata = np.array(spectra)
    b = np.array([[int(song_genre)]*len(spectra)])
    ddata = np.concatenate(( b.T, ddata), axis=1)
    ddata = ddata.astype('float32')
    # print (ddata[1,:])
    ddata = ddata[:,:N_COL]
    return ddata.reshape((1,99,N_COL))


def show_progress_bar(processed, total=100):
    """Returns a string to display the progress of processed songs.
    Arguments
    ---------
    processed : int
        Number of processed songs
    total : int, optional
        Total number of songs (default 100)
    Returns
    -------
        str showing progress of processed songs
    """

    n_dots = processed * 50 // total
    n_spaces = 50 - n_dots
    progress_string = '[{0}{1}] Processed {2}/{3} songs'.format(
        '.' * int(n_dots), ' ' * int(n_spaces), processed, total)

    return progress_string


if __name__ == '__main__':
    song_features_dataset = pd.DataFrame()

    data_np3 = np.empty((0,99,N_COL), 'float32')
    for genre_name in GENRES[:10]:
        song_paths = sorted(get_genre_songs_path(genre_name))
        print("Processing songs of {0} genre...".format(genre_name))
        for index, song_path in enumerate(song_paths[:30]):
            song_data = extract_features(song_path)
            # data_np3 = data_np3.append(dasong_data)
            data_np3 = np.append(data_np3, song_data, axis=0)
            sys.stdout.write('\r')
            sys.stdout.write(show_progress_bar(index))
            sys.stdout.flush()
    print(data_np3.shape)
    
    np.save('/mnt/2082D50C82D4E6F6/DATA/np/song_spectra_np_min_2.npy',data_np3)