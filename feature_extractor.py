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
    fixed_frames = fixed_frames[:N_SEGMENTS]
    
    # By Window Size
    # fixed_frames = song_data.frames( WINDOW_SIZE, np.hamming)
    # fixed_frames = fixed_frames[:MAX_SEGMENTS]
    
    spectra = [frame.spectrum() for frame in fixed_frames]

    zcr = pd.Series([frame.zcr() for frame in fixed_frames])
    centroid = pd.Series([spectrum.centroid() for spectrum in spectra])
    rolloff = pd.Series([spectrum.rolloff() for spectrum in spectra])

    N_SEGMENTS = len(centroid)
    features = pd.DataFrame(data={
        'genre': pd.Series([song_genre] * N_SEGMENTS),
        'song_idx': pd.Series([song_idx] * N_SEGMENTS),
        'file_name': pd.Series([file_name] * N_SEGMENTS),
        'centroid': centroid,
        'zcr': zcr,
        'rolloff': rolloff
    })

    mfcc2 = pd.DataFrame([spectrum.mfcc2() for spectrum in spectra],
                         columns=['mfcc{}'.format(i) for i in range(32)])
    features = features.join(mfcc2)

    chroma = pd.DataFrame([spectrum.chroma() for spectrum in spectra],
                          columns=['chroma{}'.format(i) for i in range(12)])
    features = features.join(chroma)
    return features


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

    for genre_name in GENRES:
        song_paths = sorted(get_genre_songs_path(genre_name))

        print("Processing songs of {0} genre...".format(genre_name))
        for index, song_path in enumerate(song_paths):
            song_features_dataset = song_features_dataset.append(
                extract_features(song_path), ignore_index=True)
            sys.stdout.write('\r')
            sys.stdout.write(show_progress_bar(index))
            sys.stdout.flush()

    song_features_dataset = song_features_dataset[np.isfinite(
        song_features_dataset['genre'])]
    # song_features_dataset.to_hdf('data/features_dataset_WS'+str(WINDOW_SIZE)+'.h5', key='dataset')
    song_features_dataset.to_hdf('data/features_dataset_TS'+str((30.0/N_SEGMENTS))+'.h5', key='dataset')
    print("Saved features dataset in data directory.")