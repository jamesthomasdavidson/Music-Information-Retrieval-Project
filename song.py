from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from scipy import spatial
import matplotlib.pyplot as plt
import numpy as np
import os, pickle


class Song(object):

    def __init__(self, metadata, song_wav):
        self._song = song_wav
        self._artist = metadata['artist']
        self._album = metadata['album']
        self._release_date = metadata['release_date']
        self._name = metadata['song']
        self._timbral_features = None
        self._tempo = None

    @property
    def name(self):
        return self._song

    @property
    def timbral(self):
        return self._timbral_features

    @property
    def tempo(self):
        return self._tempo

class Mashup(Song):

    def __init__(self, s1, s2):
        def ratio(n1, n2):
            r = None
            if np.abs(n1) > np.abs(n2):
                r = n2/n1
            else:
                r = n1/n2
            return int(r*100)/100.0
        self._n = len(s1.timbral)
        self._similiarity = []
        for f1 in s1.timbral:
            row = []
            for f2 in s2.timbral:
                row.append(ratio(f1, f2))
            self._similiarity.append(row)

    @property
    def feature_matrix(self):
        return self._similiarity

    def out(self):
        for i in range(self._n):
            for j in range(self._n):
                if i < j: continue
                print('%5.2f ' % (self._similiarity[i][j])),
            print

class MashupMatrix():

    def __init__(self, mashups):
        feature_matrices = [mashup.feature_matrix for mashup in mashups]
        self._n = len(feature_matrices[0])
        self._mashup_matrix = {}
        for i in range(self._n):
            for j in range(self._n):
                if i < j: continue
                self._mashup_matrix[(i,j)] = [matrix[i][j] for matrix in feature_matrices]

    def vector(self, i, j):
        return self._mashup_matrix[(i,j)]

    def out(self, val = None):
        func = np.std
        if val is 'mean':
            func = np.mean
        elif val is 'median':
            func = np.median

        for i in range(self._n):
            for j in range(self._n):
                if i < j: continue
                print('%5.2f ' % (func(self._mashup_matrix[(i,j)]))),
            print
