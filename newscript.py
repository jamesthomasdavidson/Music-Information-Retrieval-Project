from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt

import numpy as np
import pickle

with open('songs_pickle.pck', 'rb') as f:
    songs = pickle.load(f)


for song in songs:

	[Fs, x] = audioBasicIO.readAudioFile(songs[0]._filename);
	F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
	print("processing song number ", song._song_id )
	B = audioFeatureExtraction.beatExtraction(F, .050)

	features = []
	for f in F:
		feature = np.mean(f)
		features.append(feature)

	song._timbral_features = features 
	song._tempo = B


f = open('songs_pickle.pck', mode='wb')
pickle.dump(songs, file=f, protocol=2)