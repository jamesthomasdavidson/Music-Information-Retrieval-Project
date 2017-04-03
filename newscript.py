from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt

import numpy as np
import pickle

with open('songs_pickle.pck', 'rb') as f:
    songs = pickle.load(f)


for song in songs:

	[Fs, x] = audioBasicIO.readAudioFile(song._filename);
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


"""
uncomment for progress report csvs 


"""



# for i in range(4):
# 	[Fs, x] = audioBasicIO.readAudioFile(songs[i]._filename);
# 	F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);
# 	print('processing song number ', songs[i]._song_id )
# 	B = audioFeatureExtraction.beatExtraction(F, .050)

# 	features = []
# 	for f in F:
# 		feature = np.mean(f)
# 		features.append(feature)

# 	songs[i]._timbral_features = features 
# 	songs[i]._tempo = B



# for i in range(4):
# 	timbre = np.asarray(songs[i]._timbral_features)
# 	timbre.transpose()
# 	print('Timbral feature vector: ', songs[i]._timbral_features)
# 	timbre.tofile(name ,sep=',',format='%10.5f')



# timbre1 = np.asarray(songs[0]._timbral_features)
# N = len(timbre1)
# print('SUP')

# for i in range(4):
# 	songs[i]._timbral_features = np.square(songs[i]._timbral_features)
# 	print(songs[i]._timbral_features)




# simularity = []
# for i in range(N):
# 	minimum = min(songs[0]._timbral_features[i], songs[1]._timbral_features[i])
# 	maximum = max(songs[0]._timbral_features[i], songs[1]._timbral_features[i])
# 	sim = minimum / maximum

# 	simularity.append(sim)


# simularity = np.asarray(simularity)
# simularity = np.absolute(simularity)

# simularity.tofile('simularity1.csv' ,sep=',',format='%10.5f')
# print(simularity)

# simularity = []
# for i in range(N):
# 	minimum = min(songs[2]._timbral_features[i], songs[3]._timbral_features[i])
# 	maximum = max(songs[2]._timbral_features[i], songs[3]._timbral_features[i])
# 	sim = minimum / maximum

# 	simularity.append(sim)

# simularity = np.asarray(simularity)
# simularity = np.absolute(simularity)
# simularity.tofile('simularity2.csv' ,sep=',',format='%10.5f')
# print(simularity)






















