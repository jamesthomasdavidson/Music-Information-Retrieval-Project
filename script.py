from song import *

with open('songs_pickle.pck', 'rb') as f:
    songs = pickle.load(f)

# for song in songs:
#     #print(song._filename)
#     if not os.path.exists(song._filename):
#         continue
#     [Fs, x] = audioBasicIO.readAudioFile(song._filename)
#     F = audioFeatureExtraction.stFeatureExtraction(x.tolist(), Fs, 0.050*Fs, 0.050*Fs);
#     print("processing song number ", song._song_id )
#     B = audioFeatureExtraction.beatExtraction(F, .050)
#
#     features = []
#     for f in F:
#     	feature = np.mean(f)
#     	features.append(feature)
#
#     song._timbral_features = features
#     song._tempo = B


# f = open('songs_pickle.pck', mode='wb')
# pickle.dump(songs, file=f, protocol=2)

Mashups = []
for i in range(0, 80, 2):
    Mashups.append(Mashup(songs[i], songs[i+1]))

m = MashupMatrix(Mashups)
#m.out(val = 'std')
print(m.vector(2,1))
