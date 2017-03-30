import pickle
from song import Song, Mashup
import ffmpy
import os
import marsyas

with open('songs_pickle.pck', 'rb') as f:
    songs = pickle.load(f, encoding='latin1')

savedir = "gooddata"
if not os.path.exists(savedir):
    os.makedirs(savedir)

def make_savepath(id, savedir=savedir):
    return os.path.join(savedir, "%s.wav" % id)

# for song in songs:

for song in songs:
	save_path = make_savepath(song._song_id)
	if (os.path.isfile(save_path)):
		print('continuing...')
	else:
		ff = ffmpy.FFmpeg(
			global_options=["-i " + song._filename + " -ac 1 " + save_path]
			)

		ff.run()


mashups = []
N= len(songs)

for i in range(N):
	if i%2==0:
		mashup = []
		if(i!= N-1):
			j = i+1
			song1 = songs[j]
			song2 = songs[j-1]

		mashup.append(song1)
		mashup.append(song2)

		m = Mashup(songs=mashup)

		mashups.append(m)









	


