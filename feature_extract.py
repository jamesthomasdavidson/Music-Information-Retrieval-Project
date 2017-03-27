import pickle
from song import Song
import ffmpy
import os

savedir = "gooddata"
if not os.path.exists(savedir):
    os.makedirs(savedir)

def make_savepath(id, savedir=savedir):
    return os.path.join(savedir, "%s.wav" % id)

with open('songs_pickle.pck', 'rb') as f:
    songs = pickle.load(f, encoding='latin1')



# for song in songs:

for song in songs:
	save_path = make_savepath(song._song_id)
	ff = ffmpy.FFmpeg(
		global_options=["-i " + song._filename + " -ac 1 " + save_path]
		# inputs={song._filename: None},
		# outputs={save_path: None},
		)

	ff.run()