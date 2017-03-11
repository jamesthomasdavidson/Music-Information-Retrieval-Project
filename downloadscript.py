import youtube_dl
import pandas as pd
import os
import traceback

CSV = "videos.csv"

# create directory
savedir = "data"
if not os.path.exists(savedir):
    os.makedirs(savedir)

def make_savepath(title, artist, savedir=savedir):
    return os.path.join(savedir, "%s%s.wav" % (title, artist))

# create YouTube downloader
options = {
    'verbose': False,
    'outtmpl': '%(id)s.%(ext)s',
    'format': 'bestaudio/best', # choice of quality
    'extractaudio' : True,      # only keep the audio
    'audioformat' : "wav",      # convert to wav
    'outtmpl': '%(id)s',        # name the file the ID of the video       
    'noplaylist' : True,
    }       # only download single song, not playlist
ydl = youtube_dl.YoutubeDL(options)

with ydl:

    # read in videos CSV with pandas
    df = pd.read_csv(CSV, sep=";", skipinitialspace=True)
    df.Link = df.Link.map(str.strip)  # strip space from URLs

    # for each row, download
    for _, row in df.iterrows():
        print ("Downloading: {} from {}...".format(row.Title,row.Link ) )

        # download location, check for progress
        savepath = make_savepath(row.Title, row.Artist)
        try:
            os.stat(savepath)
            print ("{} already downloaded, continuing...".format(savepath))
            continue

        except OSError:
            # download video
            try:
                result = ydl.extract_info(row.Link, download=True)
                os.rename(result['id'], savepath)
                print ("Downloaded and converted {} successfully!".format(savepath))

            except Exception as e:
                print ("Can't download audio!\n")


