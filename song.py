


class Song(object):

    def __init__(self, metadata, song_wav):
        self._song = song_wav
        self._artist = metadata['artist']
        self._album = metadata['album']
        self._release_date = metadata['release_date']
        self._song = metadata['song']



class Mashup(Song):

    def __init__(self, metadata, song_wav, songs):
        super(self.__class__, self).__init__(metadata, song_wav)
        self._songs = []
        for song in songs:
            self._songs.append(song)
