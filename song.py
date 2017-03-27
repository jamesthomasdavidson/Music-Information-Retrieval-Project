


class Song(object):

    def __init__(self, title, artist, mash_id, song_id, filename):
        self._song = title
        self._artist = artist
        # self._release_date = metadata['release_date']
        self._mash_id = mash_id
        self._song_id = song_id
        self._filename = filename

    @property
    def name(self):
        return self._song




class Mashup(Song):

    def __init__(self, metadata, song_wav, songs):
        super(self.__class__, self).__init__(metadata, song_wav)
        self._songs = []
        for song in songs:
            self._songs.append(song)
