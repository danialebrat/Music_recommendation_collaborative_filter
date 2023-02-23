"""This module features functions and classes to manipulate data for the
collaborative filtering algorithm.
"""



from pathlib import Path

from scipy import sparse
import pandas as pd


ARTIST_DATA_PATH = "../dataset/hetrec2011-lastfm-2k/artists.dat"
USER_ARTIST_DATA_PATH = "../dataset/hetrec2011-lastfm-2k/user_artists.dat"

def load_user_artists(user_artists_file: Path):
    """
    load the user artists file and return a user-artist matrix in csr format
    """


    user_artists = pd.read_csv(user_artists_file, sep= "\t")
    user_artists.set_index(["userID", "artistID"], inplace=True)
    coo = sparse.coo_matrix(
        (user_artists.weight.astype(float),
         (
             user_artists.index.get_level_values(0),
             user_artists.index.get_level_values(1)
         ),
        )
    )
    return coo.tocsr()



class ArtistRetriever:
    """ gets the artist name using the artist ID"""

    def __init__(self):
        self._artist_df = None

    def get_artist_name_from_id(self, artist_id: int):
        """return the artist name from the artist_ID."""
        return self._artist_df.loc[artist_id, "name"]


    def load_artists(self, artist_file:Path) -> None:

        """load the artists file and stores it as a pandas dataframe in a private attribute"""
        artists_df = pd.read_csv(artist_file, sep= "\t")
        artists_df = artists_df.set_index("id")
        self._artist_df = artists_df



if __name__ == "__main__":

    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path(ARTIST_DATA_PATH))
    artist = artist_retriever.get_artist_name_from_id(2)
    print(artist)