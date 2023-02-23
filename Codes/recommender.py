from typing import Tuple, List

from implicit import recommender_base, als
from scipy import sparse
from pathlib import Path

from Data import ArtistRetriever, load_user_artists, ARTIST_DATA_PATH, USER_ARTIST_DATA_PATH


class ImplicitRecommender:
    """computes recommendation for a given user
    using the implicit library

    :argument
        - artist_retreiver: ArtistRetriever instance
        - implicit_model: implicit model
    """

    def __init__(
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: recommender_base.RecommenderBase,
        ):

        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: sparse.csr_matrix):
        """Fit the model to the user artists matrix"""
        self.implicit_model.fit(user_artists_matrix)

    def recommend(self, user_id: int, user_artists_matrix: sparse.csr_matrix, n: int = 10) -> Tuple[List[str], List[float]]:
        """return the top n recommendation for the given user"""
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[n], N=n
        )

        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores


if __name__ == "__main__":

    # load user artist
    user_artists = load_user_artists(USER_ARTIST_DATA_PATH)

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path(ARTIST_DATA_PATH))

    # instantiate ALS using implicit
    implicit_model = als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    # instantiate recommender, fit and recommend
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    artists, scores = recommender.recommend(2, user_artists, n=5)

    # print the result
    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")




