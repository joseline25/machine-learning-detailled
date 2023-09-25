# Ratings data
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


# Load data on movie ratings.
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)