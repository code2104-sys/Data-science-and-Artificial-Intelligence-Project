# Data-science-and-Artificial-Intelligence-Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

print("Libraries imported successfully.")

     
Libraries imported successfully.

# Load movies.dat file with specified encoding
movies = pd.read_csv('/content/movies.dat', sep='::', engine='python', header=None, names=['movieId', 'title', 'genres'], encoding='latin1')

# Load ratings.dat file with specified encoding
ratings = pd.read_csv('/content/ratings.dat', sep='::', engine='python', header=None, names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin1')

     

# Merge movies and ratings DataFrames
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Create a user-movie matrix
user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# Fill missing values with 0
user_movie_matrix.fillna(0, inplace=True)

     

from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Compute cosine similarity between movies
movie_similarity = cosine_similarity(user_movie_matrix.T)
movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

     

def recommend_movies(user_id, num_recommendations=5):
    # Get the similarity scores for the user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:num_recommendations+1]

    # Get the movies watched by similar users
    similar_users_movies = user_movie_matrix.loc[similar_users].mean().sort_values(ascending=False)

    # Exclude movies already watched by the user
    watched_movies = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    recommendations = similar_users_movies.drop(watched_movies).head(num_recommendations)

    return recommendations

     

# Example: Recommend movies for user 1
print(recommend_movies(1))

     
title
Little Mermaid, The (1989)          4.2
Jungle Book, The (1967)             3.4
Silence of the Lambs, The (1991)    3.4
Lady and the Tramp (1955)           2.6
101 Dalmatians (1961)               2.4
dtype: float64
Explanation
The movie recommendation system uses cosine similarity to find similar users and recommend movies based on the ratings of those similar users. The model evaluates the similarity between users and between movies, and generates a list of recommended movies that the user has not yet watched. The effectiveness of the recommendations can be improved by incorporating more sophisticated algorithms and additional features such as genre, director, and user demographics.
