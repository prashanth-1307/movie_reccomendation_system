import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# Load Data
# -----------------------------------

movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")


# -----------------------------------
# Merge datasets
# -----------------------------------

movies = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)


# -----------------------------------
# Select required columns
# -----------------------------------

movies = movies[
    [
        "movie_id",
        "title_x",
        "overview",
        "genres",
        "keywords",
        "cast",
        "crew"
    ]
]

movies.rename(columns={"title_x": "title"}, inplace=True)

movies["overview"] = movies["overview"].fillna("")


# -----------------------------------
# Feature extraction
# -----------------------------------

def convert(obj):
    result = []

    for item in ast.literal_eval(obj):
        result.append(item["name"])

    return result


def convert3(obj):
    result = []

    for item in ast.literal_eval(obj):
        result.append(item["name"])

    return result[:3]


def fetch_director(obj):
    for item in ast.literal_eval(obj):
        if item["job"] == "Director":
            return item["name"]

    return ""


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert3)

movies["crew"] = movies["crew"].apply(fetch_director)

movies.rename(columns={"crew": "director"}, inplace=True)

movies["overview"] = movies["overview"].apply(lambda x: x.split())


# -----------------------------------
# Create tags
# -----------------------------------

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["director"].apply(lambda x: [x])
)

movies["tags"] = movies["tags"].apply(lambda x: " ".join(x))


# -----------------------------------
# Keep required columns
# -----------------------------------

movies = movies[
    [
        "movie_id",
        "title",
        "tags"
    ]
]


# -----------------------------------
# TF-IDF
# -----------------------------------

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(movies["tags"])


# -----------------------------------
# Cosine Similarity
# -----------------------------------

similarity = cosine_similarity(tfidf_matrix)


# -----------------------------------
# Movie Index
# -----------------------------------

movie_index = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()


# -----------------------------------
# Recommendation Function
# -----------------------------------

def recommend(movie):

    if movie not in movie_index:
        return []

    index = movie_index[movie]

    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:11]

    recommendations = []

    for i, score in movies_list:
        recommendations.append({
            "movie_id": movies.iloc[i]["movie_id"],
            "title": movies.iloc[i]["title"],
            "score": float(score)
        })

    return recommendations
