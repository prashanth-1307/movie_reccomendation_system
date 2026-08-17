import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")


# Merge both datasets
movies = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)


# Select only the columns we need
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

# Rename title_x to title
movies.rename(columns={"title_x": "title"}, inplace=True)


# Handle missing overview
movies["overview"] = movies["overview"].fillna("")


# -----------------------------
# Feature Extraction Functions
# -----------------------------

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

# recommendation function ========================

def recommend(movie):
    if movie not in movie_index:
        print("Movie not found.")
        return

    index = movie_index[movie]

    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:11]

    print(f"\nRecommendations for {movie}:\n")

    for i, score in movies_list:
        print(f"{movies.iloc[i].title}  →  {score:.3f}")


# ===============

def clean_tags(text):
    return text.lower().replace(" ", "")




# -----------------------------
# Feature Engineering
# -----------------------------

movies["genres"] = movies["genres"].apply(convert)

movies["keywords"] = movies["keywords"].apply(convert)

movies["cast"] = movies["cast"].apply(convert3)

movies["crew"] = movies["crew"].apply(fetch_director)

movies.rename(columns={"crew": "director"}, inplace=True)

movies["overview"] = movies["overview"].apply(lambda x: x.split())


# -----------------------------
# Create Tags
# -----------------------------

movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["director"].apply(lambda x: [x])
)


# Convert list into string
movies["tags"] = movies["tags"].apply(lambda x: " ".join(x))


# Keep only required columns
movies = movies[
    [
        "movie_id",
        "title",
        "tags"
    ]
]

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(movies["tags"])

# print("TF-IDF matrix shape:", tfidf_matrix.shape)

similarity = cosine_similarity(tfidf_matrix)

# print("Similarity matrix shape:", similarity.shape)

movie_index = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()


recommend("Avatar")

# Check result
# print(movies.head())