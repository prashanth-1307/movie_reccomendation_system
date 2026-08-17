import streamlit as st

from src.recommender import movies, recommend


# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# Title
st.title("🎬 Movie Recommendation System")

st.write(
    "Select a movie and get 10 similar movie recommendations "
    "using content-based filtering."
)


# Movie selection
st.subheader("🔎 Search for a movie")

search_query = st.text_input(
    "Enter movie name:",
    placeholder="e.g. Batman, Avatar, Inception..."
)

movie = None

if search_query:

    matching_movies = movies[
        movies["title"].str.contains(
            search_query,
            case=False,
            na=False
        )
    ]

    if len(matching_movies) > 0:

        movie = st.selectbox(
            "Select a movie:",
            matching_movies["title"].values
        )

    else:
        st.warning("No movies found. Try another search.")


# Recommendation button
if st.button("🎯 Recommend"):

    if movie is None:

        st.warning("Please search and select a movie first.")

    else:

        recommendations = recommend(movie)

        st.subheader(f"Movies similar to **{movie}**")

        col1, col2 = st.columns(2)

        for i, recommendation in enumerate(recommendations):

            if i % 2 == 0:

                with col1:

                    st.markdown(
                        f"### {i + 1}. {recommendation['title']}"
                    )

                    st.write(
                        f"Similarity Score: "
                        f"**{recommendation['score']:.3f}**"
                    )

                    st.divider()

            else:

                with col2:

                    st.markdown(
                        f"### {i + 1}. {recommendation['title']}"
                    )

                    st.write(
                        f"Similarity Score: "
                        f"**{recommendation['score']:.3f}**"
                    )

                    st.divider()


st.divider()

st.subheader("🧠 How the Recommendation System Works")

st.write("""
This system uses content-based filtering to recommend movies based
on their similarity.

1. Movie metadata such as genres, keywords, cast, director, and
   overview are combined into a single feature called `tags`.

2. TF-IDF Vectorization converts the movie tags into numerical
   feature vectors.

3. Cosine Similarity measures the similarity between movies.

4. For a selected movie, the system returns the 10 movies with
   the highest similarity scores.
""")


st.subheader("📊 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Movies", len(movies))

with col2:
    st.metric("TF-IDF Features", 5000)

with col3:
    st.metric("Recommendations", 10)