# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Machine Learning.
The system recommends movies similar to a selected movie based on its
overview, genres, keywords, cast, and director.

The project also provides an interactive Streamlit web application where
users can search for a movie and receive the top 10 similar movies.

---

## 🚀 Features

- 🔎 Search for movies by name
- 🎬 Select a movie from matching results
- 🤖 Content-based movie recommendations
- 📊 TF-IDF feature extraction
- 📐 Cosine similarity for finding similar movies
- 🎯 Top 10 movie recommendations
- 🖥️ Interactive Streamlit interface

---

## 🧠 Machine Learning Approach

This project uses **Content-Based Filtering**.

### 1. Data Preprocessing

Movie information such as:

- Movie overview
- Genres
- Keywords
- Cast
- Director

is extracted and combined into a single `tags` feature.

### 2. Feature Engineering

The textual movie information is converted into numerical
representations using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The model uses up to 5,000 TF-IDF features.

### 3. Similarity Calculation

**Cosine Similarity** is used to calculate how similar two movies are
based on their TF-IDF vectors.

### 4. Recommendation

For a selected movie, the system:

1. Finds its corresponding TF-IDF vector.
2. Compares it with all other movies.
3. Sorts movies based on similarity score.
4. Returns the top 10 most similar movies.

---

## 📊 Dataset

The project uses the **TMDB 5000 Movies Dataset**.

Dataset contains approximately:

- 4,803 movies
- Movie metadata
- Genres
- Keywords
- Cast
- Crew
- Movie overviews

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

---

## 📁 Project Structure

```text
movie_recommendation_system/
│
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── src/
│   └── recommender.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore