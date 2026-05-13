import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommendation System", layout="wide")

st.title("📚 Book Recommendation System")
st.write("Recommending books based on user reading history and content similarity")

# -------------------------------
# Load and preprocess data (cached)
# -------------------------------
@st.cache_data
def load_data():
    users = pd.read_csv("Users.csv")
    books = pd.read_csv("Books.csv", encoding="latin-1")
    ratings = pd.read_csv("Ratings.csv")

    ratings = ratings.rename(columns={
        "User-ID": "user_id",
        "ISBN": "isbn",
        "Book-Rating": "rating"
    })

    books = books.rename(columns={
        "ISBN": "isbn",
        "Book-Title": "title",
        "Book-Author": "author",
        "Publisher": "publisher"
    })

    ratings = ratings[ratings["rating"] > 0]

    data = ratings.merge(books, on="isbn")

    data["text"] = (
        data["title"].fillna("") + " " +
        data["author"].fillna("") + " " +
        data["publisher"].fillna("")
    )

    book_features = (
        data[['isbn', 'title', 'author', 'publisher', 'text']]
        .drop_duplicates(subset='isbn')
        .reset_index(drop=True)
    )

    return data, book_features

# -------------------------------
# Feature generation (cached)
# -------------------------------
@st.cache_data
def build_tfidf(book_features):
    tfidf = TfidfVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.8
    )

    tfidf_matrix = tfidf.fit_transform(book_features["text"])
    return tfidf_matrix

# -------------------------------
# Recommendation logic
# -------------------------------
def get_similar_books(book_idx, tfidf_matrix, top_n=5):
    query_vec = tfidf_matrix[book_idx]
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    similar_scores = similarity_scores[similar_indices]

    return list(zip(similar_indices, similar_scores))


def recommend_books_for_user(user_id, data, book_features, tfidf_matrix, top_n=5):
    user_books = data[data["user_id"] == user_id]

    if user_books.empty:
        return None

    top_book = user_books.sort_values("rating", ascending=False).iloc[0]
    book_idx = book_features[book_features["isbn"] == top_book["isbn"]].index[0]

    similar_books = get_similar_books(book_idx, tfidf_matrix, top_n * 2)

    recs = []
    for idx, score in similar_books:
        recs.append({
            "Title": book_features.iloc[idx]["title"],
            "Author": book_features.iloc[idx]["author"],
            "Similarity Score": round(score, 3)
        })

    df = pd.DataFrame(recs).drop_duplicates(subset="Title").head(top_n)
    return df


# -------------------------------
# App execution
# -------------------------------
data, book_features = load_data()
tfidf_matrix = build_tfidf(book_features)

st.sidebar.header("User Input")

user_ids = sorted(data["user_id"].unique())
selected_user = st.sidebar.selectbox("Select User ID", user_ids)

top_n = st.sidebar.slider("Number of recommendations", 1, 10, 5)

if st.sidebar.button("Recommend Books"):
    result = recommend_books_for_user(
        selected_user,
        data,
        book_features,
        tfidf_matrix,
        top_n
    )

    if result is None:
        st.warning("No interaction history for this user.")
    else:
        st.subheader("📖 Recommended Books")
        st.table(result)

