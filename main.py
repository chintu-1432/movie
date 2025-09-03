import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# CONFIG
# ============================
API_KEY = "YOUR_TMDB_API_KEY"  # Replace with your TMDB API key
BASE_URL = "https://api.themoviedb.org/3"

# ============================
# FETCH MOVIES FROM TMDB
# ============================
def fetch_movies(language_code="en"):
    url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&with_original_language={language_code}&sort_by=popularity.desc&page=1"
    response = requests.get(url)
    data = response.json()
    if "results" in data:
        return pd.DataFrame(data["results"])
    return pd.DataFrame()

# ============================
# BUILD CONTENT-BASED RECOMMENDER
# ============================
def build_recommendations(df, movie_title, top_n=6):
    if df.empty or "title" not in df:
        return []

    # Fill missing genres/overview
    df["overview"] = df["overview"].fillna("")

    # Combine genres + overview for better recommendations
    df["features"] = df["overview"] + " " + df["genre_ids"].astype(str)

    # Convert text to vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["features"])

    # Compute similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Build title-to-index map
    indices = pd.Series(df.index, index=df["title"].str.lower())

    movie_title = movie_title.lower()
    if movie_title not in indices:
        return []

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    return df.iloc[movie_indices]

# ============================
# STREAMLIT APP
# ============================
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Custom CSS for frontend styling + animations
st.markdown("""
    <style>
        .movie-card {
            background-color: #1e1e1e;
            color: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            margin-bottom: 20px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: fadeIn 1s ease;
        }
        .movie-card:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 18px rgba(255,75,75,0.6);
        }
        .movie-title {
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
        }
        .movie-year {
            font-size: 14px;
            color: #ccc;
        }
        .overview {
            font-size: 13px;
            margin-top: 10px;
            color: #ddd;
        }
        .stButton>button {
            background: linear-gradient(90deg, #FF4B4B, #FF8C00);
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            transition: all 0.3s ease;
            font-weight: bold;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #FF8C00, #FF4B4B);
            transform: scale(1.05);
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(15px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🎬 Movie Recommendation System 🎥</h1>
    <p style='text-align: center; color: #888;'>Find similar movies in English, Hindi, and Telugu</p>
""", unsafe_allow_html=True)

st.sidebar.header("🔎 Search & Recommend")

# Language Selector
language_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
selected_language = st.sidebar.selectbox("Choose Language", list(language_map.keys()))

# Fetch Movies
df_movies = fetch_movies(language_map[selected_language])

if df_movies.empty:
    st.error("⚠️ Could not fetch movies. Check your TMDB API key.")
else:
    movie_list = df_movies["title"].tolist()
    selected_movie = st.sidebar.selectbox("🎞️ Choose a movie", movie_list)

    if st.sidebar.button("✨ Get Recommendations"):
        recommendations = build_recommendations(df_movies, selected_movie)
        if len(recommendations) == 0:
            st.warning("No recommendations found.")
        else:
            st.subheader(f"Movies similar to **{selected_movie}** ({selected_language}):")
            cols = st.columns(3)
            for i, row in enumerate(recommendations.itertuples(), 1):
                with cols[(i-1) % 3]:
                    poster_url = f"https://image.tmdb.org/t/p/w300{row.poster_path}" if row.poster_path else ""
                    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                    if poster_url:
                        st.image(poster_url, width=180, caption="")
                    st.markdown(f"<div class='movie-title'>{row.title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='movie-year'>{row.release_date if row.release_date else 'Unknown Year'}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='overview'>{row.overview[:150]}...</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
