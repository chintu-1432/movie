import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ============================
# CONFIG
# ============================
API_KEY = "YOUR_TMDB_API_KEY"
BASE_URL = "https://api.themoviedb.org/3"

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# ============================
# CUSTOM CSS with Background Image
# ============================
st.markdown(
    """
    <style>
    body {
        background: url('https://res.cloudinary.com/dkx0ai3f6/image/upload/v1757430418/movie_wnonkj.jpg') no-repeat center center fixed;
        background-size: cover;
    }
    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.7);
        z-index: -1;
    }
    .movie-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 15px;
        margin: 10px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
    }
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(255,255,255,0.6);
    }
    .movie-poster {
        border-radius: 10px;
        box-shadow: 0px 0px 12px rgba(0,0,0,0.6);
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff0066, #ffcc00);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        transition: 0.4s;
    }
    .stButton>button:hover {
        transform: scale(1.1);
        background: linear-gradient(45deg, #ffcc00, #ff0066);
    }
    </style>
    <div class="overlay"></div>
    """,
    unsafe_allow_html=True
)

# ============================
# FETCH MOVIES
# ============================
def fetch_movies(language="en"):
    url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&with_original_language={language}&sort_by=popularity.desc&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("results", [])
    return []

# ============================
# RECOMMENDER
# ============================
def build_similarity_matrix(movies):
    df = pd.DataFrame(movies)
    if df.empty or "overview" not in df:
        return None, None
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["overview"].fillna(""))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim, df

def recommend(movie_title, cosine_sim, df):
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

# ============================
# UI
# ============================
st.title("🎥 Movie Recommendation System")
st.markdown("Find movies you'll love in Telugu, Hindi, or English!")

language = st.sidebar.selectbox("🌐 Select Language", ["Telugu", "Hindi", "English"])
language_map = {"Telugu": "te", "Hindi": "hi", "English": "en"}

movies = fetch_movies(language_map[language])
cosine_sim, df = build_similarity_matrix(movies)

if df is not None:
    movie_choice = st.selectbox("🎬 Choose a movie to get recommendations", df['title'].values)

    if st.button("Recommend 🎯"):
        recommendations = recommend(movie_choice, cosine_sim, df)
        st.subheader("✨ Recommended Movies")

        cols = st.columns(3)
        for i, row in recommendations.iterrows():
            with cols[i % 3]:
                movie_url = f"https://www.themoviedb.org/movie/{row['id']}"
                st.markdown(
                    f"""
                    <a href="{movie_url}" target="_blank" style="text-decoration:none; color:white;">
                        <div class="movie-card">
                            <img src="https://image.tmdb.org/t/p/w200{row['poster_path']}" class="movie-poster" width="200">
                            <h4>{row['title']} ({row.get('release_date', 'N/A')[:4]})</h4>
                            <p>{row['overview'][:120]}...</p>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True
                )
else:
    st.warning("No movies found. Try changing the language or API key.")
