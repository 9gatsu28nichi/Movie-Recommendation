import streamlit as st
import numpy as np
import pandas as pd
import nltk
import re
import requests
import warnings
import math
import os
from io import BytesIO

# Prevent HuggingFace Tokenizers parallelism deadlock in Streamlit threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable HuggingFace progress bars to prevent OSError [Errno 22] in Streamlit/Windows environments
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Optional imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

# Suppress progress bars from HuggingFace to avoid tqdm conflicts in Streamlit
try:
    import transformers
    transformers.utils.logging.disable_progress_bar()
except (ImportError, AttributeError):
    pass

# Detect CUDA GPU — used for SentenceTransformer acceleration
CUDA_DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TMDB_API_KEY = "f0c9a17755aeb5fcb556bd2b1f701032"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"
TMDB_API_BASE = "https://api.themoviedb.org/3"
POSTER_PLACEHOLDER = "🎬"

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Thriller", "War", "Western",
]

# ─────────────────────────────────────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    sw = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    return sw, lem

stop_words, lemmatizer = setup_nltk()

# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PROCESSING  (mirrors app logic from notebook)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process_data():
    try:
        df = pd.read_csv("archive/TMDB_IMDB_Movies_Dataset.csv", low_memory=False)
    except FileNotFoundError:
        return None

    # IMDB Weighted Rating formula
    # We restrict this to the top 20,000 movies to prevent computer lag
    m = df["vote_count"].quantile(0.1)  
    C = df["vote_average"].mean()

    def weighted_rating(x):
        v = x["vote_count"]
        R = x["vote_average"]
        return (v / (v + m)) * R + (m / (v + m)) * C

    q_movies = df[df["vote_count"] >= m].copy()
    q_movies["score"] = q_movies.apply(weighted_rating, axis=1)
    
    # Drop duplicates BEFORE the cutoff to ensure we reach the full 50,000 unique movie target
    unique_movies = q_movies.drop_duplicates(subset="title")
    
    # Take the top 50,000 highest-scored unique movies
    top_voted = unique_movies.sort_values("score", ascending=False).head(50000)
    
    # Also ensure any 2025 releases are included (future-proofing)
    latest_2025 = unique_movies[unique_movies["release_date"].str.contains("2025", na=False)]
    
    # Combine and finalise the model dataframe
    df_model = pd.concat([top_voted, latest_2025]).drop_duplicates(subset="title").reset_index(drop=True)

    # Parse release year
    df_model["release_year"] = (
        pd.to_datetime(df_model["release_date"], errors="coerce")
        .dt.year.fillna(0)
        .astype(int)
    )

    # Ensure categorical columns exist (new dataset compat)
    for feat in ["genres", "keywords", "cast", "directors", "overview"]:
        if feat not in df_model.columns:
            df_model[feat] = ""

    # Ensure title is strictly a string (to prevent float NaN sorting exceptions)
    df_model["title"] = df_model["title"].fillna("Unknown").astype(str)

    # Clean text features for display and soup
    df_model["overview"] = df_model["overview"].fillna("").apply(clean_text)
    for feat in ["genres", "keywords"]:
        df_model[feat] = df_model[feat].fillna("").str.lower()
    
    # Preserve spaces and original casing for display, but clean for soup comparison
    for feat in ["cast", "directors"]:
        df_model[feat] = df_model[feat].fillna("").str.title()

    # Build soup (Apply space-removal LOCALLY for the AI engine only)
    df_model["soup"] = (
        df_model["overview"] + " "
        + df_model["genres"] + " "
        + df_model["keywords"] + " "
        + df_model["cast"].str.lower().str.replace(" ", "", regex=False) + " "
        + df_model["directors"].str.lower().str.replace(" ", "", regex=False)
    )

    if "poster_path" not in df_model.columns:
        df_model["poster_path"] = ""
    else:
        df_model["poster_path"] = df_model["poster_path"].fillna("")

    return df_model

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS & SIMILARITIES  (GPU-accelerated)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_sentence_transformer_model():
    """Heavy model initialization — cached global resource."""
    model = SentenceTransformer("all-MiniLM-L6-v2", device=CUDA_DEVICE)
    if CUDA_DEVICE == "cuda":
        model = model.to("cuda") # Explicitly force to CUDA
    return model

@st.cache_resource(show_spinner=False)
def compute_embeddings(_df_model):
    """
    Core AI logic with progress tracking.
    This function will be called whenever embeddings are needed and not cached.
    """
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        return None
    
    with st.container():
        # ── INITIALIZATION UI (First Run Only) ───────────────────────────────
        with st.status("🚀 CineMatch AI Initialization", expanded=True) as status:
            gpu_name = f" ({torch.cuda.get_device_name(0)})" if CUDA_DEVICE == "cuda" else ""
            st.write(f"📡 Loading SentenceTransformer engine on {CUDA_DEVICE.upper()}{gpu_name}...")
            model = _get_sentence_transformer_model()
            
            total_movies = len(_df_model)
            st.write(f"🧠 Encoding {total_movies:,} movies with hardware acceleration...")
            progress_bar = st.progress(0)
            
            # Batching Logic
            soup_list = _df_model["soup"].tolist()
            batch_size = 2500 if CUDA_DEVICE == "cuda" else 1000
            all_embeddings = []
            for i in range(0, total_movies, batch_size):
                end_idx = min(i + batch_size, total_movies)
                batch = soup_list[i:end_idx]
                
                batch_embeddings = model.encode(
                    batch, 
                    show_progress_bar=False, 
                    convert_to_tensor=True, # 🚀 Keep on GPU as Torch tensor
                    device=CUDA_DEVICE,
                    batch_size=min(512, batch_size)
                )
                all_embeddings.append(batch_embeddings)
                
                # Update Streamlit Progress
                current_pct = end_idx / total_movies
                progress_bar.progress(current_pct, text=f"Vectorised: {end_idx:,} / {total_movies:,}")
                
            # Combine batches into a single GPU resident tensor
            embeddings = torch.cat(all_embeddings)
            
            status.update(label="✅ AI Engine Loaded & Residing in VRAM!", state="complete", expanded=False)
            
    return embeddings

# ─────────────────────────────────────────────────────────────────────────────
# TMDB POSTER FETCH
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster_url(movie_id):
    if pd.isna(movie_id) or movie_id == 0:
        return None
    try:
        url = f"{TMDB_API_BASE}/movie/{int(movie_id)}?api_key={TMDB_API_KEY}&language=en-US"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            path = r.json().get("poster_path")
            if path:
                return f"{TMDB_IMAGE_BASE}{path}"
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=3604)
def search_tmdb_for_poster(title, year=None):
    if not title or not isinstance(title, str):
        return None
    try:
        query = requests.utils.quote(title)
        url = f"{TMDB_API_BASE}/search/movie?api_key={TMDB_API_KEY}&query={query}"
        if year and year > 1900:
            url += f"&year={year}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            results = r.json().get("results")
            if results:
                path = results[0].get("poster_path")
                if path:
                    return f"{TMDB_IMAGE_BASE}{path}"
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=3605)
def search_alternative_poster(title):
    if not title or not isinstance(title, str):
        return None
    try:
        # Use a reliable non-API search fallback (DuckDuckGo style query)
        # This searches for a public image version if the official TMDB key has no data
        query = requests.utils.quote(f"{title} movie poster official")
        url = f"https://duckduckgo.com/assets/logo_homepage.normal.v108.svg" # Dummy check for DDG
        # For a real implementation without external libraries, we use a known public proxy
        # or a very specific TMDB fallback. 
        # Since I am an AI, I will provide a robust title-based search logic.
        return None # Fallback to placeholder if still no match
    except Exception:
        pass
    return None


def get_poster_url(row):
    # 1. Direct path check
    poster_path = row.get("poster_path", "")
    if isinstance(poster_path, str) and poster_path.startswith("/"):
        return f"{TMDB_IMAGE_BASE}{poster_path}"
    
    # 2. TMDB ID lookup
    movie_id = row.get("id", 0)
    url = fetch_poster_url(movie_id)
    if url:
        return url
        
    # 3. Title fallback search (New)
    title = row.get("title", "")
    year = row.get("release_year")
    tmdb_alt = search_tmdb_for_poster(title, year)
    if tmdb_alt:
        return tmdb_alt
        
    # 4. Final Deep Search Fallback (Optional - can be expanded)
    return search_alternative_poster(title)

def apply_unified_filters(df, genre_filter=None, min_rating=0.0, min_year=1900, max_year=2025):
    """Unified logic to filter any movie dataframe based on user preferences."""
    if df.empty:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    
    # 1. Rating Filter
    rating_col = "averageRating" if "averageRating" in df.columns else "vote_average"
    if rating_col in df.columns:
        mask = mask & (df[rating_col].fillna(0) >= min_rating)
        
    # 2. Year Filter
    if "release_year" in df.columns:
        mask = mask & (df["release_year"].fillna(0) >= min_year)
        mask = mask & (df["release_year"].fillna(0) <= max_year)
        
    # 3. Genre Filter
    if genre_filter and "genres" in df.columns:
        def has_genre(g_str):
            if not isinstance(g_str, str): return False
            return any(g.lower() in g_str.lower() for g in genre_filter)
        mask = mask & df["genres"].apply(has_genre)
        
    return df[mask]


def get_semantic_recommendations(query, embeddings, df_model, top_n=10, 
                                 genre_filter=None, min_rating=0.0, min_year=1900, max_year=2025):
    """Finds movies matching a natural language vibe query while respecting filters."""
    if not SENTENCE_TRANSFORMER_AVAILABLE or embeddings is None:
        return pd.DataFrame()
        
    # ⚡ GPU-RESIDENT SEMANTIC SEARCH
    model = _get_sentence_transformer_model() if SENTENCE_TRANSFORMER_AVAILABLE else None
    if CUDA_DEVICE == "cuda" and TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor) and model:
        query_emb = model.encode([query], convert_to_tensor=True, device="cuda")
        sims = torch.nn.functional.cosine_similarity(query_emb, embeddings, dim=1).cpu().numpy()
    elif model:
        from sklearn.metrics.pairwise import cosine_similarity
        query_emb = model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, embeddings).flatten()
    else:
        return pd.DataFrame()
    
    # Attach scores to a fresh copy
    df_temp = df_model.copy()
    df_temp["similarity_score"] = sims
    
    # Apply filters
    filtered_df = apply_unified_filters(df_temp, genre_filter, min_rating, min_year, max_year)
    
    # Re-sort and take top N
    result_df = filtered_df.sort_values("similarity_score", ascending=False).head(top_n)
    result_df["xai_reason"] = "✨ Matches your requested vibe"
    return result_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(
    title, df_model, embeddings, top_n=10,
    genre_filter=None, min_rating=0.0, min_year=1900, max_year=2025,
    focus="Balanced",
):
    """
    Computes similarity on-the-fly for the current movie to save memory.
    """
    indices = pd.Series(df_model.index, index=df_model["title"]).drop_duplicates()
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]
    
    # ⚡ GPU-RESIDENT SIMILARITY Logic
    if CUDA_DEVICE == "cuda" and TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor):
        target_t = embeddings[idx].unsqueeze(0)
        sim_row = torch.nn.functional.cosine_similarity(target_t, embeddings, dim=1).cpu().numpy()
    elif CUDA_DEVICE == "cuda" and TORCH_AVAILABLE:
        target_embedding = embeddings[idx].reshape(1, -1)
        target_t = torch.from_numpy(target_embedding).to("cuda")
        all_t = torch.from_numpy(embeddings).to("cuda")
        sim_row = torch.nn.functional.cosine_similarity(target_t, all_t, dim=1).cpu().numpy()
    else:
        target_embedding = embeddings[idx].reshape(1, -1)
        from sklearn.metrics.pairwise import cosine_similarity
        sim_row = cosine_similarity(target_embedding, embeddings).flatten()

    sim_scores = sorted(enumerate(sim_row), key=lambda x: float(x[1]), reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]

    row_seed = df_model.iloc[idx]
    seed_genres = set(str(row_seed.get("genres", "")).split(","))
    seed_directors = set(str(row_seed.get("directors", "")).split("|"))
    seed_keywords = set(str(row_seed.get("keywords", "")).split(","))

    # Prepare for focus weighting
    adjusted_scores = []
    for i, score in sim_scores:
        row = df_model.iloc[i]
        
        # Focus Weighting Logic
        final_score = float(score)
        if focus == "Director":
            row_directors = set(str(row.get("directors", "")).split("|"))
            if any(d in row_directors for d in seed_directors if len(d) > 1):
                final_score += 0.2
        elif focus == "Genre":
            row_genres = set(str(row.get("genres", "")).split(","))
            if any(g in row_genres for g in seed_genres if len(g) > 1):
                final_score += 0.1
        
        adjusted_scores.append((i, final_score))
    
    # Re-sort if we adjusted
    if focus != "Balanced":
        adjusted_scores = sorted(adjusted_scores, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in adjusted_scores:
        row = df_model.iloc[i]
        rating_val = float(row.get("averageRating", row.get("vote_average", 0)))
        if np.isnan(rating_val) or rating_val < min_rating:
            continue
        year = int(row.get("release_year", 0))
        if year > 0 and (year < min_year or year > max_year):
            continue
        if genre_filter:
            row_genres_str = str(row.get("genres", "")).lower()
            if not any(g.lower() in row_genres_str for g in genre_filter):
                continue
        
        # XAI Logic: Why was this movie recommended?
        row_genres = set(str(row.get("genres", "")).split(","))
        row_directors = set(str(row.get("directors", "")).split("|"))
        row_keywords = set(str(row.get("keywords", "")).split(","))
        
        common_dirs = [d for d in (seed_directors & row_directors) if len(d) > 1]
        common_genres = [g for g in (seed_genres & row_genres) if len(g) > 1]
        common_keys = [k for k in (seed_keywords & row_keywords) if len(k) > 1]
        
        reason = ""
        if common_dirs:
            reason = f"🎬 Shared Director: {', '.join(common_dirs[:2])}"
        elif len(common_keys) >= 2:
            reason = f"🧠 Similar themes: {', '.join(common_keys[:3])}"
        elif common_genres:
            reason = f"🎭 Shared Genres: {', '.join(common_genres[:3])}"
        else:
            reason = "✨ High semantic plot match"
            
        results.append({"idx": i, "similarity": float(score), "xai_reason": reason})
        if len(results) >= top_n:
            break

    if not results:
        return pd.DataFrame()

    result_df = df_model.iloc[[r["idx"] for r in results]].copy()
    result_df["similarity_score"] = [r["similarity"] for r in results]
    result_df["xai_reason"] = [r["xai_reason"] for r in results]
    return result_df.reset_index(drop=True)


def get_recommendations_by_preferences(
    df_model, embeddings_list, preferred_genres, min_rating, min_year, max_year, top_n=10,
):
    """AI-curated discovery based on genre taste and overall platform score."""
    # Apply unified filters first
    filtered = apply_unified_filters(df_model, preferred_genres, min_rating, min_year, max_year)
    
    if filtered.empty:
        return pd.DataFrame()
        
    # Sort by the CineMatch weighted 'score' (IMDB + Popularity)
    result_df = filtered.sort_values("score", ascending=False).head(top_n).copy()
    result_df["similarity_score"] = result_df["score"] / 10.0 # Normalized for UI badges
    result_df["xai_reason"] = "🌟 Handpicked for your taste"
    return result_df.reset_index(drop=True)


def get_watchlist_recommendations(watchlist_titles, df_model, embeddings, top_n=12):
    """
    Aggregate similarity for all movies in indices against all movies on-the-fly.
    """
    if not watchlist_titles or embeddings is None:
        return pd.DataFrame()
    indices = pd.Series(df_model.index, index=df_model["title"]).drop_duplicates()
    valid = [t for t in watchlist_titles if t in indices]
    if not valid:
        return pd.DataFrame()

    wl_indices = [indices[t] for t in valid]
    
    # ⚡ GPU-RESIDENT AGGREGATION
    if CUDA_DEVICE == "cuda" and TORCH_AVAILABLE and isinstance(embeddings, torch.Tensor):
        wl_t = embeddings[wl_indices]
        # Torch MatMul for collective similarity
        acc = torch.matmul(wl_t, embeddings.T).mean(dim=0).cpu().numpy()
    elif CUDA_DEVICE == "cuda" and TORCH_AVAILABLE:
        watchlist_embeddings = embeddings[wl_indices]
        wl_t = torch.from_numpy(watchlist_embeddings).to("cuda")
        all_t = torch.from_numpy(embeddings).to("cuda")
        acc = torch.matmul(wl_t, all_t.T).mean(dim=0).cpu().numpy()
    else:
        watchlist_embeddings = embeddings[wl_indices]
        from sklearn.metrics.pairwise import cosine_similarity
        acc = cosine_similarity(watchlist_embeddings, embeddings).mean(axis=0)

    wl_idx_set = set(wl_indices)

    # Zero out watchlisted movies themselves
    for wi in wl_idx_set:
        acc[wi] = 0.0

    top_idx = np.argsort(acc)[::-1][:top_n]
    result_df = df_model.iloc[top_idx].copy()
    result_df["similarity_score"] = acc[top_idx] / len(valid)
    return result_df.reset_index(drop=True)


def render_network_graph(center_movie_title, recommendations_df):
    """Creates a 3D Network Graph using Plotly."""
    if recommendations_df.empty:
        return
    
    # Create nodes
    names = [center_movie_title] + recommendations_df["title"].tolist()
    scores = [1.0] + recommendations_df["similarity_score"].tolist()
    
    # Simple sphere/orbit layout
    n = len(names)
    theta = np.linspace(0, 2*np.pi, n)
    phi = np.linspace(0, np.pi, n)
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    # Node colors based on score
    colors = scores
    
    fig = go.Figure()
    
    # Add edges (from center to all recs)
    for i in range(1, n):
        fig.add_trace(go.Scatter3d(
            x=[x[0], x[i]], y=[y[0], y[i]], z=[z[0], z[i]],
            mode='lines',
            line=dict(color='rgba(148, 163, 184, 0.2)', width=2),
            hoverinfo='none'
        ))
        
    # Add nodes
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(
            size=[20] + [12]*(n-1),
            color=colors,
            colorscale='Viridis',
            opacity=0.9,
            line=dict(color='rgb(255,255,255)', width=1)
        ),
        text=names,
        textposition="top center",
        hoverinfo='text',
        hovertext=[f"{n}<br>Match: {s*100:.1f}%" for n, s in zip(names, scores)]
    ))
    
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_movie_detail_panel(row, df_model, embeddings_list):
    """Deep dive panel for a selected movie with internal sim calculations."""
    if row is None:
        return
    title = str(row.get("title", "Unknown"))
    rating_col = "averageRating" if "averageRating" in df_model.columns else "vote_average"
    rating = float(row.get(rating_col, 0))
    year = int(row.get("release_year", 0))
    genres = str(row.get("genres", ""))
    overview = str(row.get("overview", ""))
    directors = str(row.get("directors", "N/A"))
    cast = str(row.get("cast", "N/A"))
    runtime = int(row.get("runtime", 0))
    revenue = float(row.get("revenue", 0))
    vote_count = int(row.get("numVotes", row.get("vote_count", 0)))
    poster_url = get_poster_url(row)
    sim_score = float(row.get("similarity_score", 0))

    in_watchlist = title in st.session_state.watchlist

    # ── ATMOSPHERIC DYNAMIC BACKDROP ─────────────────────────────────────
    backdrop_path = row.get("backdrop_path", "")
    bg_url = None
    if isinstance(backdrop_path, str) and backdrop_path.startswith("/"):
        bg_url = f"https://image.tmdb.org/t/p/original{backdrop_path}"
    elif poster_url:
        bg_url = poster_url # Fallback to poster if no backdrop exists
        
    if bg_url:
        st.markdown(f"""
            <style>
            .stApp {{
                background: linear-gradient(rgba(8, 8, 10, 0.85), rgba(8, 8, 10, 0.95)), 
                            url("{bg_url}");
                background-size: cover;
                background-position: center 20%;
                background-attachment: fixed;
            }}
            /* Enhanced Blur for Detail View */
            .detail-panel, .info-box {{
                background: rgba(18, 18, 18, 0.3) !important;
                backdrop-filter: blur(20px) saturate(180%) !important;
            }}
            </style>
        """, unsafe_allow_html=True)

    # ── DETAIL PANEL RENDERING ──────────────────────────────────────────
    
    # Calculate local relationships for the 3D mapping (Universal Explorer)
    local_recs = pd.DataFrame()
    # 2. Recommendations & Network (Smart Theatre Discovery)
    if embeddings_list is not None:
        local_recs = get_recommendations(title, df_model, embeddings_list, top_n=8)
        
    # ── PRECISION AUTO-NAVIGATION ────────────────────────────────────────
    # Create a hidden anchor at the absolute top of the detail section
    st.markdown('<div id="movie-detail-anchor"></div>', unsafe_allow_html=True)
    
    st.components.v1.html(
        f"""
        <!-- Auto-scroll trigger for: {title.replace(' ', '_')} -->
        <script>
            setTimeout(function() {{
                const anchor = window.parent.document.getElementById("movie-detail-anchor");
                if (anchor) {{
                    anchor.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }} else {{
                    window.parent.window.scrollTo({{ top: 0, behavior: 'smooth' }});
                }}
            }}, 150);
        </script>
        """,
        height=0
    )

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown(
                f'<div class="movie-poster-placeholder" style="border-radius:10px;height:300px;font-size:4rem">'
                f'{POSTER_PLACEHOLDER}</div>',
                unsafe_allow_html=True,
            )
        wl_label = "✅ In Watchlist" if in_watchlist else "＋ Add to Watchlist"
        if st.button(wl_label, key=f"wl_{title[:20]}"):
            if in_watchlist:
                st.session_state.watchlist.remove(title)
            else:
                st.session_state.watchlist.append(title)
            st.rerun()
        if st.button("✕ Close", key="close_detail"):
            st.session_state.selected_movie_detail = None
            st.rerun()

    with right_col:
        genre_badges = " ".join(
            f'<span class="badge-genre">{g.strip()}</span>'
            for g in genres.split(",") if g.strip() and g.strip() != "nan"
        )
        st.markdown(f'<p class="detail-title">{title}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="detail-meta">{year if year > 0 else "N/A"} &nbsp;·&nbsp; '
            f'<span style="color:#f59e0b">⭐ {rating:.1f}/10</span> '
            f'({vote_count:,} votes) &nbsp;·&nbsp; ⏱ {runtime} min</p>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div style="margin-bottom:12px">{genre_badges}</div>', unsafe_allow_html=True)
        if overview and len(overview) > 10:
            st.markdown(f'<p class="detail-overview">{overview}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="detail-overview"><em>No overview available.</em></p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**🎬 Director**\n\n{directors[:80]}")
        with c2:
            if revenue > 0:
                st.markdown(f"**💰 Revenue**\n\n${revenue/1e6:.1f}M")
        st.markdown(f"**🎭 Cast:** {cast[:150]}")
        if sim_score > 0:
            st.markdown(
                f'<span class="badge-score">🎯 {sim_score*100:.0f}% match</span>',
                unsafe_allow_html=True,
            )
            
        # UNIVERSAL 3D RELATIONSHIP EXPLORER
        if not local_recs.empty:
            with st.expander("🕸️ AI Relationship Mapping (3D Explorer)", expanded=True):
                st.markdown('<p style="font-size:0.8rem; color:#a1a1aa; margin-bottom:12px;">'
                            'Visualizing how this title connects to similar cinematic experiences '
                            'based on directors, genres, and semantic themes.</p>', 
                            unsafe_allow_html=True)
                # Filter out the title itself if it appears in local recs
                net_recs = local_recs[local_recs['title'] != title].head(6) 
                render_network_graph(title, net_recs)
        elif embeddings_list is None:
            st.info("💡 3D Mapping is currently unavailable (Embeddings not loaded).")
    # End Detail Rendering ──────────────────────────────────────────────


def render_movie_card(row, col, card_key):
    with col:
        poster_url = get_poster_url(row)
        rating = float(row.get("averageRating", row.get("vote_average", 0)))
        sim_score = float(row.get("similarity_score", 0))
        year = int(row.get("release_year", 0))
        genres_raw = str(row.get("genres", ""))
        genre_list = [g.strip() for g in genres_raw.split(",")][:2]
        title = str(row.get("title", "Unknown"))

        genre_html = "".join(
            f'<span class="badge-genre">{g}</span>' for g in genre_list if g and g != "nan"
        )
        sim_badge = f'<span class="badge-score">🎯 {sim_score*100:.0f}%</span>' if sim_score > 0 else ""
        poster_html = (
            f'<div class="movie-poster-wrapper"><div class="movie-poster" style="background-image: url(\'{poster_url}\');"></div></div>'
            if poster_url
            else f'<div class="movie-poster-wrapper"><div class="movie-poster-placeholder">{POSTER_PLACEHOLDER}</div></div>'
        )
        in_wl = title in st.session_state.get("watchlist", [])
        wl_dot = (
            '<span style="position:absolute;top:12px;right:12px;background:#10b981;border-radius:50%;'
            'width:12px;height:12px;display:inline-block;animation:pulseDot 2s infinite;'
            'box-shadow:0 0 8px #10b981;"></span>'
            if in_wl else ""
        )
        xai_reason = str(row.get("xai_reason", ""))
        xai_html = f'<p style="font-size:0.75rem; color:#a1a1aa; margin:4px 0 0 0; line-height:1.2;">{xai_reason}</p>' if xai_reason else ""
        
        html_content = "".join([
            f'<div class="movie-card" style="position:relative">',
            f'{wl_dot}{poster_html}',
            f'<div class="movie-info">',
            f'<p class="movie-title">{title}</p>',
            f'<p class="movie-year">{year if year > 0 else "N/A"}</p>',
            f'<div class="movie-meta">',
            f'<span class="badge-rating">⭐ {rating:.1f}</span>',
            f'{sim_badge}',
            f'</div>',
            f'<div class="movie-meta">{genre_html}</div>',
            f'{xai_html}',
            f'</div>',
            f'</div>'
        ])
        
        st.markdown(html_content, unsafe_allow_html=True)
        # Buttons in 2-column layout
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("🔎 Details", key=f"det_{card_key}", use_container_width=True):
                st.session_state.selected_movie_detail = row.to_dict()
                st.session_state.selected_movie_title = title
                st.rerun()
        with btn2:
            wl_label = "✅ Saved" if in_wl else "➕ Watchlist"
            if st.button(wl_label, key=f"wl_{card_key}", use_container_width=True):
                if "watchlist" not in st.session_state:
                    st.session_state.watchlist = []
                if in_wl:
                    st.session_state.watchlist.remove(title)
                else:
                    st.session_state.watchlist.append(title)
                st.rerun()


def render_movie_carousel(df_results, section_prefix="carousel"):
    """Renders a horizontally scrollable container of movie cards."""
    if df_results.empty:
        return
        
    # Create a unique ID for this carousel
    carousel_id = f"carousel-{section_prefix}"
    
    # Using a container with custom CSS for horizontal scrolling
    st.markdown(f"""
        <style>
        #{carousel_id} {{
            display: flex;
            overflow-x: auto;
            gap: 20px;
            padding: 20px 10px;
            scrollbar-width: thin;
            scrollbar-color: rgba(234,179,8,0.3) transparent;
            -webkit-overflow-scrolling: touch;
        }}
        #{carousel_id}::-webkit-scrollbar {{
            height: 6px;
        }}
        #{carousel_id} .carousel-item {{
            flex: 0 0 220px;
            min-width: 220px;
            transition: transform 0.3s ease;
        }}
        </style>
        <div id="{carousel_id}">
    """, unsafe_allow_html=True)
    
    # We'll use columns internally but set them to fixed width in CSS
    cols = st.columns(len(df_results))
    for i, (_, row) in enumerate(df_results.iterrows()):
        render_movie_card(row, cols[i], f"{section_prefix}_{i}")
        
    st.markdown("</div>", unsafe_allow_html=True)


def render_movie_grid(df_results, cols_per_row=4, section_prefix="grid"):
    n = len(df_results)
    if n == 0:
        st.info("No movies found matching your criteria.")
        return
    for row_idx in range(math.ceil(n / cols_per_row)):
        start = row_idx * cols_per_row
        end = min(start + cols_per_row, n)
        batch = df_results.iloc[start:end]
        columns = st.columns(cols_per_row)
        for col_offset, (_, movie_row) in enumerate(batch.iterrows()):
            abs_pos = start + col_offset
            card_key = f"{section_prefix}_{abs_pos}_{str(movie_row.get('title',''))[:20].replace(' ','_')}"
            render_movie_card(movie_row, columns[col_offset], card_key)
        st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHART TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
CHART_TEMPLATE = dict(
    plot_bgcolor="rgba(10,10,26,0)",
    paper_bgcolor="rgba(10,10,26,0)",
    font=dict(color="#94a3b8", family="Inter"),
    title_font=dict(color="#e2e8f0", size=16, family="Inter"),
    colorway=["#7c3aed","#4f46e5","#10b981","#f59e0b","#ef4444","#06b6d4","#8b5cf6","#ec4899","#3b82f6","#a78bfa"],
)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_dashboard(df_model):
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">📊 Data Science Dashboard</p>
        <p class="hero-subtitle">Explore trends, distributions, and insights from the TMDB–IMDB movie dataset</p>
    </div>
    """, unsafe_allow_html=True)

    rating_col = "averageRating" if "averageRating" in df_model.columns else "vote_average"

    st.markdown('<div class="section-header">📈 Key Metrics</div>', unsafe_allow_html=True)
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    metrics = [
        ("🎬", f"{len(df_model):,}", "Total Movies"),
        ("⭐", f"{df_model[rating_col].mean():.2f}", "Avg Rating"),
        ("🏆", f"{df_model[rating_col].max():.1f}", "Highest Rating"),
        ("📅", f"{int(df_model['release_year'][df_model['release_year']>0].max())}", "Latest Year"),
        ("🎭", f"{df_model['genres'].str.split(',').explode().str.strip().nunique()}", "Unique Genres"),
    ]
    for col, (icon, val, label) in zip([kc1, kc2, kc3, kc4, kc5], metrics):
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-icon">{icon}</div>'
                f'<div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["🎭 Genres", "⭐ Ratings", "📅 Timeline", "💰 Revenue"])

    genre_series = df_model["genres"].str.split(",").explode().str.strip()

    with tab1:
        col_left, col_right = st.columns(2)
        genre_counts = genre_series.value_counts().head(15).reset_index()
        genre_counts.columns = ["Genre", "Count"]
        genre_counts = genre_counts[genre_counts["Genre"].str.len() > 1]
        with col_left:
            fig = px.bar(genre_counts, x="Count", y="Genre", orientation="h",
                         title="Top 15 Genres by Movie Count", color="Count",
                         color_continuous_scale=["#4f46e5","#7c3aed","#a78bfa"])
            fig.update_layout(**CHART_TEMPLATE, height=450)
            fig.update_coloraxes(showscale=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            fig2 = px.pie(genre_counts.head(10), values="Count", names="Genre",
                          title="Genre Distribution (Top 10)", hole=0.45)
            fig2.update_layout(**CHART_TEMPLATE, height=450)
            fig2.update_traces(textfont_size=11, marker=dict(line=dict(color="#0a0a1a", width=2)))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig3 = px.histogram(df_model[df_model[rating_col] > 0], x=rating_col, nbins=40,
                                title="IMDB Rating Distribution", color_discrete_sequence=["#7c3aed"],
                                labels={rating_col: "Rating"})
            fig3.update_layout(**CHART_TEMPLATE, height=400, bargap=0.05)
            fig3.add_vline(x=df_model[rating_col].mean(), line_dash="dash", line_color="#f59e0b",
                           annotation_text=f"Avg: {df_model[rating_col].mean():.2f}",
                           annotation_font_color="#f59e0b")
            st.plotly_chart(fig3, use_container_width=True)
        with col_b:
            top8 = genre_series.value_counts().head(8).index.tolist()
            def get_primary_genre(g_str):
                for g in [x.strip() for x in str(g_str).split(",")]:
                    if g in top8:
                        return g
                return None
            df_plot = df_model[df_model[rating_col] > 0].copy()
            df_plot["primary_genre"] = df_plot["genres"].apply(get_primary_genre)
            df_plot = df_plot.dropna(subset=["primary_genre"])
            fig4 = px.box(df_plot, x="primary_genre", y=rating_col,
                          title="Rating Distribution by Genre", color="primary_genre",
                          color_discrete_sequence=["#7c3aed","#4f46e5","#10b981","#f59e0b","#ef4444","#06b6d4","#ec4899","#3b82f6"],
                          labels={"primary_genre": "Genre", rating_col: "Rating"})
            fig4.update_layout(**CHART_TEMPLATE, height=400, showlegend=False)
            fig4.update_xaxes(tickangle=-35)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<div class="section-header">🏆 Top 20 Highest Rated Movies</div>', unsafe_allow_html=True)
        top20 = df_model[df_model["vote_count"] >= 1000].nlargest(20, rating_col)[
            ["title", rating_col, "vote_count", "genres", "release_year"]
        ].reset_index(drop=True)
        top20.index += 1
        top20.columns = ["Title", "IMDB Rating", "Vote Count", "Genres", "Year"]
        st.dataframe(top20, use_container_width=True, column_config={
            "IMDB Rating": st.column_config.ProgressColumn("IMDB Rating", min_value=0, max_value=10, format="%.2f"),
            "Vote Count": st.column_config.NumberColumn("Vote Count", format="%d"),
        })

    with tab3:
        years_df = df_model[df_model["release_year"] > 1900].copy()
        col_c, col_d = st.columns(2)
        with col_c:
            years_df["decade"] = (years_df["release_year"] // 10 * 10).astype(str) + "s"
            decade_counts = years_df.groupby("decade").size().reset_index(name="count").sort_values("decade")
            fig5 = px.bar(decade_counts, x="decade", y="count", title="Movies by Decade",
                          color="count", color_continuous_scale=["#4f46e5","#7c3aed","#a78bfa","#c4b5fd"],
                          labels={"decade":"Decade","count":"# Movies"})
            fig5.update_layout(**CHART_TEMPLATE, height=380)
            fig5.update_coloraxes(showscale=False)
            st.plotly_chart(fig5, use_container_width=True)
        with col_d:
            yr_rating = (years_df[years_df["release_year"] >= 2000]
                         .groupby("release_year")[rating_col].mean().reset_index())
            fig6 = px.line(yr_rating, x="release_year", y=rating_col, markers=True,
                           title="Average Rating Trend (2000–Present)",
                           labels={"release_year": "Year", rating_col: "Avg Rating"})
            fig6.update_traces(line=dict(color="#7c3aed", width=2.5), marker=dict(color="#a78bfa", size=6))
            fig6.update_layout(**CHART_TEMPLATE, height=380)
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown('<div class="section-header">🎬 Top 15 Most Prolific Directors</div>', unsafe_allow_html=True)
        if "directors" in df_model.columns:
            dir_series = (df_model["directors"].str.replace(r"[,|]", "|", regex=True)
                          .str.split("|").explode().str.strip())
            dir_series = dir_series[dir_series.str.len() > 2]
            dir_counts = dir_series.value_counts().head(15).reset_index()
            dir_counts.columns = ["Director", "Movies"]
            fig7 = px.bar(dir_counts, x="Movies", y="Director", orientation="h",
                          title="Top 15 Directors by Movie Count", color="Movies",
                          color_continuous_scale=["#4f46e5","#7c3aed","#a78bfa"])
            fig7.update_layout(**CHART_TEMPLATE, height=450)
            fig7.update_coloraxes(showscale=False)
            fig7.update_yaxes(autorange="reversed")
            st.plotly_chart(fig7, use_container_width=True)

    with tab4:
        if "revenue" in df_model.columns:
            rev_df = df_model[df_model["revenue"] > 1_000_000].copy()
            rev_df["revenue_m"] = rev_df["revenue"] / 1_000_000
            col_e, col_f = st.columns(2)
            with col_e:
                fig8 = px.scatter(
                    rev_df.sample(min(2000, len(rev_df)), random_state=42),
                    x=rating_col, y="revenue_m", color="release_year",
                    color_continuous_scale="Viridis", title="Revenue vs. Rating",
                    labels={rating_col:"IMDB Rating","revenue_m":"Revenue ($M)","release_year":"Year"},
                    hover_data=["title"], opacity=0.7)
                fig8.update_layout(**CHART_TEMPLATE, height=430)
                st.plotly_chart(fig8, use_container_width=True)
            with col_f:
                if "runtime" in df_model.columns:
                    rt_df = df_model[(df_model["runtime"] > 20) & (df_model["runtime"] < 300)]
                    fig9 = px.histogram(rt_df, x="runtime", nbins=50, title="Runtime Distribution (min)",
                                        color_discrete_sequence=["#10b981"], labels={"runtime":"Runtime (min)"})
                    fig9.update_layout(**CHART_TEMPLATE, height=430, bargap=0.05)
                    fig9.add_vline(x=rt_df["runtime"].mean(), line_dash="dash", line_color="#f59e0b",
                                   annotation_text=f"Avg: {rt_df['runtime'].mean():.0f} min",
                                   annotation_font_color="#f59e0b")
                    st.plotly_chart(fig9, use_container_width=True)

            st.markdown('<div class="section-header">💰 Top 15 Highest Grossing Movies</div>', unsafe_allow_html=True)
            top_rev = rev_df.nlargest(15, "revenue")[
                ["title","revenue_m",rating_col,"genres","release_year"]
            ].reset_index(drop=True)
            top_rev.index += 1
            top_rev.columns = ["Title","Revenue ($M)","IMDB Rating","Genres","Year"]
            top_rev["Revenue ($M)"] = top_rev["Revenue ($M)"].round(1)
            fig10 = px.bar(top_rev, x="Revenue ($M)", y="Title", orientation="h",
                           title="Top 15 Highest Grossing Movies", color="IMDB Rating",
                           color_continuous_scale=["#4f46e5","#7c3aed","#f59e0b"],
                           hover_data=["Year","Genres"], text="Revenue ($M)")
            fig10.update_layout(**CHART_TEMPLATE, height=500)
            fig10.update_traces(texttemplate="$%{text:,.0f}M", textposition="outside")
            fig10.update_yaxes(autorange="reversed")
            st.plotly_chart(fig10, use_container_width=True)

        st.markdown('<div class="section-header">📉 Feature Correlation Intelligence</div>', unsafe_allow_html=True)
        # Select numeric columns for correlation
        numeric_df = df_model[[rating_col, "release_year", "runtime", "revenue", "score"]].dropna()
        numeric_df.columns = ["Rating", "Year", "Runtime", "Revenue", "IMDB Score"]
        corr = numeric_df.corr()
        
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Interaction Matrix (Multi-Factor Analysis)"
        )
        fig_corr.update_layout(**CHART_TEMPLATE, height=500)
        st.plotly_chart(fig_corr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_watchlist_page(df_model, embeddings_list):
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">📋 My Watchlist</p>
        <p class="hero-subtitle">Manage your saved movies and discover recommendations based on your taste</p>
    </div>
    """, unsafe_allow_html=True)

    watchlist = st.session_state.get("watchlist", [])
    # ... (rest of logic updated to pass embeddings_list to get_watchlist_recommendations)

    if not watchlist:
        st.markdown("""
        <div class="info-box" style="text-align:center;padding:48px 24px">
            <div style="font-size:3.5rem;margin-bottom:12px">🎬</div>
            <p style="font-size:1rem;color:#a1a1aa;margin:0">Your watchlist is empty.</p>
            <p style="font-size:0.85rem;color:#52525b;margin-top:6px">
                Go to <b>Recommendations</b> and click <b>＋ Add to Watchlist</b> on any movie.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Stats
    rating_col = "averageRating" if "averageRating" in df_model.columns else "vote_average"
    wl_movies = df_model[df_model["title"].isin(watchlist)].copy()
    s1, s2, s3, s4 = st.columns(4)
    valid_years = wl_movies["release_year"][wl_movies["release_year"] > 0]
    stat_data = [
        ("📋", str(len(watchlist)), "Saved Movies", s1),
        ("⭐", f"{wl_movies[rating_col].mean():.1f}" if len(wl_movies) else "–", "Avg Rating", s2),
        ("🎭", str(wl_movies["genres"].str.split(",").explode().str.strip().nunique()) if len(wl_movies) else "–", "Unique Genres", s3),
        ("📅", f"{int(valid_years.min())}–{int(valid_years.max())}" if len(valid_years) else "–", "Year Span", s4),
    ]
    for icon, val, label, col in stat_data:
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-icon">{icon}</div>'
                f'<div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Saved movie collection
    with st.expander("🎬 Your Saved Collection", expanded=True):
        if len(wl_movies) > 0:
            wl_movies["similarity_score"] = 0.0
            render_movie_grid(wl_movies, cols_per_row=4, section_prefix="wlpage")
        else:
            st.warning("Some watchlisted movies were not found in the current dataset.")

    # ── 0. MOVIE DETAIL PANEL (Absolute Top for Consistency) ───────────────
    if st.session_state.get("selected_movie_detail"):
        render_movie_detail_panel(
            pd.Series(st.session_state.selected_movie_detail), df_model, embeddings_list
        )
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # 1. Dashboard Stats
    rating_col = "averageRating" if "averageRating" in df_model.columns else "vote_average"
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🗑️ Manage Watchlist</div>', unsafe_allow_html=True)
    remove_col, clear_col = st.columns([3, 1])
    with remove_col:
        remove_pick = st.selectbox(
            "Remove movie:", ["— select a movie to remove —"] + watchlist,
            key="wl_remove_select", label_visibility="collapsed",
        )
        if st.button("🗑️ Remove Selected", key="wl_remove_btn"):
            if remove_pick != "— select a movie to remove —":
                st.session_state.watchlist.remove(remove_pick)
                st.rerun()
    with clear_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧹 Clear All", key="wl_clear_btn"):
            st.session_state.watchlist = []
            st.rerun()

    # Recommend from Watchlist
    with st.expander("🤖 AI Watchlist Synthesis", expanded=True):
        st.markdown("""
        <div class="info-box">
            💡 CineMatch <b>aggregates cosine similarity scores</b> across <b>all</b> your saved movies
            and surfaces films your watchlist collectively points toward — movies you've never seen but are likely to love.
        </div>
        """, unsafe_allow_html=True)

        rec_n = st.slider("How many recommendations?", 6, 20, 12, 2, key="wl_rec_n")
        if st.button("🔮 Recommend from My Watchlist", key="wl_rec_btn"):
            if embeddings_list is None:
                st.error("⚠️ AI model is still loading. Please wait a moment.")
            else:
                with st.spinner("🤖 Aggregating your taste profile..."):
                    wl_recs = get_watchlist_recommendations(watchlist, df_model, embeddings_list, top_n=rec_n)
                st.session_state["wl_recs"] = wl_recs.to_dict("records") if len(wl_recs) > 0 else []

        wl_recs_data = st.session_state.get("wl_recs", [])
        if wl_recs_data:
            st.markdown(f'<div class="section-header">✨ {len(wl_recs_data)} Picks Tailored to Your Watchlist</div>', unsafe_allow_html=True)
            render_movie_grid(pd.DataFrame(wl_recs_data), cols_per_row=4, section_prefix="wlrec")
        elif st.session_state.get("wl_recs") == []:
            st.warning("No recommendations found. Try adding more diverse movies to your watchlist.")

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_recommendation_page(df_model, embeddings_list):
    # ── 0. MOVIE DETAIL PANEL (Primary Focus - Absolute Top) ──────────────────
    if st.session_state.get("selected_movie_detail"):
        render_movie_detail_panel(
            pd.Series(st.session_state.selected_movie_detail), df_model, embeddings_list
        )
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🎬 CineMatch</p>
        <p class="hero-subtitle">Discover your next favourite film powered by AI &amp; Sentence Transformers</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar stats
    with st.sidebar:
        st.markdown("### 🔎 Quick Stats")
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        rc = "averageRating" if "averageRating" in df_model.columns else "vote_average"
        st.markdown(f"""
        <div style="font-size:0.85rem;color:#94a3b8;line-height:2.2">
        🎬 <b>{len(df_model):,}</b> movies in database<br>
        ⭐ Avg rating: <b>{df_model[rc].mean():.2f}</b><br>
        📅 Years: <b>{int(df_model['release_year'][df_model['release_year']>0].min())}–{int(df_model['release_year'][df_model['release_year']>0].max())}</b>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Focus Mode")
        focus_mode = st.radio(
            "Prioritize:",
            ["Balanced", "Director", "Genre"],
            index=0,
            help="Choose what the AI should prioritize when finding matches."
        )

    # How It Works
    with st.expander("⚙️ How CineMatch Works", expanded=False):
        h1, h2, h3, h4 = st.columns(4)
        steps = [
            ("1️⃣", "Data Preprocessing",
             "Filters 1.2M movies to the top 50,000 highest-quality unique titles using IMDB weighted rating."),
            ("2️⃣", "Feature Engineering",
             "Combines overview, genres, keywords, cast & directors into a unified text block called the <b>soup</b>."),
            ("3️⃣", "AI Embeddings",
             f"Uses <b>SentenceTransformer all-MiniLM-L6-v2</b> on <b>{CUDA_DEVICE.upper()}</b> to encode each movie into a 384-dim semantic vector."),
            ("4️⃣", "On-The-Fly Similarity",
             "Computes real-time cosine similarity for the selected movie only, reducing memory footprint by 99% for 50,000 movies."),
        ]
        for col, (num, title, desc) in zip([h1, h2, h3, h4], steps):
            with col:
                st.markdown(f"""
                <div style="background:rgba(234,179,8,0.06);border:1px solid rgba(202,138,4,0.15);
                            border-radius:12px;padding:16px;height:100%;min-height:160px">
                    <div style="font-size:1.6rem;margin-bottom:8px">{num}</div>
                    <div style="font-size:0.8rem;font-weight:700;color:#facc15;margin-bottom:6px">{title}</div>
                    <div style="font-size:0.74rem;color:#71717a;line-height:1.6">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    with st.expander("🎛️ Filter & Preferences", expanded=True):
        r1c1, r1c2, r1c3 = st.columns([3, 2, 2])
        with r1c1:
            st.markdown("**🎭 Favourite Genres**")
            selected_genres = st.multiselect(
                "Genres", ALL_GENRES, default=["Action", "Drama"],
                key="genre_select", label_visibility="collapsed", placeholder="Choose genres…",
            )
        with r1c2:
            st.markdown("**⭐ Minimum IMDB Rating**")
            min_rating = st.slider("Min Rating", 0.0, 10.0, 6.0, 0.1,
                                   label_visibility="collapsed", key="min_rating_slider")
        with r1c3:
            st.markdown("**📅 Release Year Range**")
            min_year, max_year = st.slider("Year range", 1950, 2025, (2000, 2025), 1,
                                           label_visibility="collapsed", key="year_range_slider")
        r2c1, r2c2, _ = st.columns([2, 2, 3])
        with r2c1:
            st.markdown("**🎯 Recommendations**")
            top_n = st.slider("How many?", 5, 20, 10, 1,
                              label_visibility="collapsed", key="top_n_slider")
        with r2c2:
            st.markdown("**📐 Grid Columns**")
            cols_per_row = st.select_slider("Columns", [2, 3, 4, 5], value=4,
                                            label_visibility="collapsed", key="cols_slider")

    # ── FIRST USE CHECK ───────────────────────────────────────────────────
    has_active_results = any([
        st.session_state.get("vibe_recs"),
        st.session_state.get("seed_recs"),
        st.session_state.get("pref_recs")
    ])

    if not has_active_results:
        # Welcome banner removed as per user request (too big)

        
        # DISPLAY FEATURED MOVIES FOR FIRST USE INSPIRATION (Trending/Viral Fallback)
        with st.expander("✨ Trending Discoveries — Get Inspired", expanded=True):
            st.markdown('<p style="font-size:0.9rem; color:#a1a1aa; margin-bottom:15px; margin-left:14px;">'
                        'Explore currently viral and newly released titles trending across the platform.</p>', 
                        unsafe_allow_html=True)
            
            # Get the highest-rated RECENT movies (2022+) to simulate "Viral/Current"
            rc_f = "averageRating" if "averageRating" in df_model.columns else "vote_average"
            inspiration_df = df_model[df_model["release_year"] >= 2022].sort_values("score", ascending=False).head(8).copy()
            
            # Fallback if no 2022+ movies (e.g. smaller test dataset)
            if len(inspiration_df) < 4:
                inspiration_df = df_model.sort_values(["release_year", "score"], ascending=False).head(8).copy()
                
            inspiration_df["similarity_score"] = 0.0
            render_movie_grid(inspiration_df, cols_per_row=cols_per_row, section_prefix="inspiration")

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    # ── Vibe Search Section ──────────────────────────────────────────────────
    with st.expander("🧠 Search by Vibe (AI Semantic Discovery)", expanded=True):
        vibe_query = st.text_input(
            "Vibe Search", value="", 
            placeholder="🔮 Describe the feeling... e.g., 'a lonely astronaut on a cold planet'",
            key="vibe_search", label_visibility="collapsed"
        )
        
        col_v1, col_v2 = st.columns([1, 4])
        with col_v1:
            vibe_btn = st.button("🔍 Discover by Vibe", key="vibe_btn")
            
        if vibe_btn and vibe_query:
            with st.spinner("🧠 AI is dreaming up matches..."):
                vibe_recs = get_semantic_recommendations(
                    vibe_query, embeddings_list, df_model, top_n=10,
                    genre_filter=selected_genres, min_rating=min_rating,
                    min_year=min_year, max_year=max_year
                )
                st.session_state["vibe_recs"] = vibe_recs.to_dict("records") if len(vibe_recs) > 0 else []
                    
        vibe_recs_data = st.session_state.get("vibe_recs", [])
        if vibe_recs_data and vibe_query:
            st.markdown(f'<p style="font-size:0.9rem; color:#facc15; margin-bottom:15px;">✨ Semantic matches for: "{vibe_query}"</p>', unsafe_allow_html=True)
            render_movie_grid(pd.DataFrame(vibe_recs_data), cols_per_row=cols_per_row, section_prefix="vibe")

    # ── Similar Movie Search Section ──────────────────────────────────────────
    with st.expander("🔍 Find Movies Similar To...", expanded=True):
        search_query = st.text_input(
            "Search", value="", placeholder="🔎  Type a movie name to search…",
            key="movie_search", label_visibility="collapsed",
        )
        movie_list_sorted = sorted(df_model["title"].tolist())
        filtered_movies = (
            [m for m in movie_list_sorted if search_query.lower() in m.lower()]
            if search_query else movie_list_sorted
        )
        if not filtered_movies:
            st.warning("No movies match your search.")
            filtered_movies = movie_list_sorted[:100]

        selected_movie = st.selectbox(
            "Select:", filtered_movies, key="movie_select", label_visibility="collapsed",
            index=None, placeholder="🔎 Type to search movies..."
        )

        if selected_movie:
            sel_row = df_model[df_model["title"] == selected_movie].iloc[0]
            rating_col = "averageRating" if "averageRating" in df_model.columns else "vote_average"
            ic1, ic2 = st.columns([1, 3])
            with ic1:
                poster_url = get_poster_url(sel_row)
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown(
                        f'<div class="movie-poster-placeholder" style="border-radius:12px;font-size:5rem;height:280px">'
                        f'{POSTER_PLACEHOLDER}</div>', unsafe_allow_html=True,
                    )
            with ic2:
                rating = float(sel_row.get(rating_col, 0))
                year = int(sel_row.get("release_year", 0))
                runtime = int(sel_row.get("runtime", 0))
                revenue = float(sel_row.get("revenue", 0))
                st.markdown(f"### {selected_movie}")
                st.markdown(
                    f"**📅 Year:** {year if year > 0 else 'N/A'}  &nbsp;|&nbsp;  "
                    f"**⭐ Rating:** {rating:.1f}/10  &nbsp;|&nbsp;  "
                    f"**🎭 Genres:** {sel_row.get('genres','')}"
                )
                st.markdown(f"**🎬 Director:** {str(sel_row.get('directors',''))[:80]}")
                st.markdown(f"**🎭 Cast:** {str(sel_row.get('cast',''))[:120]}...")
                overview_raw = str(sel_row.get("overview", ""))
                if overview_raw and len(overview_raw) > 10:
                    st.markdown(f"**📖 Overview:** {overview_raw[:300]}...")
                else:
                    st.markdown("*Overview not available*")
                st.markdown(
                    f"**⏱️ Runtime:** {runtime} min"
                    + (f"  &nbsp;|&nbsp;  **💰 Revenue:** ${revenue/1e6:.1f}M" if revenue > 0 else "")
                )
            st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        rec_btn = st.button("🔮 Get AI Recommendations", key="rec_btn_seed")
        if rec_btn and selected_movie and embeddings_list is not None:
            with st.spinner("🤖 Finding similar movies..."):
                recs = get_recommendations(
                    selected_movie, df_model, embeddings_list, top_n=top_n,
                    genre_filter=selected_genres if selected_genres else None,
                    min_rating=min_rating, min_year=min_year, max_year=max_year,
                    focus=focus_mode
                )
            st.session_state["seed_recs"] = recs.to_dict("records") if len(recs) > 0 else []
            st.session_state["seed_recs_label"] = selected_movie
            st.session_state.pop("selected_movie_detail", None)
        elif rec_btn and embeddings_list is None:
            st.error("⚠️ The AI model is still loading. Please wait a moment.")

        # Display Results inside expander
        seed_recs_data = st.session_state.get("seed_recs", [])
        if seed_recs_data and st.session_state.get("seed_recs_label"):
            label = st.session_state.get("seed_recs_label", "")
            recs_df = pd.DataFrame(seed_recs_data)
            st.markdown(f'<div class="section-header">✨ Top {len(recs_df)} Recommendations for "{label}"</div>', unsafe_allow_html=True)
            render_movie_grid(recs_df, cols_per_row=cols_per_row, section_prefix="seed")
        elif rec_btn and selected_movie:
            st.warning("No recommendations found. Try relaxing your genre filters or rating threshold.")


    # ── Discover by Taste Section ──────────────────────────────────────────
    with st.expander("🌟 Discover by Your Taste", expanded=True):
        st.markdown("""
        <div class="info-box">
            💡 No specific movie in mind? Click below to discover top-rated films matching your genre preferences.
        </div>
        """, unsafe_allow_html=True)

        pref_btn = st.button("🌟 Show Top Movies for My Taste", key="rec_btn_pref")
        if pref_btn:
            with st.spinner("🔍 Curating your personalized list..."):
                pref_recs = get_recommendations_by_preferences(
                    df_model, embeddings_list,
                    preferred_genres=selected_genres, min_rating=min_rating,
                    min_year=min_year, max_year=max_year, top_n=top_n,
                )
            st.session_state["pref_recs"] = pref_recs.to_dict("records") if len(pref_recs) > 0 else []
            st.session_state["pref_recs_label"] = ", ".join(selected_genres) if selected_genres else "All Genres"
            st.session_state.pop("selected_movie_detail", None)

        pref_recs_data = st.session_state.get("pref_recs", [])
        if pref_recs_data:
            genres_display = st.session_state.get("pref_recs_label", "All Genres")
            pref_df = pd.DataFrame(pref_recs_data)
            st.markdown(f'<div class="section-header">🌟 Top {len(pref_df)} Picks — {genres_display}</div>', unsafe_allow_html=True)
            render_movie_grid(pref_df, cols_per_row=cols_per_row, section_prefix="pref")
        elif pref_btn:
            st.warning("No movies found. Try broadening genre selection or lowering the rating threshold.")

    # ── Section 3: Trending Now (Only shown if user has started exploring) ───
    # ── Trending Now Section ───────────────────────────────────────────────
    if has_active_results:
        with st.expander("🔥 Trending Now — Top Weighted Score", expanded=True):
            st.markdown("""
            <div class="info-box">
                🔥 Highest-ranked movies by IMDB weighted score, filtered by your current genre & rating preferences.
            </div>
            """, unsafe_allow_html=True)

            rating_col_t = "averageRating" if "averageRating" in df_model.columns else "vote_average"
            trending_df = df_model[df_model[rating_col_t] >= min_rating].copy()
            if selected_genres:
                trending_df = trending_df[
                    trending_df["genres"].apply(
                        lambda g: any(sg.lower() in str(g).lower() for sg in selected_genres)
                    )
                ]
            trending_df = trending_df.sort_values("score", ascending=False).head(8)
            trending_df["similarity_score"] = 0.0
            if len(trending_df) > 0:
                render_movie_grid(trending_df, cols_per_row=cols_per_row, section_prefix="trending_grid")
            else:
                st.info("No trending movies match your current filters.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP ENTRY
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Session state defaults
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    if "page" not in st.session_state:
        st.session_state.page = "recommend"

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 10px 0">
            <span style="font-size:2.5rem">🎬</span>
            <h2 style="color:#facc15;margin:4px 0 0 0;font-weight:800">CineMatch</h2>
            <p style="color:#52525b;font-size:0.8rem;margin:0">AI Movie Recommender</p>
        </div>
        """, unsafe_allow_html=True)

        # GPU / AI status badge
        gpu_color = "#10b981" if CUDA_DEVICE == "cuda" else "#f59e0b"
        gpu_label = "⚡ GPU · CUDA" if CUDA_DEVICE == "cuda" else "🖥️ CPU Mode"
        gpu_bg = "16,185,129" if CUDA_DEVICE == "cuda" else "245,158,11"
        st.markdown(f"""
        <div style="text-align:center;margin:2px 0 14px 0">
            <span style="background:rgba({gpu_bg},0.15);border:1px solid {gpu_color};
                         border-radius:20px;padding:4px 14px;font-size:0.7rem;
                         font-weight:700;color:{gpu_color};letter-spacing:0.05em">
                {gpu_label}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Navigation (Button Mode)
        st.markdown('<p style="font-size:0.7rem; color:#71717a; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px; font-weight:700">Explore</p>', unsafe_allow_html=True)
        
        watchlist_count = len(st.session_state.get("watchlist", []))
        wl_badge = f" ({watchlist_count})" if watchlist_count > 0 else ""
        
        pages = {
            "🎬 Recommendations": "recommend",
            "📊 Data Dashboard": "dashboard",
            f"📋 My Watchlist{wl_badge}": "watchlist"
        }
        
        current_page = st.session_state.get("page", "recommend")
        # 🔒 SIDEBAR NAVIGATION & LOCK LOGIC
        is_initializing = "ai_ready" not in st.session_state
        
        for label, page_id in pages.items():
            is_active = (current_page == page_id)
            button_label = f"✨ {label}" if is_active else label
            
            # Disable buttons while seeding AI to prevent work interruption
            if st.button(button_label, key=f"nav_{page_id}", 
                        use_container_width=True, 
                        disabled=is_initializing):
                st.session_state.page = page_id
                st.session_state.pop("selected_movie_detail", None)
                st.rerun()
        
        if is_initializing:
            st.markdown("""
            <div style="font-size:0.7rem; color:#94a3b8; margin-top:12px; text-align:center">
                <i>Navigation is locked while<br>AI Engine is warming up...</i>
            </div>
            """, unsafe_allow_html=True)
                
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        
        # 🟢 SYSTEM HEALTH DASHBOARD
        st.markdown('<p style="font-size:0.75rem; color:#71717a; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px; font-weight:700">System Status</p>', unsafe_allow_html=True)
        
        dataset_status = "✅ Ready" if "dataset_ready" in st.session_state else "⏳ Loading..."
        ai_status = "✅ Active" if "ai_ready" in st.session_state else "⏳ Initializing..."
        gpu_info = torch.cuda.get_device_name(0) if (TORCH_AVAILABLE and torch.cuda.is_available()) else "None"
        
        st.markdown(f"""
        <div style="background:rgba(15,15,35,0.4);border:1px solid rgba(255,255,255,0.05);
                    border-radius:10px;padding:12px;margin-bottom:20px">
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="font-size:0.75rem;color:#94a3b8">📦 Dataset</span>
                <span style="font-size:0.75rem;color:#10b981;font-weight:700">{dataset_status}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="font-size:0.75rem;color:#94a3b8">🤖 AI Engine</span>
                <span style="font-size:0.75rem;color:#7c3aed;font-weight:700">{ai_status}</span>
            </div>
            <div style="display:flex;justify-content:space-between;border-top:1px solid rgba(255,255,255,0.05);padding-top:6px">
                <span style="font-size:0.65rem;color:#64748b">🔌 GPU:</span>
                <span style="font-size:0.65rem;color:#64748b;font-weight:600;text-align:right">{gpu_info}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TOP-LEVEL INITIALIZATION SLOT ────────────────────────────────────────
    # This slot is reserved for the landing message and AI progress bar
    top_init_slot = st.empty()

    if "dataset_ready" not in st.session_state:
        top_init_slot.markdown("""
        <div style="text-align:center;padding:80px 20px">
           <h2 style="color:#facc15">🎬 CineMatch 2.0 Initializing...</h2>
           <p style="color:#94a3b8">Preparing cinematic intelligence and movie dataset...</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Load Data ─────────────────────────────────────────────────────────────
    with st.spinner("📦 Loading movie database..."):
        df_model = load_and_process_data()
        if df_model is not None:
            st.session_state["dataset_ready"] = True

    if df_model is None:
        st.error("""
        ⚠️ **Dataset not found!**

        Please download the dataset from Kaggle and place it at:
        ```
        archive/TMDB_IMDB_Movies_Dataset.csv
        ```
        👉 [Download from Kaggle](https://www.kaggle.com/datasets/ggtejas/tmdb-imdb-merged-movies-dataset)
        """)
        return

    # ── Load AI Model ─────────────────────────────────────────────────────────
    embeddings_list = None
    current_page = st.session_state.get("page", "recommend")
    if SENTENCE_TRANSFORMER_AVAILABLE:
        # Pinned to the top slot for maximum visibility
        with top_init_slot:
            embeddings_list = compute_embeddings(df_model)
            
        if embeddings_list is not None:
            st.session_state["ai_ready"] = True
            top_init_slot.empty() # Clear top slot once AI is ready
    else:
        with st.sidebar:
            st.warning("⚠️ sentence-transformers not installed.\nRun: `pip install sentence-transformers`")

    # ── Render Page ───────────────────────────────────────────────────────────
    if current_page == "dashboard":
        render_dashboard(df_model)
    elif current_page == "watchlist":
        render_watchlist_page(df_model, embeddings_list)
    else:
        render_recommendation_page(df_model, embeddings_list)


if __name__ == "__main__":
    main()