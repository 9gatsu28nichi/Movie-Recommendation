import streamlit as st
import numpy as np
import pandas as pd
import nltk
import re
import requests
import warnings
import math
from io import BytesIO

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

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
    page_title="CineMatch — Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — DARK CINEMATIC THEME
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
    "Thriller", "War", "Western"
]

# ─────────────────────────────────────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

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
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process_data():
    try:
        df = pd.read_csv("archive/TMDB_IMDB_Movies_Dataset.csv", low_memory=False)
    except FileNotFoundError:
        return None

    # IMDB Weighted Rating formula
    m = df["vote_count"].quantile(0.9)
    C = df["vote_average"].mean()

    def weighted_rating(x):
        v = x["vote_count"]
        R = x["vote_average"]
        return (v / (v + m)) * R + (m / (v + m)) * C

    q_movies = df[df["vote_count"] >= m].copy()
    q_movies["score"] = q_movies.apply(weighted_rating, axis=1)
    df_model = q_movies.sort_values("score", ascending=False).head(20000).copy()
    df_model = df_model.drop_duplicates(subset="title").reset_index(drop=True)

    # Parse release year
    df_model["release_year"] = pd.to_datetime(
        df_model["release_date"], errors="coerce"
    ).dt.year.fillna(0).astype(int)

    # Clean text features
    df_model["overview"] = df_model["overview"].fillna("").apply(clean_text)
    for feat in ["genres", "keywords"]:
        df_model[feat] = df_model[feat].fillna("").str.lower()
    for feat in ["cast", "directors"]:
        df_model[feat] = df_model[feat].fillna("").str.lower()
        df_model[feat] = df_model[feat].str.replace(" ", "", regex=False)

    df_model["soup"] = (
        df_model["overview"] + " "
        + df_model["genres"] + " "
        + df_model["keywords"] + " "
        + df_model["cast"] + " "
        + df_model["directors"]
    )

    # Keep poster_path for TMDB
    if "poster_path" not in df_model.columns:
        df_model["poster_path"] = ""
    else:
        df_model["poster_path"] = df_model["poster_path"].fillna("")

    return df_model

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS & SIMILARITIES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def compute_embeddings(_df_model):
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        return None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        _df_model["soup"].tolist(),
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=256,
    )
    similarities = model.similarity(embeddings, embeddings)
    return similarities

# ─────────────────────────────────────────────────────────────────────────────
# TMDB POSTER FETCH
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster_url(movie_id):
    """Fetch poster URL from TMDB by movie ID."""
    if pd.isna(movie_id) or movie_id == 0:
        return None
    try:
        url = f"{TMDB_API_BASE}/movie/{int(movie_id)}?api_key={TMDB_API_KEY}&language=en-US"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            path = data.get("poster_path")
            if path:
                return f"{TMDB_IMAGE_BASE}{path}"
    except Exception:
        pass
    return None


def get_poster_url(row):
    """Get poster URL from CSV poster_path or fallback to TMDB API."""
    poster_path = row.get("poster_path", "")
    if isinstance(poster_path, str) and poster_path.startswith("/"):
        return f"{TMDB_IMAGE_BASE}{poster_path}"
    # Fallback: use TMDB ID
    tmdb_id = row.get("id", 0)
    return fetch_poster_url(tmdb_id)

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def get_recommendations(
    title, similarities, df_model, top_n=10,
    genre_filter=None, min_rating=0.0, min_year=1900, max_year=2025
):
    indices = pd.Series(df_model.index, index=df_model["title"]).drop_duplicates()
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]
    sim_scores = list(enumerate(similarities[idx].numpy() if hasattr(similarities[idx], 'numpy') else similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx]  # exclude self

    results = []
    for i, score in sim_scores:
        row = df_model.iloc[i]
        # Rating filter
        rating_val = float(row.get("averageRating", row.get("vote_average", 0)))
        if np.isnan(rating_val) or rating_val < min_rating:
            continue
        # Year filter
        year = int(row.get("release_year", 0))
        if year > 0 and (year < min_year or year > max_year):
            continue
        # Genre filter
        if genre_filter:
            row_genres = str(row.get("genres", "")).lower()
            if not any(g.lower() in row_genres for g in genre_filter):
                continue
        results.append({
            "idx": i,
            "similarity": float(score),
        })
        if len(results) >= top_n:
            break

    if not results:
        return pd.DataFrame()

    movie_indices = [r["idx"] for r in results]
    scores = [r["similarity"] for r in results]
    result_df = df_model.iloc[movie_indices].copy()
    result_df["similarity_score"] = scores
    return result_df.reset_index(drop=True)


def get_recommendations_by_preferences(
    df_model, similarities, preferred_genres, min_rating, min_year, max_year, top_n=10
):
    """Recommend top movies by genre preferences without a seed movie."""
    mask = pd.Series([True] * len(df_model))

    if preferred_genres:
        def has_genre(g_str):
            g_lower = str(g_str).lower()
            return any(g.lower() in g_lower for g in preferred_genres)
        mask = mask & df_model["genres"].apply(has_genre)

    rating_col = "averageRating" if "averageRating" in df_model.columns else "vote_average"
    mask = mask & (df_model[rating_col] >= min_rating)
    if min_year > 1900:
        mask = mask & (df_model["release_year"] >= min_year)
    if max_year < 2025:
        mask = mask & (df_model["release_year"] <= max_year)

    filtered = df_model[mask].copy()
    filtered["similarity_score"] = filtered["score"]
    return filtered.head(top_n).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def render_movie_detail_panel(row, df_model, similarities):
    """Render a full-detail panel for a selected movie."""
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

    # Watchlist state
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    in_watchlist = title in st.session_state.watchlist

    st.markdown('<div class="detail-panel">', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown(f'<div class="movie-poster-placeholder" style="border-radius:10px;height:300px;font-size:4rem">{POSTER_PLACEHOLDER}</div>', unsafe_allow_html=True)

        # Watchlist toggle
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
        # Title & meta
        genre_badges = " ".join(
            f'<span class="badge-genre">{g.strip()}</span>'
            for g in genres.split(",") if g.strip() and g.strip() != "nan"
        )
        st.markdown(f'<p class="detail-title">{title}</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="detail-meta">{year if year > 0 else "N/A"} &nbsp;·&nbsp; '
            f'<span style="color:#f59e0b">⭐ {rating:.1f}/10</span> '
            f'({vote_count:,} votes) &nbsp;·&nbsp; '
            f'⏱ {runtime} min</p>',
            unsafe_allow_html=True
        )
        st.markdown(f'<div style="margin-bottom:12px">{genre_badges}</div>', unsafe_allow_html=True)

        # Overview
        if overview and len(overview) > 10:
            st.markdown(f'<p class="detail-overview">{overview}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="detail-overview"><em>No overview available.</em></p>', unsafe_allow_html=True)

        # Crew & stats
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**🎬 Director**\n\n{directors[:80]}")
        with c2:
            if revenue > 0:
                st.markdown(f"**💰 Revenue**\n\n${revenue/1e6:.1f}M")
        st.markdown(f"**🎭 Cast:** {cast[:150]}")

        if sim_score > 0:
            st.markdown(f'<span class="badge-score">🎯 {sim_score*100:.0f}% match</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_movie_card(row, col, card_key):
    """Render a single clickable movie card."""
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
            f'<img class="movie-poster" src="{poster_url}" alt="{title}" loading="lazy"/>'
            if poster_url
            else f'<div class="movie-poster-placeholder">{POSTER_PLACEHOLDER}</div>'
        )
        in_wl = title in st.session_state.get("watchlist", [])
        wl_dot = '<span style="position:absolute;top:12px;right:12px;background:#10b981;border-radius:50%;width:12px;height:12px;display:inline-block;animation:pulseDot 2s infinite;box-shadow:0 0 8px #10b981;"></span>' if in_wl else ""

        st.markdown(f"""
        <div class="movie-card" style="position:relative">
            {wl_dot}{poster_html}
            <div class="movie-info">
                <p class="movie-title">{title}</p>
                <p class="movie-year">{year if year > 0 else 'N/A'}</p>
                <div class="movie-meta">
                    <span class="badge-rating">⭐ {rating:.1f}</span>
                    {sim_badge}
                </div>
                <div class="movie-meta">{genre_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Use stable key: section prefix + position index avoids hash collisions across reruns
        if st.button("View Details", key=card_key, help=f"Details for {title}",
                     use_container_width=True):
            # Store both the movie data AND which section triggered the click
            st.session_state.selected_movie_detail = row.to_dict()
            st.session_state.selected_movie_title = title
            st.rerun()


def render_movie_grid(df_results, cols_per_row=4, section_prefix="grid"):
    """Render a responsive grid of clickable movie cards."""
    n = len(df_results)
    if n == 0:
        st.info("No movies found matching your criteria.")
        return

    rows = math.ceil(n / cols_per_row)
    for row_idx in range(rows):
        start = row_idx * cols_per_row
        end = min(start + cols_per_row, n)
        batch = df_results.iloc[start:end]
        columns = st.columns(cols_per_row)
        for col_offset, (_, movie_row) in enumerate(batch.iterrows()):
            # Key = section prefix + absolute position → unique across all grids on page
            abs_pos = start + col_offset
            card_key = f"{section_prefix}_{abs_pos}_{str(movie_row.get('title', ''))[:20].replace(' ', '_')}"
            render_movie_card(movie_row, columns[col_offset], card_key)
        st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD CHARTS
# ─────────────────────────────────────────────────────────────────────────────
CHART_TEMPLATE = dict(
    plot_bgcolor="rgba(10,10,26,0)",
    paper_bgcolor="rgba(10,10,26,0)",
    font=dict(color="#94a3b8", family="Inter"),
    title_font=dict(color="#e2e8f0", size=16, family="Inter"),
    colorway=["#7c3aed", "#4f46e5", "#10b981", "#f59e0b", "#ef4444",
              "#06b6d4", "#8b5cf6", "#ec4899", "#3b82f6", "#a78bfa"],
)


def render_dashboard(df_model):
    st.markdown('<div class="hero-banner">', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">📊 Data Science Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Explore trends, distributions, and insights from the TMDB–IMDB movie dataset</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    rating_col = "averageRating" if "averageRating" in df_model.columns else "vote_average"

    # ── KPI Cards ──────────────────────────────────────────────────────────
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
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tab Layout ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🎭 Genres", "⭐ Ratings", "📅 Timeline", "💰 Revenue"])

    # ── TAB 1: Genres ──────────────────────────────────────────────────────
    with tab1:
        col_left, col_right = st.columns(2)

        # Genre frequency
        genre_series = df_model["genres"].str.split(",").explode().str.strip()
        genre_counts = genre_series.value_counts().head(15).reset_index()
        genre_counts.columns = ["Genre", "Count"]
        genre_counts = genre_counts[genre_counts["Genre"].str.len() > 1]

        with col_left:
            fig = px.bar(
                genre_counts,
                x="Count", y="Genre",
                orientation="h",
                title="Top 15 Genres by Movie Count",
                color="Count",
                color_continuous_scale=["#4f46e5", "#7c3aed", "#a78bfa"],
            )
            fig.update_layout(**CHART_TEMPLATE, height=450)
            fig.update_coloraxes(showscale=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            fig2 = px.pie(
                genre_counts.head(10),
                values="Count", names="Genre",
                title="Genre Distribution (Top 10)",
                hole=0.45,
            )
            fig2.update_layout(**CHART_TEMPLATE, height=450)
            fig2.update_traces(
                textfont_size=11,
                marker=dict(line=dict(color="#0a0a1a", width=2))
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 2: Ratings ──────────────────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            fig3 = px.histogram(
                df_model[df_model[rating_col] > 0],
                x=rating_col,
                nbins=40,
                title="IMDB Rating Distribution",
                color_discrete_sequence=["#7c3aed"],
                labels={rating_col: "Rating"},
            )
            fig3.update_layout(**CHART_TEMPLATE, height=400, bargap=0.05)
            fig3.add_vline(
                x=df_model[rating_col].mean(),
                line_dash="dash", line_color="#f59e0b",
                annotation_text=f"Avg: {df_model[rating_col].mean():.2f}",
                annotation_font_color="#f59e0b",
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            # Rating by genre (box plot top 8 genres)
            top8 = genre_series.value_counts().head(8).index.tolist()

            def get_primary_genre(g_str):
                genres = [x.strip() for x in str(g_str).split(",")]
                for g in genres:
                    if g in top8:
                        return g
                return None

            df_plot = df_model[df_model[rating_col] > 0].copy()
            df_plot["primary_genre"] = df_plot["genres"].apply(get_primary_genre)
            df_plot = df_plot.dropna(subset=["primary_genre"])

            fig4 = px.box(
                df_plot,
                x="primary_genre", y=rating_col,
                title="Rating Distribution by Genre",
                color="primary_genre",
                color_discrete_sequence=["#7c3aed", "#4f46e5", "#10b981", "#f59e0b",
                                          "#ef4444", "#06b6d4", "#ec4899", "#3b82f6"],
                labels={"primary_genre": "Genre", rating_col: "Rating"},
            )
            fig4.update_layout(**CHART_TEMPLATE, height=400, showlegend=False)
            fig4.update_xaxes(tickangle=-35)
            st.plotly_chart(fig4, use_container_width=True)

        # Top 20 Highest Rated Movies
        st.markdown('<div class="section-header">🏆 Top 20 Highest Rated Movies</div>', unsafe_allow_html=True)
        top20 = df_model[df_model["vote_count"] >= 1000].nlargest(20, rating_col)[
            ["title", rating_col, "vote_count", "genres", "release_year"]
        ].reset_index(drop=True)
        top20.index += 1
        top20.columns = ["Title", "IMDB Rating", "Vote Count", "Genres", "Year"]
        st.dataframe(
            top20,
            use_container_width=True,
            column_config={
                "IMDB Rating": st.column_config.ProgressColumn(
                    "IMDB Rating", min_value=0, max_value=10, format="%.2f"
                ),
                "Vote Count": st.column_config.NumberColumn("Vote Count", format="%d"),
            },
        )

    # ── TAB 3: Timeline ──────────────────────────────────────────────────────
    with tab3:
        years_df = df_model[df_model["release_year"] > 1900].copy()

        col_c, col_d = st.columns(2)

        with col_c:
            # Movies per decade
            years_df["decade"] = (years_df["release_year"] // 10 * 10).astype(str) + "s"
            decade_counts = years_df.groupby("decade").size().reset_index(name="count")
            decade_counts = decade_counts.sort_values("decade")

            fig5 = px.bar(
                decade_counts,
                x="decade", y="count",
                title="Movies by Decade",
                color="count",
                color_continuous_scale=["#4f46e5", "#7c3aed", "#a78bfa", "#c4b5fd"],
                labels={"decade": "Decade", "count": "# Movies"},
            )
            fig5.update_layout(**CHART_TEMPLATE, height=380)
            fig5.update_coloraxes(showscale=False)
            st.plotly_chart(fig5, use_container_width=True)

        with col_d:
            # Average rating by year (line chart - 2000 onwards)
            yr_rating = (
                years_df[years_df["release_year"] >= 2000]
                .groupby("release_year")[rating_col]
                .mean()
                .reset_index()
            )
            fig6 = px.line(
                yr_rating,
                x="release_year", y=rating_col,
                title="Average Rating Trend (2000–Present)",
                markers=True,
                labels={"release_year": "Year", rating_col: "Avg Rating"},
            )
            fig6.update_traces(
                line=dict(color="#7c3aed", width=2.5),
                marker=dict(color="#a78bfa", size=6),
            )
            fig6.update_layout(**CHART_TEMPLATE, height=380)
            st.plotly_chart(fig6, use_container_width=True)

        # Top 15 Directors
        st.markdown('<div class="section-header">🎬 Top 15 Most Prolific Directors</div>', unsafe_allow_html=True)
        if "directors" in df_model.columns:
            director_raw = df_model["directors"].str.replace(r"[,|]", "|", regex=True)
            dir_series = director_raw.str.split("|").explode().str.strip()
            dir_series = dir_series[dir_series.str.len() > 2]
            dir_counts = dir_series.value_counts().head(15).reset_index()
            dir_counts.columns = ["Director", "Movies"]

            fig7 = px.bar(
                dir_counts,
                x="Movies", y="Director",
                orientation="h",
                title="Top 15 Directors by Movie Count",
                color="Movies",
                color_continuous_scale=["#4f46e5", "#7c3aed", "#a78bfa"],
            )
            fig7.update_layout(**CHART_TEMPLATE, height=450)
            fig7.update_coloraxes(showscale=False)
            fig7.update_yaxes(autorange="reversed")
            st.plotly_chart(fig7, use_container_width=True)

    # ── TAB 4: Revenue ──────────────────────────────────────────────────────
    with tab4:
        if "revenue" in df_model.columns:
            rev_df = df_model[df_model["revenue"] > 1_000_000].copy()
            rev_df["revenue_m"] = rev_df["revenue"] / 1_000_000

            col_e, col_f = st.columns(2)

            with col_e:
                fig8 = px.scatter(
                    rev_df.sample(min(2000, len(rev_df)), random_state=42),
                    x=rating_col, y="revenue_m",
                    color="release_year",
                    color_continuous_scale="Viridis",
                    title="Revenue vs. Rating",
                    labels={rating_col: "IMDB Rating", "revenue_m": "Revenue ($M)", "release_year": "Year"},
                    hover_data=["title"],
                    opacity=0.7,
                    size_max=8,
                )
                fig8.update_layout(**CHART_TEMPLATE, height=430)
                st.plotly_chart(fig8, use_container_width=True)

            with col_f:
                # Runtime distribution
                if "runtime" in df_model.columns:
                    rt_df = df_model[(df_model["runtime"] > 20) & (df_model["runtime"] < 300)]
                    fig9 = px.histogram(
                        rt_df, x="runtime",
                        nbins=50,
                        title="Movie Runtime Distribution (minutes)",
                        color_discrete_sequence=["#10b981"],
                        labels={"runtime": "Runtime (min)"},
                    )
                    fig9.update_layout(**CHART_TEMPLATE, height=430, bargap=0.05)
                    fig9.add_vline(
                        x=rt_df["runtime"].mean(),
                        line_dash="dash", line_color="#f59e0b",
                        annotation_text=f"Avg: {rt_df['runtime'].mean():.0f} min",
                        annotation_font_color="#f59e0b",
                    )
                    st.plotly_chart(fig9, use_container_width=True)

            # Top grossing movies
            st.markdown('<div class="section-header">💰 Top 15 Highest Grossing Movies</div>', unsafe_allow_html=True)
            top_rev = rev_df.nlargest(15, "revenue")[
                ["title", "revenue_m", rating_col, "genres", "release_year"]
            ].reset_index(drop=True)
            top_rev.index += 1
            top_rev.columns = ["Title", "Revenue ($M)", "IMDB Rating", "Genres", "Year"]
            top_rev["Revenue ($M)"] = top_rev["Revenue ($M)"].round(1)

            fig10 = px.bar(
                top_rev,
                x="Revenue ($M)", y="Title",
                orientation="h",
                title="Top 15 Highest Grossing Movies",
                color="IMDB Rating",
                color_continuous_scale=["#4f46e5", "#7c3aed", "#f59e0b"],
                hover_data=["Year", "Genres"],
                text="Revenue ($M)",
            )
            fig10.update_layout(**CHART_TEMPLATE, height=500)
            fig10.update_traces(texttemplate="$%{text:,.0f}M", textposition="outside")
            fig10.update_yaxes(autorange="reversed")
            st.plotly_chart(fig10, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_recommendation_page(df_model, similarities):
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">🎬 CineMatch</p>
        <p class="hero-subtitle">
            Discover your next favourite film powered by AI &amp; Sentence Transformers
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: Quick Stats only ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔎 Quick Stats")
        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        rating_col_s = "averageRating" if "averageRating" in df_model.columns else "vote_average"
        st.markdown(f"""
        <div style="font-size:0.85rem; color:#94a3b8; line-height:2.2">
        🎬 <b>{len(df_model):,}</b> movies in database<br>
        ⭐ Avg rating: <b>{df_model[rating_col_s].mean():.2f}</b><br>
        📅 Years: <b>{int(df_model['release_year'][df_model['release_year']>0].min())}–{int(df_model['release_year'][df_model['release_year']>0].max())}</b>
        </div>
        """, unsafe_allow_html=True)

    # ── Main Page Preference Controls ──────────────────────────────────────
    with st.expander("🎛️ Filter & Preferences", expanded=True):
        row1_c1, row1_c2, row1_c3 = st.columns([3, 2, 2])

        with row1_c1:
            st.markdown("**🎭 Favourite Genres**")
            selected_genres = st.multiselect(
                "Favourite Genres",
                options=ALL_GENRES,
                default=["Action", "Drama"],
                key="genre_select",
                label_visibility="collapsed",
                placeholder="Choose genres…",
            )

        with row1_c2:
            st.markdown("**⭐ Minimum IMDB Rating**")
            min_rating = st.slider(
                "Min Rating", 0.0, 10.0, 6.0, 0.1,
                label_visibility="collapsed", key="min_rating_slider"
            )

        with row1_c3:
            st.markdown("**📅 Release Year Range**")
            min_year, max_year = st.slider(
                "Year range", 1950, 2025, (2000, 2025), 1,
                label_visibility="collapsed", key="year_range_slider"
            )

        row2_c1, row2_c2, row2_spacer = st.columns([2, 2, 3])

        with row2_c1:
            st.markdown("**🎯 Number of Recommendations**")
            top_n = st.slider(
                "How many movies?", 5, 20, 10, 1,
                label_visibility="collapsed", key="top_n_slider"
            )

        with row2_c2:
            st.markdown("**📐 Grid Columns**")
            cols_per_row = st.select_slider(
                "Cards per row", options=[2, 3, 4, 5], value=4,
                label_visibility="collapsed", key="cols_slider"
            )

    # ── Main Content ────────────────────────────────────────────────────────

    # ── Section 1: By Movie Title ────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Find Movies Similar To...</div>', unsafe_allow_html=True)

    movie_list = df_model["title"].tolist()
    movie_list_sorted = sorted(movie_list)

    search_query = st.text_input(
        "Search movies...",
        value="",
        placeholder="🔎  Type a movie name to search...",
        key="movie_search",
        label_visibility="collapsed",
    )

    if search_query:
        filtered_movies = [m for m in movie_list_sorted if search_query.lower() in m.lower()]
    else:
        filtered_movies = movie_list_sorted

    if not filtered_movies:
        st.warning("No movies match your search. Try a different title.")
        filtered_movies = movie_list_sorted[:100]

    selected_movie = st.selectbox(
        "Select a movie:",
        options=filtered_movies,
        key="movie_select",
        label_visibility="collapsed",
    )

    if selected_movie:
        sel_row = df_model[df_model["title"] == selected_movie].iloc[0]
        info_col1, info_col2 = st.columns([1, 3])
        with info_col1:
            poster_url = get_poster_url(sel_row)
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.markdown(f'<div class="movie-poster-placeholder" style="border-radius:12px;font-size:5rem;height:280px">{POSTER_PLACEHOLDER}</div>', unsafe_allow_html=True)

        with info_col2:
            rating_col = "averageRating" if "averageRating" in df_model.columns else "vote_average"
            rating = float(sel_row.get(rating_col, 0))
            year = int(sel_row.get("release_year", 0))
            genres = str(sel_row.get("genres", ""))
            directors_raw = str(sel_row.get("directors", ""))
            cast_raw = str(sel_row.get("cast", ""))
            overview_raw = str(sel_row.get("overview", ""))

            st.markdown(f"### {selected_movie}")
            st.markdown(f"**📅 Year:** {year if year > 0 else 'N/A'}  &nbsp;|&nbsp;  **⭐ Rating:** {rating:.1f}/10  &nbsp;|&nbsp;  **🎭 Genres:** {genres}")
            st.markdown(f"**🎬 Director:** {directors_raw[:80]}")
            st.markdown(f"**🎭 Cast:** {cast_raw[:120]}...")
            if overview_raw and len(overview_raw) > 10:
                st.markdown(f"**📖 Overview:** {overview_raw[:300]}...")
            else:
                st.markdown("*Overview not available*")
            runtime = int(sel_row.get("runtime", 0))
            revenue = float(sel_row.get("revenue", 0))
            st.markdown(f"**⏱️ Runtime:** {runtime} min  &nbsp;|&nbsp;  **💰 Revenue:** ${revenue/1e6:.1f}M" if revenue > 0 else f"**⏱️ Runtime:** {runtime} min")

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    rec_btn = st.button("🔮 Get AI Recommendations", key="rec_btn_seed")

    if rec_btn and selected_movie and similarities is not None:
        with st.spinner("🤖 Finding similar movies..."):
            recs = get_recommendations(
                selected_movie, similarities, df_model,
                top_n=top_n,
                genre_filter=selected_genres if selected_genres else None,
                min_rating=min_rating,
                min_year=min_year,
                max_year=max_year,
            )
        # Store results in session_state so they survive the rerun from "View Details" clicks
        st.session_state["seed_recs"] = recs.to_dict("records") if len(recs) > 0 else []
        st.session_state["seed_recs_label"] = selected_movie
        # Clear any stale detail panel when new results arrive
        st.session_state.pop("selected_movie_detail", None)
    elif rec_btn and similarities is None:
        st.error("⚠️ The AI model is still loading. Please wait a moment and try again.")

    # ── Detail panel (shown above grid when a card is clicked) ───────────────
    if st.session_state.get("selected_movie_detail") and df_model is not None:
        detail_row = pd.Series(st.session_state.selected_movie_detail)
        render_movie_detail_panel(detail_row, df_model, similarities)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Render seed-movie recommendations (persists across reruns) ───────────
    seed_recs_data = st.session_state.get("seed_recs", [])
    if seed_recs_data:
        seed_label = st.session_state.get("seed_recs_label", "")
        recs_df = pd.DataFrame(seed_recs_data)
        st.markdown(f'<div class="section-header">✨ Top {len(recs_df)} Recommendations for "{seed_label}"</div>', unsafe_allow_html=True)
        render_movie_grid(recs_df, cols_per_row=cols_per_row, section_prefix="seed")
    elif rec_btn and selected_movie:
        st.warning("No recommendations found with current filters. Try relaxing your genre filters or rating threshold.")

    # ── Section 2: Discover by Preferences ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🌟 Discover by Your Taste</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        💡 No specific movie in mind? Click below to discover top-rated films that match your genre preferences and filters above.
    </div>
    """, unsafe_allow_html=True)

    pref_btn = st.button("🌟 Show Top Movies for My Taste", key="rec_btn_pref")

    if pref_btn:
        with st.spinner("🔍 Curating your personalized list..."):
            pref_recs = get_recommendations_by_preferences(
                df_model, similarities,
                preferred_genres=selected_genres,
                min_rating=min_rating,
                min_year=min_year,
                max_year=max_year,
                top_n=top_n,
            )
        # Store preference results in session_state too
        st.session_state["pref_recs"] = pref_recs.to_dict("records") if len(pref_recs) > 0 else []
        st.session_state["pref_recs_label"] = ", ".join(selected_genres) if selected_genres else "All Genres"
        st.session_state.pop("selected_movie_detail", None)

    pref_recs_data = st.session_state.get("pref_recs", [])
    if pref_recs_data:
        genres_display = st.session_state.get("pref_recs_label", "All Genres")
        pref_df = pd.DataFrame(pref_recs_data)
        st.markdown(f'<div class="section-header">🌟 Top {len(pref_df)} Picks for You — {genres_display}</div>', unsafe_allow_html=True)
        render_movie_grid(pref_df, cols_per_row=cols_per_row, section_prefix="pref")
    elif pref_btn:
        st.warning("No movies found with your filters. Try broadening genre selection or lowering rating threshold.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP ENTRY
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Sidebar Navigation ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <span style="font-size:2.5rem">🎬</span>
            <h2 style="color:#a78bfa; margin:4px 0 0 0; font-weight:800">CineMatch</h2>
            <p style="color:#64748b; font-size:0.8rem; margin:0">AI Movie Recommender</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        if "page" not in st.session_state:
            st.session_state.page = "recommend"

        if st.button("🎬  Recommendations", key="nav_rec", use_container_width=True):
            st.session_state.page = "recommend"
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        if st.button("📊  Data Dashboard", key="nav_dash", use_container_width=True):
            st.session_state.page = "dashboard"

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # ── Load Data ────────────────────────────────────────────────────────────
    with st.spinner("📦 Loading movie database..."):
        df_model = load_and_process_data()

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

    # ── Load AI Model (only for recommendation page) ─────────────────────────
    similarities = None
    if st.session_state.get("page", "recommend") == "recommend":
        if SENTENCE_TRANSFORMER_AVAILABLE:
            with st.spinner("🤖 Loading AI model (first run may take a minute)..."):
                similarities = compute_embeddings(df_model)
        else:
            with st.sidebar:
                st.warning("⚠️ sentence-transformers not installed. Run: `pip install sentence-transformers`")

    # ── Render Active Page ─────────────────────────────────────────────────────
    if st.session_state.get("page", "recommend") == "dashboard":
        render_dashboard(df_model)
    else:
        render_recommendation_page(df_model, similarities)


if __name__ == "__main__":
    main()