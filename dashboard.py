# dashboard.py
"""
MovieLens Professional Analytics Dashboard
Comprehensive analysis of viewer behavior, content performance, and recommendation systems
"""
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
import glob
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Configuration
# -----------------------
# Try multiple possible data directory locations
POSSIBLE_DATA_DIRS = [
    "./data",
    "data",
    os.path.join(os.path.dirname(__file__), "data"),
    os.path.join(os.getcwd(), "data")
]

DATA_DIR = None
for directory in POSSIBLE_DATA_DIRS:
    if os.path.exists(directory) and os.path.isdir(directory):
        DATA_DIR = directory
        break

if DATA_DIR is None:
    DATA_DIR = "./data"  # Fallback

REQUIRED_FILES = {
    "ratings": "ratings_df.parquet",
    "movies": "movies.parquet",
    "tags": "tags.parquet",
    "hidden_gems": "hidden_gems.parquet",
    "centroid": "centroid_df.parquet",
    "user_clusters": "user_clusters.parquet",
    "release_stats": "release_stats_table.parquet",
    "genre_stats": "genre_stats_pd.parquet",
    "tag_rating": "tag_rating_pd.parquet",
    "tag_trends": "tag_trends_pd.parquet",
    "activity_trend": "activity_trend_pd.parquet",
    "weighted_popularity": "weighted_popularity.parquet",
    "movie_features": "movie_features.parquet",
    "content_model_metrics": "content_model_metrics.parquet",
    "feature_importance": "feature_importance.parquet",
}

# -----------------------
# Page Config & Styling
# -----------------------
st.set_page_config(
    page_title="MovieLens Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main background with animated gradient */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Smooth page transitions */
    .main > div {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Headers with modern styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        letter-spacing: -1.5px;
        padding: 30px 0 10px 0;
        margin-bottom: 0;
    }
    
    h2 {
        color: #e0e7ff;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.75rem;
        margin-top: 50px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, #667eea, #764ba2);
        border-image-slice: 1;
        letter-spacing: -0.5px;
    }
    
    h3 {
        color: #c7d2fe;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 30px;
        margin-bottom: 15px;
        letter-spacing: -0.3px;
    }
    
    /* Sidebar with glassmorphism */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(10, 14, 39, 0.98) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    /* Sidebar text styling */
    [data-testid="stSidebar"] .element-container {
        color: #e0e7ff;
    }
    
    /* Metric containers with card design */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide delta */
    div[data-testid="stMetricDelta"] {
        display: none;
    }
    
    /* Dataframe styling with modern look */
    .dataframe {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #e0e7ff !important;
        border-radius: 12px !important;
        overflow: hidden;
        font-family: 'Inter', sans-serif;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%) !important;
        color: #e0e7ff !important;
        font-weight: 600;
        padding: 12px !important;
        border: none !important;
        font-family: 'Inter', sans-serif;
    }
    
    .dataframe td {
        border-color: rgba(102, 126, 234, 0.1) !important;
        padding: 10px !important;
    }
    
    /* Buttons with modern gradient */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: none;
        letter-spacing: 0.3px;
        font-size: 0.95rem;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Selectbox and input styling */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #e0e7ff !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: rgba(15, 23, 42, 0.4);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Plotly charts with card wrapper */
    .js-plotly-plot {
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.4);
        padding: 10px;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Section dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), rgba(118, 75, 162, 0.5), transparent);
        margin: 50px 0;
        border-radius: 2px;
    }
    
    /* Info/Error boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #667eea;
        background: rgba(102, 126, 234, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(15, 23, 42, 0.6);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    /* Checkbox styling */
    .stCheckbox {
        color: #e0e7ff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Markdown text */
    .main p, .main li {
        color: #cbd5e1;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
    }
    
    /* Success/Info/Warning/Error messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Helper Functions
# -----------------------
def load_chunked_parquet(base_filename, data_dir):
    """Load a parquet file that may be split into chunks"""
    # Remove .parquet extension to get base name
    base_name = base_filename.replace('.parquet', '')
    
    # Look for chunked files
    chunk_pattern = os.path.join(data_dir, f"{base_name}_chunk_*.parquet")
    chunk_files = sorted(glob.glob(chunk_pattern))
    
    if chunk_files:
        # Load and combine all chunks
        chunks = []
        for f in chunk_files:
            try:
                chunks.append(pd.read_parquet(f))
            except Exception as e:
                st.warning(f"Error loading chunk {f}: {e}")
                continue
        
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return None
    
    # If no chunks, try loading the regular file
    full_path = os.path.join(data_dir, base_filename)
    if os.path.exists(full_path):
        try:
            return pd.read_parquet(full_path)
        except Exception as e:
            st.error(f"Error loading {base_filename}: {e}")
            return None
    
    # File doesn't exist in any form
    return None

@st.cache_data
def load_parquet_file(filename):
    """Load parquet file (handles both single files and chunks)"""
    if not filename:
        return None
    
    # Try loading (handles both chunked and non-chunked)
    df = load_chunked_parquet(filename, DATA_DIR)
    return df

def get_chart_layout(title, height=500, theme="plotly_dark"):
    """Returns consistent chart layout configuration"""
    return dict(
        title=dict(text=title, font=dict(size=18, color="#c7d2fe", family="Inter")),
        height=height,
        template=theme,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e7ff", family="Inter"),
        margin=dict(l=50, r=50, t=80, b=50)
    )

@st.cache_resource
def load_content_model():
    """Load the content-based model"""
    try:
        model_path = os.path.join(DATA_DIR, "content_based_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -----------------------
# Header
# -----------------------
st.markdown("# MovieLens Analytics Dashboard")
st.markdown("<p style='font-size: 1.1rem; color: #94a3b8; margin-top: -10px; font-weight: 400;'>Professional insights into viewer behavior, content performance, and recommendation patterns</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Sidebar
# -----------------------
# Logo/Title
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem 0 1.5rem 0;'>
    <h1 style='font-size: 1.8rem; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;'>
        🎬 MovieLens
    </h1>
    <p style='color: #94a3b8; font-size: 0.85rem; margin: 0.5rem 0 0 0;'>Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation Section
st.sidebar.markdown("### 📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Analytics Overview", "Recommendation Models"],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Filters Section
st.sidebar.markdown("### ⚙️ Filters")

# Time Range
with st.sidebar.expander("📅 Time Range", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        min_year = st.number_input("From", value=1990, step=1, min_value=1900, max_value=2023, key="min_year")
    with col2:
        max_year = st.number_input("To", value=2023, step=1, min_value=1900, max_value=2023, key="max_year")
    
    if min_year > max_year:
        st.error("⚠️ Min year must be ≤ max year")

# Display Settings
with st.sidebar.expander("🎨 Display", expanded=False):
    show_raw_data = st.checkbox("Show Data Tables", False)
    chart_theme = st.selectbox("Chart Theme", ["plotly_dark", "plotly", "seaborn"], index=0)

# -----------------------
# Data Loading
# -----------------------
with st.spinner("Loading data..."):
    data = {}
    missing_files = []
    loaded_files = []
    
    for key, fname in REQUIRED_FILES.items():
        df = load_parquet_file(fname)
        if df is None:
            missing_files.append(fname)
        else:
            data[key] = df
            loaded_files.append(f"{fname} ({len(df)} rows)")
    
    if missing_files:
        st.error(f"Missing files: {', '.join(missing_files)}")
        st.info(f"Looking in: {os.path.abspath(DATA_DIR)}")
        
        if os.path.exists(DATA_DIR):
            st.write("Files in data directory:")
            st.write(os.listdir(DATA_DIR)[:20])
        
        st.stop()

# Unpack data
ratings_df = data.get("ratings")
movies = data.get("movies")
tags = data.get("tags")
hidden_gems = data.get("hidden_gems")
centroid_df = data.get("centroid")
user_clusters = data.get("user_clusters")
release_stats_table = data.get("release_stats")
genre_stats_pd = data.get("genre_stats")
tag_rating_pd = data.get("tag_rating")
tag_trends_pd = data.get("tag_trends")
activity_trend_pd = data.get("activity_trend")
weighted_popularity = data.get("weighted_popularity")
movie_features = data.get("movie_features")
content_model_metrics = data.get("content_model_metrics")
feature_importance = data.get("feature_importance")

# Load ML model
content_model = load_content_model()

# -----------------------
# Genre Helper (used in sidebar)
# -----------------------
def safe_genre_iter(g):
    if isinstance(g, (list, tuple)): return tuple(g)
    if isinstance(g, np.ndarray): return tuple(g.tolist())
    return (g,)

try:
    all_genres = sorted({genre for sub in movies["genre_list"].dropna() for genre in safe_genre_iter(sub)})
except Exception as e:
    st.error(f"Error loading genres: {e}")
    all_genres = []

# -----------------------
# PAGE ROUTING
# -----------------------

if page == "Analytics Overview":
    # -----------------------
    # KPI Cards
    # -----------------------
    st.markdown("### Key Performance Indicators")

    avg_rating = ratings_df["rating"].mean() if ratings_df is not None and "rating" in ratings_df.columns else 0
    total_users = ratings_df["userId"].nunique() if ratings_df is not None and "userId" in ratings_df.columns else 0
    total_movies = len(movies) if movies is not None else 0
    total_ratings = len(ratings_df) if ratings_df is not None else 0

    try:
        top_genre = genre_stats_pd.sort_values("avg_rating", ascending=False).iloc[0]["genre_list"]
    except:
        top_genre = "N/A"

    try:
        movie_avg = ratings_df.groupby("movieId")["rating"].mean().reset_index()
        movie_count = ratings_df.groupby("movieId")["rating"].count().reset_index().rename(columns={"rating":"num_ratings"})
        movie_stats = movie_avg.merge(movie_count, on="movieId").merge(movies[["movieId","title"]], on="movieId")
        top_movie_row = movie_stats.sort_values(["rating","num_ratings"], ascending=[False,False]).iloc[0]
        top_movie = top_movie_row["title"]
    except:
        top_movie = "N/A"

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

    with col1:
        st.metric("TOTAL RATINGS", f"{total_ratings:,}")
    with col2:
        st.metric("TOTAL MOVIES", f"{total_movies:,}")
    with col3:
        st.metric("ACTIVE USERS", f"{total_users:,}")
    with col4:
        st.metric("AVG RATING", f"{avg_rating:.2f}/5.0")
    with col5:
        st.markdown(f"**Top Genre:** {top_genre}")
        st.markdown(f"**Top Movie:** {top_movie}")

    st.markdown("---")

    # -----------------------
    # Genre Filter
    # -----------------------
    genre_filter = st.sidebar.multiselect("Filter by Genre", all_genres, default=[])
    
    # -----------------------
    # Movie Search & Discovery
    # -----------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Discover")
    
    with st.sidebar.expander("🎬 Search Movies", expanded=True):
        search_term = st.text_input("Search", placeholder="Type movie title...", label_visibility="collapsed", key="movie_search")
        
        if search_term:
            search_results = movies[movies["title"].str.contains(search_term, case=False, na=False)]
            if len(search_results) > 0:
                st.markdown(f"**{len(search_results)} found:**")
                for idx, row in search_results.head(5).iterrows():
                    st.markdown(f"• {row['title']}")
                if len(search_results) > 5:
                    st.caption(f"... and {len(search_results) - 5} more")
            else:
                st.warning("No movies found")
    
    # Random Movie with optional genre preference
    with st.sidebar.expander("🎲 Random Movie", expanded=False):
        st.markdown("**Get a random movie suggestion**")
        
        # Optional genre preference
        surprise_genres = st.multiselect(
            "Prefer specific genres? (optional)",
            all_genres,
            default=[],
            key="surprise_genre_filter"
        )
        
        if st.button("🎲 Surprise Me!", use_container_width=True, key="surprise_btn"):
            # Filter by genre if selected
            if surprise_genres:
                filtered_movies = movies[movies["genre_list"].apply(
                    lambda x: any(g in surprise_genres for g in (
                        x.tolist() if isinstance(x, np.ndarray) else 
                        x if isinstance(x, (list, tuple)) else [x]
                    )) if x is not None else False
                )]
                if len(filtered_movies) == 0:
                    st.warning("No movies found with selected genres")
                else:
                    random_movie = filtered_movies.sample(1).iloc[0]
            else:
                random_movie = movies.sample(1).iloc[0]
            
            if 'random_movie' in locals():
                random_genres = random_movie["genre_list"]
                if isinstance(random_genres, np.ndarray):
                    genre_str = ", ".join(random_genres.tolist())
                elif isinstance(random_genres, (list, tuple)):
                    genre_str = ", ".join(random_genres)
                else:
                    genre_str = str(random_genres)
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                            padding: 1rem; border-radius: 10px; border: 1px solid rgba(102, 126, 234, 0.3); margin-top: 0.5rem;'>
                    <p style='color: #e0e7ff; font-weight: 600; margin: 0 0 0.5rem 0;'>{random_movie['title']}</p>
                    <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>{genre_str}</p>
                </div>
                """, unsafe_allow_html=True)

    # -----------------------
    # Section 1: User Behavior Insights
    # -----------------------
    st.markdown("## User Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Rater Classification")
        user_pd = user_clusters.copy()
        if "rater_type" in user_pd.columns and not user_pd.empty:
            fig = px.histogram(
                user_pd, x="rater_type", color="rater_type",
                category_orders={"rater_type":["harsh","neutral","generous"]},
                color_discrete_map={"harsh":"#f87171", "neutral":"#a78bfa", "generous":"#34d399"}
            )
            fig.update_layout(**get_chart_layout("User Rating Patterns", theme=chart_theme, height=400))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Activity Trends")
        fig = px.line(
            activity_trend_pd,
            x="year",
            y="avg_ratings_per_user",
            markers=True,
            color_discrete_sequence=["#667eea"]
        )
        fig.update_layout(**get_chart_layout("Average Ratings per User Over Time", theme=chart_theme, height=400))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Rating Evolution Analysis")
    rp = release_stats_table[(release_stats_table["release_year"] >= min_year) & (release_stats_table["release_year"] <= max_year)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rp["release_year"], 
        y=rp["num_ratings"], 
        name="Rating Volume", 
        marker_color="#8b5cf6"
    ))
    fig.add_trace(go.Scatter(
        x=rp["release_year"], 
        y=rp["avg_rating"], 
        mode="lines+markers", 
        name="Average Rating", 
        marker=dict(color="#667eea", size=8),
        yaxis="y2"
    ))
    fig.update_layout(
        **get_chart_layout("Rating Volume & Average by Release Year", theme=chart_theme),
        yaxis2=dict(overlaying='y', side='right', range=[0,5], title="Average Rating")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -----------------------
    # Section 2: Content Insights
    # -----------------------
    st.markdown("## Content Performance Analysis")

    genre_df = genre_stats_pd.copy()
    if genre_filter:
        genre_df = genre_df[genre_df["genre_list"].isin(genre_filter)]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Genre Quality Ratings")
        fig = px.bar(
            genre_df.sort_values("avg_rating"), 
            x="avg_rating", 
            y="genre_list", 
            orientation="h",
            color="avg_rating",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(**get_chart_layout("Average Rating by Genre", theme=chart_theme, height=500))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Genre Popularity")
        fig = px.bar(
            genre_df.sort_values("num_ratings", ascending=False), 
            x="num_ratings", 
            y="genre_list", 
            orientation="h",
            color="num_ratings",
            color_continuous_scale="Blues"
        )
        fig.update_layout(**get_chart_layout("Rating Volume by Genre", theme=chart_theme, height=500))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tag Sentiment Analysis")
    fig = px.bar(
        tag_rating_pd, 
        x="tag", 
        y="avg_rating", 
        color="num_ratings",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(**get_chart_layout("Sentiment Tags vs Average Rating", theme=chart_theme))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tag Trends Over Time")
    top_tags = sorted(tags["tag"].value_counts().index[:20])
    groups = [top_tags[i:i+5] for i in range(0, 20, 5)]

    for i, grp in enumerate(groups, 1):
        grp_df = tag_trends_pd[tag_trends_pd["tag"].isin(grp)]
        fig = px.line(
            grp_df, 
            x="year", 
            y="count", 
            color="tag",
            markers=True
        )
        fig.update_layout(**get_chart_layout(f"Tag Trends Group {i}", theme=chart_theme, height=400))
        fig.update_xaxes(range=[2005, 2025])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -----------------------
    # Section 3: Hidden Patterns & Clusters
    # -----------------------
    st.markdown("## Advanced Analytics")

    st.markdown("### User Segmentation by Genre Preferences")

    cent = centroid_df.copy()

    # Force correct orientation
    if "Action" in cent.columns or "Drama" in cent.columns or "Comedy" in cent.columns:
        cent = cent.T

    cent.index.name = "Genre"
    cent.columns = cent.columns.astype(str)

    cluster_options = list(cent.columns)

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_clusters = st.multiselect(
            "Select clusters to compare",
            cluster_options,
            default=cluster_options[:min(3, len(cluster_options))]
        )

    with col2:
        st.markdown(f"**Total Clusters:** {len(cluster_options)}")
        st.markdown(f"**Genres Analyzed:** {len(cent.index)}")

    if selected_clusters:
        comp = cent[selected_clusters].copy().reset_index()
        
        comp_long = comp.melt(
            id_vars="Genre",
            value_vars=selected_clusters,
            var_name="Cluster",
            value_name="AvgRating"
        )
        
        fig = px.bar(
            comp_long,
            x="Cluster",
            y="AvgRating",
            color="Genre",
            barmode="group",
            hover_data={"Cluster": True, "Genre": True, "AvgRating": ":.2f"}
        )
        
        fig.update_layout(**get_chart_layout("Genre Preferences by User Cluster", theme=chart_theme))
        st.plotly_chart(fig, use_container_width=True)
        
        if show_raw_data:
            st.markdown("#### Detailed Cluster Data")
            ordered = comp_long.sort_values(["Cluster", "Genre"])[["Cluster", "Genre", "AvgRating"]]
            st.dataframe(ordered, use_container_width=True)
            
            # Download option
            csv = ordered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Cluster Data",
                csv,
                "cluster_comparison.csv",
                "text/csv"
            )
    else:
        st.info("Select at least one cluster to view the analysis")

    st.markdown("### Hidden Gems Discovery")
    st.markdown("*High-rated movies with low visibility*")
    
    if hidden_gems is not None and len(hidden_gems) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hidden Gems Found", f"{len(hidden_gems):,}")
        with col2:
            st.metric("Avg Rating", f"{hidden_gems['avg_rating'].mean():.2f}")
        with col3:
            st.metric("Avg Votes", f"{int(hidden_gems['num_ratings'].mean())}")
        
        # Show top hidden gems
        num_gems = st.slider("Number of hidden gems to display", 5, 50, 20, key="hidden_gems_slider")
        
        display_gems = hidden_gems[["title", "avg_rating", "num_ratings"]].head(num_gems).copy()
        display_gems.columns = ["Movie Title", "Avg Rating", "# Ratings"]
        display_gems["Avg Rating"] = display_gems["Avg Rating"].round(2)
        
        st.dataframe(display_gems, use_container_width=True, height=400)
        
        # Download button
        csv = display_gems.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hidden Gems",
            data=csv,
            file_name="hidden_gems.csv",
            mime="text/csv",
        )
    else:
        st.warning("No hidden gems data available")

elif page == "Recommendation Models":
    # -----------------------
    # RECOMMENDATION MODELS PAGE
    # -----------------------
    st.markdown("## Movie Recommendation System")
    st.markdown("*Compare different recommendation approaches*")
    
    # Model selector
    model_type = st.radio(
        "Select Recommendation Model",
        ["Weighted Popularity (Non-Personalized)", "Content-Based Filtering (ML)"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if model_type == "Weighted Popularity (Non-Personalized)":
        st.markdown("### Weighted Popularity Model")
        st.markdown("*IMDb-style weighted score based on ratings volume and average*")
        
        if weighted_popularity is None:
            st.error("Weighted popularity data not found. Please ensure weighted_popularity.parquet is in the data directory.")
            st.info("This file should contain pre-computed weighted scores for all movies.")
        else:
            # Merge with movies to get genres
            wp_with_genres = weighted_popularity.merge(movies[["movieId", "genre_list"]], on="movieId", how="left")
            
            # Filters
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Genre filter - handle numpy arrays
                all_genres_wp = set()
                for genres in wp_with_genres["genre_list"].dropna():
                    if isinstance(genres, np.ndarray):
                        all_genres_wp.update(genres.tolist())
                    elif isinstance(genres, (list, tuple)):
                        all_genres_wp.update(genres)
                    else:
                        all_genres_wp.add(genres)
                all_genres_wp = sorted(all_genres_wp)
                selected_genres = st.multiselect("Filter by Genre", all_genres_wp, default=[])
            
            with col2:
                min_votes = st.number_input("Min Votes", min_value=0, value=100, step=50)
            
            with col3:
                n_recommendations = st.slider("Top N Movies", 5, 100, 20)
            
            # Apply filters
            filtered_wp = wp_with_genres.copy()
            
            if selected_genres:
                def check_genre_match(x):
                    if x is None:
                        return False
                    if isinstance(x, np.ndarray):
                        return any(g in selected_genres for g in x.tolist())
                    elif isinstance(x, (list, tuple)):
                        return any(g in selected_genres for g in x)
                    else:
                        return x in selected_genres
                
                filtered_wp = filtered_wp[filtered_wp["genre_list"].apply(check_genre_match)]
            
            filtered_wp = filtered_wp[filtered_wp["v"] >= min_votes]
            
            # Get top N
            top_movies = filtered_wp.head(n_recommendations)
            
            # Show count
            st.markdown(f"**Showing {len(top_movies)} of {len(filtered_wp):,} movies** (Total in dataset: {len(weighted_popularity):,})")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Top Recommended Movies")
                display_df = top_movies[["title", "score", "R", "v"]].copy()
                display_df.columns = ["Movie Title", "Weighted Score", "Avg Rating", "# Ratings"]
                display_df["Weighted Score"] = display_df["Weighted Score"].round(3)
                display_df["Avg Rating"] = display_df["Avg Rating"].round(2)
                st.dataframe(display_df, use_container_width=True, height=600)
                
                # Download button
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Recommendations",
                    data=csv,
                    file_name="top_movies_weighted_popularity.csv",
                    mime="text/csv",
                )
            
            with col2:
                st.markdown("#### Model Formula")
                st.latex(r"""
                Score = \frac{v}{v+m} \times R + \frac{m}{v+m} \times C
                """)
                st.markdown("""
                Where:
                - **v** = number of ratings
                - **R** = average rating
                - **m** = minimum votes threshold
                - **C** = global mean rating
                """)
                
                st.markdown("#### Model Stats")
                st.metric("Movies Shown", f"{len(top_movies)}")
                st.metric("Avg Score", f"{top_movies['score'].mean():.3f}")
                st.metric("Avg Votes", f"{int(top_movies['v'].mean()):,}")
            
    else:  # Content-Based Filtering
        st.markdown("### Content-Based Filtering Model")
        st.markdown("*ML-powered recommendations based on movie features (genres, tags, metadata)*")
        
        if content_model is None:
            st.error("Content-based model not found. Please ensure content_based_model.pkl is in the data directory.")
        elif content_model_metrics is None or feature_importance is None:
            st.error("Model metrics or feature importance data not found.")
            st.info("Please ensure content_model_metrics.parquet and feature_importance.parquet are in the data directory.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Performance")
                metrics_display = content_model_metrics.set_index("metric")
                
                for metric, value in metrics_display.iterrows():
                    st.metric(metric, f"{value['value']:.4f}")
            
            with col2:
                st.markdown("#### Feature Importance")
                top_features = feature_importance.head(10)
                
                fig = px.bar(
                    top_features,
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Reds"
                )
                fig.update_layout(**get_chart_layout("Top 10 Features", height=400, theme=chart_theme))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Movie recommendation interface
            st.markdown("#### Get Recommendations for a Movie")
            
            if movie_features is None or movies is None:
                st.error("Movie features or movies data not found.")
            else:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Movie selector with search
                    movie_titles = movies["title"].tolist()
                    selected_movie_title = st.selectbox("Select a movie", movie_titles, 
                                                        help="Start typing to search for a movie")
                
                with col2:
                    num_recommendations = st.number_input("Number of recommendations", min_value=5, max_value=50, value=10)
                
                if st.button("🎬 Get Similar Movies", use_container_width=True):
                    # Find selected movie ID
                    selected_movie_id = movies[movies["title"] == selected_movie_title]["movieId"].values[0]
                    selected_movie_info = movies[movies["movieId"] == selected_movie_id].iloc[0]
                    
                    # Get movie features
                    if selected_movie_id in movie_features.index:
                        selected_features = movie_features.loc[selected_movie_id].values.reshape(1, -1)
                        
                        # Predict rating
                        predicted_rating = content_model.predict(selected_features)[0]
                        
                        # Show selected movie info
                        st.markdown("---")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"### 🎥 {selected_movie_title}")
                            genres = selected_movie_info["genre_list"]
                            if isinstance(genres, np.ndarray):
                                genre_str = ", ".join(genres.tolist())
                            elif isinstance(genres, (list, tuple)):
                                genre_str = ", ".join(genres)
                            else:
                                genre_str = str(genres)
                            st.markdown(f"**Genres:** {genre_str}")
                        
                        with col2:
                            st.metric("Predicted Rating", f"{predicted_rating:.2f} / 5.0")
                        
                        st.markdown("---")
                        
                        # Find similar movies (cosine similarity on features)
                        similarities = cosine_similarity(selected_features, movie_features.values)[0]
                        similar_indices = similarities.argsort()[-(num_recommendations+1):-1][::-1]  # Top N similar (excluding self)
                        
                        similar_movies = movie_features.iloc[similar_indices].index.tolist()
                        similar_titles = movies[movies["movieId"].isin(similar_movies)][["movieId", "title", "genre_list"]].copy()
                        similar_titles["similarity"] = [similarities[idx] for idx in similar_indices]
                        similar_titles = similar_titles.sort_values("similarity", ascending=False)
                        
                        # Calculate genre overlap
                        if isinstance(genres, np.ndarray):
                            selected_genres = set(genres.tolist())
                        elif isinstance(genres, (list, tuple)):
                            selected_genres = set(genres)
                        else:
                            selected_genres = {genres}
                        
                        def calc_genre_overlap(row_genres):
                            # Check for None or empty first
                            if row_genres is None:
                                return 0
                            # Handle numpy arrays
                            if isinstance(row_genres, np.ndarray):
                                if len(row_genres) == 0:
                                    return 0
                                row_genre_set = set(row_genres.tolist())
                            elif isinstance(row_genres, (list, tuple)):
                                if len(row_genres) == 0:
                                    return 0
                                row_genre_set = set(row_genres)
                            else:
                                # For scalar values, check if it's NaN
                                try:
                                    if pd.isna(row_genres):
                                        return 0
                                except (TypeError, ValueError):
                                    pass
                                row_genre_set = {row_genres}
                            overlap = len(selected_genres & row_genre_set)
                            return overlap
                        
                        similar_titles["genre_overlap"] = similar_titles["genre_list"].apply(calc_genre_overlap)
                        
                        st.markdown(f"#### 🎯 Top {len(similar_titles)} Similar Movies")
                        
                        # Format display
                        display_similar = similar_titles[["title", "similarity", "genre_overlap", "genre_list"]].copy()
                        display_similar["genre_list"] = display_similar["genre_list"].apply(
                            lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else str(x)
                        )
                        display_similar.columns = ["Movie Title", "Similarity Score", "Shared Genres", "Genres"]
                        display_similar["Similarity Score"] = display_similar["Similarity Score"].round(3)
                        
                        st.dataframe(display_similar, use_container_width=True, height=400)
                        
                        # Download button
                        csv = display_similar.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Similar Movies",
                            data=csv,
                            file_name=f"similar_to_{selected_movie_title.replace(' ', '_')[:30]}.csv",
                            mime="text/csv",
                        )
                        
                        # Show explanation
                        st.info(f"""
                        **How similarity works:** Movies are compared based on their features (genres, tags, metadata). 
                        A score of 1.0 means identical features, while 0.0 means completely different. 
                        The selected movie shares {int(similar_titles['genre_overlap'].mean())} genres on average with these recommendations.
                        """)
                    else:
                        st.error("Movie features not found for this movie.")