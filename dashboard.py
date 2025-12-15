# dashboard.py
"""
MovieLens Professional Analytics Dashboard
Comprehensive analysis of viewer behavior, content performance, and recommendation systems
"""
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# -----------------------
# Configuration
# -----------------------
DATA_DIR = os.environ.get("MOVIELENS_DATA_DIR", "./data")

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
    /* Main background */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Headers */
    h1 {
        color: #e94560;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
        padding: 20px 0;
    }

    h2, h3 {
        color: #eaeaea;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 500;
        margin-top: 30px;
        border-bottom: 2px solid #e94560;
        padding-bottom: 10px;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%);
    }

    /* Metric containers */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #e94560;
        font-weight: 600;
    }

    [data-testid="stMetricLabel"] {
        color: #c5c5c5;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Hide delta */
    div[data-testid="stMetricDelta"] {
        display: none;
    }

    /* Dataframe styling */
    .dataframe {
        background-color: #16213e !important;
        color: #eaeaea !important;
    }

    .dataframe th {
        background-color: #0f3460 !important;
        color: #e94560 !important;
        font-weight: 600;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #e94560 0%, #c03555 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 12px;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.4);
    }

    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 8px;
    }

    /* Section dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e94560, transparent);
        margin: 40px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Helper Functions
# -----------------------
import glob

def load_chunked_parquet(base_filename, data_dir):
    """Load a parquet file that may be split into chunks"""
    import glob
    
    # Check if chunked files exist
    base_name = base_filename.replace('.parquet', '')
    chunk_pattern = os.path.join(data_dir, f"{base_name}_chunk_*.parquet")
    chunk_files = sorted(glob.glob(chunk_pattern))
    
    if chunk_files:
        # Load and combine all chunks
        print(f"Loading {len(chunk_files)} chunks for {base_filename}")
        chunks = [pd.read_parquet(f) for f in chunk_files]
        return pd.concat(chunks, ignore_index=True)
    else:
        # Try loading single file (non-chunked)
        full_path = os.path.join(data_dir, base_filename)
        if os.path.exists(full_path):
            print(f"Loading single file: {base_filename}")
            return pd.read_parquet(full_path)
        
        print(f"File not found: {base_filename}")
        return None


        
def get_chart_layout(title, height=500, theme="plotly_dark"):
    """Returns consistent chart layout configuration"""
    return dict(
        title=dict(text=title, font=dict(size=16, color="#e94560", family="Segoe UI")),
        height=height,
        template=theme,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eaeaea", family="Segoe UI"),
        margin=dict(l=50, r=50, t=80, b=50)
    )

@st.cache_data
def load_parquet_file(filename):
    """Load parquet file (handles both single files and chunks)"""
    try:
        return load_chunked_parquet(filename, DATA_DIR)
    except Exception as e:
        st.warning(f"Error reading {filename}: {e}")
        return None

@st.cache_resource
def load_content_model():
    """Load the content-based model"""
    try:
        model = joblib.load(os.path.join(DATA_DIR, "content_based_model.pkl"))
        return model
    except:
        return None

# -----------------------
# Header
# -----------------------
st.markdown("# MovieLens Analytics Dashboard")
st.markdown("*Professional insights into viewer behavior, content performance, and recommendation patterns*")
st.markdown("---")

# -----------------------
# Sidebar
# -----------------------
st.sidebar.markdown("## Dashboard Controls")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["Analytics Overview", "Recommendation Models"],
    index=0
)

data_dir_input = st.sidebar.text_input("Data Directory", DATA_DIR, help="Path to parquet files")
if data_dir_input:
    DATA_DIR = data_dir_input

st.sidebar.markdown("### Time Range Filters")
col1, col2 = st.sidebar.columns(2)
with col1:
    min_year = st.number_input("From", value=1990, step=1, min_value=1900, max_value=2023)
with col2:
    max_year = st.number_input("To", value=2023, step=1, min_value=1900, max_value=2023)

if min_year > max_year:
    st.sidebar.error("Minimum year must be less than or equal to maximum year")

st.sidebar.markdown("### Display Settings")
show_raw_data = st.sidebar.checkbox("Show Data Tables", False)
chart_theme = st.sidebar.selectbox("Chart Theme", ["plotly_dark", "plotly", "seaborn"], index=0)

# -----------------------
# Data Loading
# -----------------------
with st.spinner("Loading data..."):
    data = {}
    missing_files = []

    for key, fname in REQUIRED_FILES.items():
        df = load_parquet_file(os.path.join(DATA_DIR, fname))
        if df is None:
            missing_files.append(fname)
        else:
            data[key] = df

    if missing_files:
        st.error(f"Missing files: {', '.join(missing_files)}")
        st.info("Please ensure all required parquet files are in the data directory.")
        st.stop()

# Unpack data
ratings_df = data["ratings"]
movies = data["movies"]
tags = data["tags"]
hidden_gems = data["hidden_gems"]
centroid_df = data["centroid"]
user_clusters = data["user_clusters"]
release_stats_table = data["release_stats"]
genre_stats_pd = data["genre_stats"]
tag_rating_pd = data["tag_rating"]
tag_trends_pd = data["tag_trends"]
activity_trend_pd = data["activity_trend"]
weighted_popularity = data["weighted_popularity"]
movie_features = data["movie_features"]
content_model_metrics = data["content_model_metrics"]
feature_importance = data["feature_importance"]

# Load ML model
content_model = load_content_model()

# -----------------------
# PAGE ROUTING
# -----------------------

if page == "Analytics Overview":
    # -----------------------
    # KPI Cards
    # -----------------------
    st.markdown("### Key Performance Indicators")

    avg_rating = ratings_df["rating"].mean() if "rating" in ratings_df.columns else 0
    total_users = ratings_df["userId"].nunique() if "userId" in ratings_df.columns else 0
    total_movies = len(movies)
    total_ratings = len(ratings_df)

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
    def safe_genre_iter(g):
        if isinstance(g, (list, tuple)): return tuple(g)
        if isinstance(g, np.ndarray): return tuple(g.tolist())
        return (g,)

    all_genres = sorted({genre for sub in movies["genre_list"].dropna() for genre in safe_genre_iter(sub)})
    genre_filter = st.sidebar.multiselect("Filter by Genre", all_genres, default=[])

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
                color_discrete_map={"harsh":"#ff6b6b", "neutral":"#ffd93d", "generous":"#6bcf7f"}
            )
            fig.update_layout(**get_chart_layout("User Rating Patterns", theme=chart_theme, height=400))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Activity Trends")
        fig = px.line(
            activity_trend_pd,
            x="year",
            y="avg_ratings_per_user",
            markers=True
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
        marker_color="#4ecdc4"
    ))
    fig.add_trace(go.Scatter(
        x=rp["release_year"],
        y=rp["avg_rating"],
        mode="lines+markers",
        name="Average Rating",
        marker=dict(color="#e94560", size=8),
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

    if show_raw_data:
        st.dataframe(
            hidden_gems[["title", "avg_rating", "num_ratings"]].head(20),
            use_container_width=True
        )
    else:
        st.markdown(f"**{len(hidden_gems)} hidden gems identified** (rating ≥ 4.5, ratings ≤ 100)")

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

        # Show top N movies
        n_recommendations = st.slider("Number of recommendations", 5, 50, 20)

        top_movies = weighted_popularity.head(n_recommendations)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Top Recommended Movies")
            display_df = top_movies[["title", "score", "R", "v"]].copy()
            display_df.columns = ["Movie Title", "Weighted Score", "Avg Rating", "# Ratings"]
            display_df["Weighted Score"] = display_df["Weighted Score"].round(3)
            display_df["Avg Rating"] = display_df["Avg Rating"].round(2)
            st.dataframe(display_df, use_container_width=True, height=600)

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
            st.metric("Total Movies Ranked", f"{len(weighted_popularity):,}")
            st.metric("Avg Score", f"{weighted_popularity['score'].mean():.3f}")

    else:  # Content-Based Filtering
        st.markdown("### Content-Based Filtering Model")
        st.markdown("*ML-powered recommendations based on movie features (genres, tags, metadata)*")

        if content_model is None:
            st.error("Content-based model not found. Please ensure content_based_model.pkl is in the data directory.")
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

            # Movie selector
            movie_titles = movies["title"].tolist()
            selected_movie_title = st.selectbox("Select a movie", movie_titles)

            if st.button("Get Similar Movies"):
                # Find selected movie ID
                selected_movie_id = movies[movies["title"] == selected_movie_title]["movieId"].values[0]

                # Get movie features
                if selected_movie_id in movie_features.index:
                    selected_features = movie_features.loc[selected_movie_id].values.reshape(1, -1)

                    # Predict rating
                    predicted_rating = content_model.predict(selected_features)[0]

                    st.success(f"Predicted Rating for this movie: **{predicted_rating:.2f}** / 5.0")

                    # Find similar movies (simple cosine similarity on features)
                    from sklearn.metrics.pairwise import cosine_similarity

                    similarities = cosine_similarity(selected_features, movie_features.values)[0]
                    similar_indices = similarities.argsort()[-11:-1][::-1]  # Top 10 similar (excluding self)

                    similar_movies = movie_features.iloc[similar_indices].index.tolist()
                    similar_titles = movies[movies["movieId"].isin(similar_movies)][["movieId", "title"]]
                    similar_titles["similarity"] = [similarities[idx] for idx in similar_indices]
                    similar_titles = similar_titles.sort_values("similarity", ascending=False)

                    st.markdown("#### Top 10 Similar Movies")
                    display_similar = similar_titles[["title", "similarity"]].copy()
                    display_similar.columns = ["Movie Title", "Similarity Score"]
                    display_similar["Similarity Score"] = display_similar["Similarity Score"].round(3)
                    st.dataframe(display_similar, use_container_width=True)
                else:
                    st.error("Movie features not found for this movie.")

