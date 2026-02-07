import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from difflib import get_close_matches

# Page configuration
st.set_page_config(
    page_title="Netflix Content Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #E50914;
    }
    .metric-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load data
@st.cache_data
def load_data():
    """Load preprocessed data and feature vectors."""
    try:
        df = pd.read_pickle("models/df.pkl")
        X_reduced = np.load("models/X_reduced.npy")

        # Try to load embeddings if available
        try:
            embeddings = np.load("models/embeddings.npy")
        except:
            embeddings = None

        return df, X_reduced, embeddings
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info(
            "Please ensure df.pkl and X_reduced.npy are in the same directory as this app."
        )
        st.stop()


# Recommendation function with error handling
def recommend_enhanced(title, df, vectors, top_n=7):
    """
    Enhanced recommendation function with fuzzy matching and error handling.

    Parameters:
    -----------
    title : str
        The title to find recommendations for
    df : pd.DataFrame
        The Netflix dataset
    vectors : np.array
        The feature vectors (embeddings or reduced features)
    top_n : int
        Number of recommendations to return

    Returns:
    --------
    tuple: (pd.DataFrame, str)
        DataFrame with recommendations and status message
    """
    try:
        # Case-insensitive exact match
        matching_titles = df[df["title"].str.lower() == title.lower()]

        if matching_titles.empty:
            # Fuzzy matching for suggestions
            suggestions = get_close_matches(
                title.lower(), df["title"].str.lower().tolist(), n=5, cutoff=0.6
            )

            if suggestions:
                suggestion_list = []
                for sugg in suggestions:
                    actual_title = df[df["title"].str.lower() == sugg].iloc[0]["title"]
                    suggestion_list.append(actual_title)

                return (
                    pd.DataFrame(),
                    f"Title '{title}' not found. Did you mean: {', '.join(suggestion_list)}?",
                )
            else:
                return (
                    pd.DataFrame(),
                    f"Title '{title}' not found and no similar titles found.",
                )

        idx = matching_titles.index[0]

        # Calculate similarities
        sims = cosine_similarity(vectors[idx].reshape(1, -1), vectors)[0]

        # Get top similar items (excluding the item itself)
        top_idx = np.argsort(sims)[::-1][1 : top_n + 1]

        recommendations = df.iloc[top_idx][
            ["title", "type", "listed_in", "description", "rating", "release_year"]
        ].copy()
        recommendations["similarity_score"] = sims[top_idx]

        return recommendations, "Success"

    except Exception as e:
        return pd.DataFrame(), f"Error in recommendation: {str(e)}"


# Main app
def main():
    # Load data
    df, X_reduced, embeddings = load_data()

    # Header
    st.markdown(
        '<div class="main-header">🎬 Netflix Content Recommender</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Discover your next favorite movie or TV show using AI-powered recommendations</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg",
            width=200,
        )
        st.markdown("---")

        st.subheader("⚙️ Settings")

        # Model selection
        if embeddings is not None:
            use_embeddings = st.radio(
                "Recommendation Model",
                options=["Advanced (Sentence Transformers)", "Fast (TF-IDF)"],
                help="Advanced model provides better recommendations but may be slower",
            )
            use_advanced = use_embeddings.startswith("Advanced")
        else:
            st.info("Using TF-IDF model (embeddings not available)")
            use_advanced = False

        # Number of recommendations
        n_recommendations = st.slider(
            "Number of Recommendations",
            min_value=3,
            max_value=20,
            value=7,
            step=1,
            help="How many similar titles to show",
        )

        # Quick filters
        st.markdown("---")
        st.subheader("🔍 Quick Filters")
        content_type = st.multiselect(
            "Content Type", options=["Movie", "TV Show"], default=["Movie", "TV Show"]
        )

        # Filter dataframe
        if content_type:
            df_filtered = df[df["type"].isin(content_type)]
        else:
            df_filtered = df

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📈 Explore Data", "ℹ️ About"])

    with tab1:
        # Search section
        col1, col2 = st.columns([3, 1])

        with col1:
            search_query = st.text_input(
                "Search for a title",
                placeholder="Start typing to search...",
                help="Type at least 3 characters to search",
                key="search_input",
            )

        # Filter titles based on search and content type
        if search_query and len(search_query) >= 3:
            matching_titles = df_filtered[
                df_filtered["title"].str.contains(search_query, case=False, na=False)
            ]["title"].tolist()
        else:
            matching_titles = df_filtered["title"].tolist()

        if not matching_titles:
            st.warning("No titles found matching your search criteria.")
            return

        # Title selection
        selected_title = st.selectbox(
            "Select a title",
            options=sorted(matching_titles),
            help="Choose a title to get recommendations",
        )

        # Get recommendations button
        if st.button("🎬 Get Recommendations", type="primary", width="stretch"):
            if selected_title:
                with st.spinner("🔍 Finding similar content..."):
                    # Select vectors
                    vectors = (
                        embeddings
                        if use_advanced and embeddings is not None
                        else X_reduced
                    )

                    # Get recommendations
                    recommendations, message = recommend_enhanced(
                        selected_title, df, vectors, n_recommendations
                    )

                    if not recommendations.empty:
                        st.success(f"✅ Found {len(recommendations)} recommendations!")

                        # Display selected title info
                        st.markdown("---")
                        st.subheader("📺 Selected Title")
                        selected_info = df[df["title"] == selected_title].iloc[0]

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Type", selected_info["type"])
                        with col2:
                            st.metric("Rating", selected_info["rating"])
                        with col3:
                            st.metric("Year", selected_info["release_year"])
                        with col4:
                            st.metric("Cluster", selected_info.get("cluster", "N/A"))

                        with st.expander("📖 View Description", expanded=True):
                            st.write(f"**Description:** {selected_info['description']}")
                            st.write(f"**Genres:** {selected_info['listed_in']}")
                            if not pd.isna(selected_info["director"]):
                                st.write(f"**Director:** {selected_info['director']}")
                            if not pd.isna(selected_info["cast"]):
                                st.write(f"**Cast:** {selected_info['cast'][:200]}...")

                        st.markdown("---")

                        # Display recommendations
                        st.subheader("🎯 Recommended for You")

                        # Create a bar chart of similarity scores
                        fig = px.bar(
                            recommendations,
                            x="title",
                            y="similarity_score",
                            title="Recommendation Similarity Scores",
                            labels={"similarity_score": "Similarity", "title": "Title"},
                            color="similarity_score",
                            color_continuous_scale="Reds",
                        )
                        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
                        st.plotly_chart(fig, width="stretch")

                        # Display each recommendation
                        for idx, row in recommendations.iterrows():
                            with st.expander(
                                f"🎬 {row['title']} ({row['type']}) - {row['similarity_score']:.1%} match",
                                expanded=False,
                            ):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**Genres:** {row['listed_in']}")
                                    st.write(f"**Description:** {row['description']}")
                                with col2:
                                    st.metric("Rating", row["rating"])
                                    st.metric("Year", row["release_year"])
                    else:
                        st.error(message)
            else:
                st.warning("Please select a title first.")

    with tab2:
        st.subheader("📊 Dataset Exploration")

        # Dataset statistics at the top
        st.markdown("")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Total Titles", f"{len(df):,}")
        with col2:
            st.metric("🎬 Movies", f"{len(df[df['type'] == 'Movie']):,}")
        with col3:
            st.metric("📺 TV Shows", f"{len(df[df['type'] == 'TV Show']):,}")
        with col4:
            st.metric("🌍 Countries", df["country"].nunique())

        st.markdown("---")

        # Content type distribution
        col1, col2 = st.columns(2)

        with col1:
            fig_pie = px.pie(
                df,
                names="type",
                title="Content Type Distribution",
                color_discrete_sequence=["#E50914", "#221f1f"],
            )
            st.plotly_chart(fig_pie, width="stretch")

        with col2:
            # Top countries
            top_countries = df["country"].value_counts().head(10)
            fig_bar = px.bar(
                x=top_countries.index,
                y=top_countries.values,
                title="Top 10 Countries",
                labels={"x": "Country", "y": "Number of Titles"},
                color=top_countries.values,
                color_continuous_scale="Reds",
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, width="stretch")

        # Rating distribution
        rating_counts = df["rating"].value_counts()
        fig_rating = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Content Rating Distribution",
            labels={"x": "Rating", "y": "Count"},
            color=rating_counts.values,
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig_rating, width="stretch")

        # Cluster distribution if available
        if "cluster" in df.columns:
            st.subheader("Cluster Analysis")
            cluster_counts = df["cluster"].value_counts().sort_index()
            fig_cluster = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                title="Title Distribution Across Clusters",
                labels={"x": "Cluster ID", "y": "Number of Titles"},
                color=cluster_counts.values,
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_cluster, width="stretch")

    with tab3:
        st.subheader("About This Project")

        st.markdown(
            """
        ### 🎯 Problem Statement
        Netflix hosts over 7,000 movies and TV shows, creating **choice paralysis** for users. 
        Traditional genre-based categorization fails to capture semantic nuances between content 
        from different cultures and contexts.
        
        ### 🔬 Solution
        This recommendation system uses **unsupervised machine learning** to:
        - Analyze textual metadata (descriptions, cast, directors, genres)
        - Create semantic clusters of similar content
        - Recommend titles based on content similarity
        - Operate **without user ratings or watch history**
        
        ### 🛠️ Technology Stack
        - **NLP**: TF-IDF, Sentence Transformers (all-MiniLM-L6-v2)
        - **Clustering**: K-Means, HDBSCAN, Leiden Algorithm
        - **Dimensionality Reduction**: TruncatedSVD
        - **Metrics**: Cosine Similarity, Silhouette Score, Davies-Bouldin Index
        - **Web Framework**: Streamlit
        
        ### 📈 Key Features
        - ✅ Privacy-compliant (no user data required)
        - ✅ Fast recommendations (< 1 second)
        - ✅ Scalable to large catalogs
        - ✅ Multiple clustering algorithms
        - ✅ Domain-independent framework
        
        ### 👨‍💻 Developer
        **Internship Project** | Data Science & Machine Learning
        
        ### 📚 References
        - Sentence-BERT: Reimers & Gurevych (2019)
        - HDBSCAN: Campello et al. (2013)
        - Leiden Algorithm: Traag et al. (2019)
        """
        )


if __name__ == "__main__":
    main()
