"""
Coursera Course Success Analysis Dashboard

Team Contributions:
- Shobha Mary (2447247) - Domain Selection, Data Collection and Streamlit
- Yash Sharma (2447160) - Data Preprocessing and EDA
- Maria Bobby (2447130) - Modelling and Documentation
- Anupama Chakraborty (2447212) - Model Evaluation and Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Set page configuration
st.set_page_config(
    page_title="Coursera Course Success Analysis",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling (similar to student performance dashboard)
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .stMultiSelect > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2d3436, #636e72);
        color: white;
    }
    .stSidebar > div {
        background: linear-gradient(135deg, #2d3436, #636e72);
    }
    .stSidebar .stSelectbox label, 
    .stSidebar .stMultiSelect label,
    .stSidebar .stSlider label {
        color: white !important;
        font-weight: bold;
    }
    .stSidebar .stMarkdown {
        color: white;
    }
    h1, h2, h3 {
        color: #2d3436;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f3f4;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data with fallback options"""
    try:
        # Try cleaned data first
        if os.path.exists("cleaned_coursera_data.csv"):
            df = pd.read_csv("cleaned_coursera_data.csv")
            return df
        # Fallback to original data
        elif os.path.exists("Coursera.csv"):
            df = pd.read_csv("Coursera.csv")
            # Basic cleaning for original data
            df = df.dropna()
            if 'rating' in df.columns:
                df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            if 'reviewcount' in df.columns:
                df['reviewcount'] = df['reviewcount'].astype(str).str.replace('k', '000').str.replace(',', '')
                df['reviewcount'] = pd.to_numeric(df['reviewcount'], errors='coerce')
            return df
        else:
            # Create sample data if no files exist (for demo purposes)
            st.warning("No data file found. Using sample data for demonstration.")
            return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Using sample data for demonstration.")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    sample_data = {
        'course': [f'Course {i}' for i in range(100)],
        'partner': np.random.choice(['Google', 'IBM', 'University', 'Microsoft'], 100),
        'level': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], 100),
        'rating': np.random.uniform(3.0, 5.0, 100),
        'reviewcount': np.random.randint(10, 5000, 100),
        'partner_category': np.random.choice(['Big Tech', 'University', 'Other'], 100)
    }
    df = pd.DataFrame(sample_data)
    df['popularity_score'] = (df['rating'] / 5.0) * 0.7 + (df['reviewcount'] / df['reviewcount'].max()) * 0.3
    return df

# Load data
df = load_data()

# Always proceed with the app, even if using sample data
if df is not None and len(df) > 0:
    # Header with gradient background
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">
            Coursera Course Success Analysis
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Analyze key factors driving online course success on Coursera
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar filters with enhanced styling
    st.sidebar.markdown("""
    <h2 style="color: white; text-align: center; margin-bottom: 2rem;">Filters</h2>
    """, unsafe_allow_html=True)

    # Get available columns safely
    available_partners = df.get("partner_category", df.get("partner", pd.Series(["All"]))).dropna().unique()
    available_levels = df.get("level", pd.Series(["All"])).dropna().unique()

    partner_filter = st.sidebar.multiselect(
        "Partner Category", 
        options=available_partners, 
        default=available_partners
    )
    
    level_filter = st.sidebar.multiselect(
        "Course Level", 
        options=available_levels, 
        default=available_levels
    )

    # Apply filters
    if 'partner_category' in df.columns:
        filtered_df = df[df["partner_category"].isin(partner_filter)]
    elif 'partner' in df.columns:
        filtered_df = df[df["partner"].isin(partner_filter)]
    else:
        filtered_df = df

    if 'level' in df.columns and level_filter:
        filtered_df = filtered_df[filtered_df["level"].isin(level_filter)]

    # Overview Section
    st.markdown("## Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Courses</h3>
            <h2>{len(filtered_df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        partners_count = filtered_df['partner'].nunique() if 'partner' in filtered_df else 'N/A'
        st.markdown(f"""
        <div class="metric-card">
            <h3>Partners</h3>
            <h2>{partners_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        levels_count = filtered_df['level'].nunique() if 'level' in filtered_df else 'N/A'
        st.markdown(f"""
        <div class="metric-card">
            <h3>Levels</h3>
            <h2>{levels_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = f"{filtered_df['rating'].mean():.2f}" if 'rating' in filtered_df and not filtered_df['rating'].empty else 'N/A'
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Rating</h3>
            <h2>{avg_rating}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Sample Data Expander
    with st.expander("Sample Data", expanded=False):
        st.dataframe(filtered_df.head(10), use_container_width=True)

    # Key Statistics
    st.markdown("## Key Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'reviewcount' in filtered_df:
            avg_reviews = filtered_df['reviewcount'].mean()
            st.metric("Avg Reviews", f"{avg_reviews:.0f}")
        else:
            st.metric("Avg Reviews", "N/A")
    
    with col2:
        if 'popularity_score' in filtered_df:
            avg_popularity = filtered_df['popularity_score'].mean()
            st.metric("Avg Popularity", f"{avg_popularity:.2f}")
        else:
            st.metric("Avg Popularity", "N/A")
    
    with col3:
        if 'skills_count' in filtered_df:
            avg_skills = filtered_df['skills_count'].mean()
            st.metric("Avg Skills", f"{avg_skills:.1f}")
        else:
            st.metric("Avg Skills", "N/A")

    # Interactive Visualizations
    st.markdown("## Interactive Visualizations")
    
    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
        "Ratings", "Reviews", "Partners", "Correlations", "Clustering"
    ])

    with viz_tab1:
        st.markdown("### Course Ratings Distribution")
        if 'rating' in filtered_df and not filtered_df['rating'].empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(filtered_df['rating'].dropna(), bins=20, color='#74b9ff', alpha=0.7, edgecolor='black')
            ax.set_title('Distribution of Course Ratings', fontsize=16, fontweight='bold')
            ax.set_xlabel('Rating', fontsize=12)
            ax.set_ylabel('Number of Courses', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Rating data not available")

    with viz_tab2:
        st.markdown("### Review Count Distribution")
        if 'reviewcount' in filtered_df and not filtered_df['reviewcount'].empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(filtered_df['reviewcount'].dropna(), bins=20, color='#fd79a8', alpha=0.7, edgecolor='black')
            ax.set_title('Distribution of Review Counts', fontsize=16, fontweight='bold')
            ax.set_xlabel('Number of Reviews', fontsize=12)
            ax.set_ylabel('Number of Courses', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Review count data not available")

    with viz_tab3:
        st.markdown("### Courses by Partner Category")
        partner_col = 'partner_category' if 'partner_category' in filtered_df else 'partner'
        if partner_col in filtered_df:
            partner_counts = filtered_df[partner_col].value_counts()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(partner_counts.index, partner_counts.values, color='#55a3ff', alpha=0.8)
            ax.set_title('Number of Courses by Partner Category', fontsize=16, fontweight='bold')
            ax.set_xlabel('Partner Category', fontsize=12)
            ax.set_ylabel('Number of Courses', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Partner data not available")

    with viz_tab4:
        st.markdown("### Correlation Analysis")
        num_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            corr = filtered_df[num_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax)
            ax.set_title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Strong correlations table
            st.markdown("#### Strong Correlations (|r| > 0.5)")
            strong_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_val = corr.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_corr.append({
                            'Feature 1': corr.columns[i],
                            'Feature 2': corr.columns[j],
                            'Correlation': f"{corr_val:.3f}"
                        })
            
            if strong_corr:
                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
            else:
                st.info("No strong correlations found (|r| > 0.5)")
        else:
            st.info("Not enough numerical columns for correlation analysis")

    with viz_tab5:
        st.markdown("### Course Clustering Analysis")
        cluster_features = [col for col in ["rating", "reviewcount", "popularity_score"] if col in filtered_df]
        
        if len(cluster_features) >= 2:
            cluster_data = filtered_df[cluster_features].dropna()
            
            if len(cluster_data) > 10:  # Need sufficient data points
                # Standardize features
                scaler = StandardScaler()
                scaled = scaler.fit_transform(cluster_data)
                
                # Perform K-means clustering
                n_clusters = st.slider("Number of Clusters", 2, 6, 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled)
                
                # Add cluster labels to data
                cluster_data_viz = cluster_data.copy()
                cluster_data_viz["Cluster"] = cluster_labels
                
                # Create scatter plot
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
                
                for i in range(n_clusters):
                    mask = cluster_labels == i
                    ax.scatter(cluster_data.iloc[mask, 0], cluster_data.iloc[mask, 1], 
                             c=[colors[i]], label=f'Cluster {i+1}', alpha=0.7, s=60)
                
                ax.set_xlabel(cluster_features[0], fontsize=12)
                ax.set_ylabel(cluster_features[1], fontsize=12)
                ax.set_title(f'Course Clusters ({n_clusters} clusters)', fontsize=16, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Cluster summary
                cluster_summary = cluster_data_viz.groupby("Cluster")[cluster_features].mean().round(2)
                st.markdown("#### Cluster Characteristics")
                st.dataframe(cluster_summary, use_container_width=True)
            else:
                st.info("Not enough data points for clustering analysis")
        else:
            st.info("Not enough features for clustering analysis")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); 
                padding: 1.5rem; border-radius: 15px; text-align: center; margin-top: 2rem;">
        <h3 style="color: white; margin: 0;">Team Members</h3>
        <p style="color: white; margin: 0.5rem 0;">
            <strong>Shobha Mary (2447247)</strong> - Domain Selection, Data Collection and Streamlit<br>
            <strong>Yash Sharma (2447160)</strong> - Data Preprocessing and EDA<br>
            <strong>Maria Bobby (2447130)</strong> - Modelling and Documentation<br>
            <strong>Anupama Chakraborty (2447212)</strong> - Model Evaluation and Visualizations
        </p>
        <p style="color: white; margin: 0.5rem 0 0 0; font-style: italic;">
            Domain: Online Education / E-Learning Analytics | Data Source: Coursera Course Catalog
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Unable to load any data. Please check the deployment configuration.")
    st.stop()
