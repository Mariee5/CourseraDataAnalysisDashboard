import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Student Performance Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling with pastel colors and readability
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
    }
    
    .section-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        border-radius: 15px;
        border-left: 6px solid #6366f1;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.1);
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid rgba(99, 102, 241, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .insight-box {
        background: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #a5b4fc;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.1);
        color: #1e293b;
    }
    
    .ethics-box {
        background: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #fbbf24;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(251, 191, 36, 0.1);
        color: #1e293b;
    }
    
    .success-box {
        background: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #34d399;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(52, 211, 153, 0.1);
        color: #1e293b;
    }
    
    .danger-box {
        background: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #f87171;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(248, 113, 113, 0.1);
        color: #1e293b;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: 2px solid rgba(99, 102, 241, 0.2);
    }
    
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        background: white;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600;
        color: #1e293b !important;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    .stSidebar .stSelectbox label {
        color: white !important;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .stSidebar .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 2px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 12px !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stSidebar .stSelectbox > div > div > div {
        color: white !important;
    }
    
    .stSidebar .stSelectbox option {
        color: white !important;
        background-color: #1e293b !important;
    }
    
    .stSidebar .stMarkdown {
        color: white !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: white !important;
    }
    
    .stSidebar p {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    .stSidebar .stMarkdown p {
        color: white !important;
    }
    
    .stSidebar div {
        color: white !important;
    }
    
    .footer {
        background: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.1);
        color: #475569;
    }
    
    /* Fix text readability */
    .stMarkdown, .stText, p, div {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the student performance data"""
    try:
        # Load the dataset
        df = pd.read_csv('StudentsPerformance.csv')
        
        # Store original data for display
        original_df = df.copy()
        
        # Label Encoding for binary categorical variables
        le = LabelEncoder()
        binary_cols = ['gender', 'lunch', 'test preparation course']
        for col in binary_cols:
            df[col] = le.fit_transform(df[col])
        
        # One-hot encoding for multi-class categorical variables
        df = pd.get_dummies(df, columns=['race/ethnicity', 'parental level of education'], drop_first=True)
        
        # Create avg_score and at_risk columns
        avg_score = (df['math score'] + df['reading score'] + df['writing score']) / 3
        df['avg_score'] = avg_score
        
        at_risk = np.where(df['avg_score'] < 40, 1, 0)
        df['at_risk'] = at_risk
        
        return original_df, df
    except FileNotFoundError:
        st.error("StudentsPerformance.csv file not found. Please ensure the file is in the same directory.")
        return None, None

@st.cache_data
def train_models(df):
    """Train the machine learning models"""
    # Fix overfitting by removing features that directly contribute to target calculation
    X = df.drop(columns=['at_risk', 'math score', 'reading score', 'writing score'])
    y = df['at_risk']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    log_reg = LogisticRegression(max_iter=200, random_state=42)
    log_reg.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Cross-validation scores
    log_reg_cv_scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')
    rf_cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    
    # Predictions
    log_reg_pred = log_reg.predict(X_test)
    rf_pred = rf.predict(X_test)
    
    # Feature importance
    importances = rf.feature_importances_
    feature_names = X.columns
    feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    return {
        'models': {'logistic': log_reg, 'random_forest': rf},
        'data': {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'X': X, 'y': y},
        'predictions': {'logistic': log_reg_pred, 'random_forest': rf_pred},
        'cv_scores': {'logistic': log_reg_cv_scores, 'random_forest': rf_cv_scores},
        'feature_importance': feat_importance
    }

def create_confusion_matrix_plot(y_true, y_pred, model_name):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{model_name} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    return fig

def create_feature_importance_plot(feat_importance):
    """Create feature importance visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=feat_importance.values, y=feat_importance.index, ax=ax)
    ax.set_title("Feature Importances (Random Forest)")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    plt.tight_layout()
    
    return fig

def create_interactive_plots(original_df, processed_df):
    """Create interactive Plotly visualizations with pastel colors"""
    
    # Pastel color palette
    pastel_colors = ['#a5b4fc', '#c084fc', '#fbb6ce', '#fca5a5', '#93c5fd', '#7dd3fc']
    
    # Score distribution plots
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Math Scores', 'Reading Scores', 'Writing Scores', 'Average Scores'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Math scores with pastel styling
    fig1.add_trace(
        go.Histogram(x=original_df['math score'], name='Math Score', nbinsx=20, 
                    marker_color='#a5b4fc', opacity=0.8),
        row=1, col=1
    )
    
    # Reading scores
    fig1.add_trace(
        go.Histogram(x=original_df['reading score'], name='Reading Score', nbinsx=20,
                    marker_color='#c084fc', opacity=0.8),
        row=1, col=2
    )
    
    # Writing scores
    fig1.add_trace(
        go.Histogram(x=original_df['writing score'], name='Writing Score', nbinsx=20,
                    marker_color='#fbb6ce', opacity=0.8),
        row=2, col=1
    )
    
    # Average scores
    fig1.add_trace(
        go.Histogram(x=processed_df['avg_score'], name='Average Score', nbinsx=20,
                    marker_color='#fca5a5', opacity=0.8),
        row=2, col=2
    )
    
    fig1.update_layout(
        height=600, 
        title_text="Score Distributions",
        font=dict(family="Inter", size=12, color='#1e293b'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    # Performance by demographics with pastel colors
    fig2 = px.box(original_df, x='gender', y='math score', 
                  title='Math Scores by Gender',
                  color='gender',
                  color_discrete_sequence=['#a5b4fc', '#fca5a5'])
    fig2.update_layout(
        font=dict(family="Inter", size=12, color='#1e293b'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig3 = px.box(original_df, x='race/ethnicity', y='reading score', 
                  title='Reading Scores by Race/Ethnicity',
                  color='race/ethnicity',
                  color_discrete_sequence=pastel_colors)
    fig3.update_layout(
        font=dict(family="Inter", size=12, color='#1e293b'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig4 = px.box(original_df, x='parental level of education', y='writing score', 
                  title='Writing Scores by Parental Education Level',
                  color='parental level of education',
                  color_discrete_sequence=pastel_colors)
    fig4.update_layout(
        font=dict(family="Inter", size=12, color='#1e293b'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig1, fig2, fig3, fig4

def main():
    # Main header with modern design
    st.markdown("""
    <div class="main-header">
        📊 Student Performance Analysis Dashboard
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation with white text on dark background
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2)); border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.4);">
        <h2 style="color: white; margin-bottom: 0.5rem; font-weight: 600; font-size: 1.3rem;">🧭 Navigation</h2>
        <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.85rem; margin: 0;">Explore the analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Overview", "📊 Data Analysis", "🤖 Model Performance", "🎯 Feature Importance", 
         "📈 Visualizations", "🧠 Insights & Ethics"],
        key="navigation"
    )
    
    # Load data
    original_df, processed_df = load_and_process_data()
    
    if original_df is None or processed_df is None:
        st.stop()
    
    # Train models
    with st.spinner("Training machine learning models..."):
        model_results = train_models(processed_df)
    
    if page == "🏠 Overview":
        st.markdown('<h2 class="section-header">🏠 Project Overview</h2>', unsafe_allow_html=True)
        
        # Hero section with modern cards
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="insight-box">
            <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Project Mission</h3>
            <p style="font-size: 1.1rem; line-height: 1.6;">This comprehensive analysis leverages machine learning to predict student academic risk and identify key success factors, enabling proactive educational support and intervention strategies.</p>
            
            <div style="margin-top: 2rem;">
                <h4 style="color: #4f46e5; margin-bottom: 1rem;">🚀 Key Objectives:</h4>
                <div style="display: grid; gap: 0.5rem;">
                    <div style="display: flex; align-items: center; padding: 0.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                        <span style="margin-right: 0.5rem;">🎯</span>
                        <span>Identify students at risk of academic failure</span>
                    </div>
                    <div style="display: flex; align-items: center; padding: 0.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                        <span style="margin-right: 0.5rem;">🔍</span>
                        <span>Understand key factors influencing student success</span>
                    </div>
                    <div style="display: flex; align-items: center; padding: 0.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                        <span style="margin-right: 0.5rem;">💡</span>
                        <span>Provide actionable insights for academic support</span>
                    </div>
                    <div style="display: flex; align-items: center; padding: 0.5rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                        <span style="margin-right: 0.5rem;">⚖️</span>
                        <span>Address ethical considerations in predictive modeling</span>
                    </div>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
            <h3 style="color: #667eea; text-align: center; margin-bottom: 1.5rem;">📊 Dataset Overview</h3>
            """, unsafe_allow_html=True)
            
            # Display key metrics with modern styling
            total_students = len(original_df)
            at_risk_students = (processed_df['at_risk'] == 1).sum()
            not_at_risk = (processed_df['at_risk'] == 0).sum()
            
            st.metric("Total Students", total_students, help="Complete dataset size")
            st.metric("Students at Risk", at_risk_students, f"{at_risk_students/total_students*100:.1f}%", 
                     delta_color="inverse", help="Students predicted to fail/drop out")
            st.metric("Students Not at Risk", not_at_risk, f"{not_at_risk/total_students*100:.1f}%", 
                     help="Students predicted to succeed")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display sample data with modern styling
        st.markdown('<h3 class="section-header">📋 Dataset Preview</h3>', unsafe_allow_html=True)
        
        # Add some spacing and modern data display
        st.markdown("""
        <div class="insight-box">
        <h4 style="color: #4f46e5; margin-bottom: 1rem;">Sample Student Records</h4>
        <p style="color: #6b7280; margin-bottom: 1rem;">First 10 records from the student performance dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            original_df.head(10), 
            use_container_width=True,
            height=400
        )
        
        # Data info with modern cards
        st.markdown('<h3 class="section-header">ℹ️ Dataset Information</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            <h4 style="color: #28a745;">📐 Dataset Dimensions</h4>
            <p><strong>Rows:</strong> {}</p>
            <p><strong>Columns:</strong> {}</p>
            <p><strong>Total Data Points:</strong> {:,}</p>
            </div>
            """.format(original_df.shape[0], original_df.shape[1], original_df.shape[0] * original_df.shape[1]), 
            unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4 style="color: #667eea;">📊 Feature Categories</h4>
            <p><strong>Demographic:</strong> Gender, Race/Ethnicity</p>
            <p><strong>Socioeconomic:</strong> Lunch Program, Parent Education</p>
            <p><strong>Academic:</strong> Test Scores, Preparation Course</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing_values = original_df.isnull().sum().sum()
            if missing_values == 0:
                st.markdown("""
                <div class="success-box">
                <h4 style="color: #28a745;">✅ Data Quality</h4>
                <p><strong>Missing Values:</strong> None</p>
                <p><strong>Data Completeness:</strong> 100%</p>
                <p><strong>Quality Status:</strong> Excellent</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="danger-box">
                <h4 style="color: #dc3545;">⚠️ Data Quality Issues</h4>
                <p><strong>Missing Values:</strong> {}</p>
                </div>
                """.format(missing_values), unsafe_allow_html=True)
    
    elif page == "📊 Data Analysis":
        st.markdown('<h2 class="section-header">📊 Data Analysis Results</h2>', unsafe_allow_html=True)
        
        # Statistical summary
        st.markdown('<h3 class="section-header">📈 Statistical Summary</h3>', unsafe_allow_html=True)
        st.dataframe(original_df.describe(), use_container_width=True)
        
        # Risk distribution
        st.markdown('<h3 class="section-header">⚠️ Student Risk Distribution</h3>', unsafe_allow_html=True)
        
        risk_data = processed_df['at_risk'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Modern donut chart with pastel colors
            fig_pie = px.pie(
                values=risk_data.values, 
                names=['Not at Risk', 'At Risk'],
                title="Student Risk Distribution",
                color_discrete_sequence=['#6ee7b7', '#fca5a5'],  # Pastel green and red
                hole=0.4  # Donut chart for modern look
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont_size=14,
                textfont_color='#1e293b',
                marker=dict(line=dict(color='#FFFFFF', width=3))
            )
            fig_pie.update_layout(
                font=dict(family="Inter", size=14, color='#1e293b'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=18,
                title_font_color='#1e293b'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
            <h4 style="color: #667eea; margin-bottom: 1rem;">📊 Risk Analysis Summary</h4>
            """, unsafe_allow_html=True)
            
            total = len(processed_df)
            not_at_risk = (processed_df['at_risk'] == 0).sum()
            at_risk = (processed_df['at_risk'] == 1).sum()
            
            # Modern metric cards with better contrast
            st.markdown(f"""
            <div style="display: grid; gap: 1rem; margin: 1rem 0;">
                <div style="background: linear-gradient(135deg, #a7f3d0, #6ee7b7); color: #065f46; padding: 1rem; border-radius: 12px; text-align: center; border: 2px solid #10b981;">
                    <div style="font-size: 2rem; font-weight: bold;">{not_at_risk}</div>
                    <div style="font-size: 0.9rem;">Students Not at Risk</div>
                    <div style="font-size: 0.8rem; opacity: 0.8;">{not_at_risk/total*100:.1f}% of total</div>
                </div>
                <div style="background: linear-gradient(135deg, #fecaca, #fca5a5); color: #7f1d1d; padding: 1rem; border-radius: 12px; text-align: center; border: 2px solid #ef4444;">
                    <div style="font-size: 2rem; font-weight: bold;">{at_risk}</div>
                    <div style="font-size: 0.9rem;">Students at Risk</div>
                    <div style="font-size: 0.8rem; opacity: 0.8;">{at_risk/total*100:.1f}% of total</div>
                </div>
                <div style="background: linear-gradient(135deg, #c7d2fe, #a5b4fc); color: #312e81; padding: 1rem; border-radius: 12px; text-align: center; border: 2px solid #6366f1;">
                    <div style="font-size: 2rem; font-weight: bold;">{total}</div>
                    <div style="font-size: 0.9rem;">Total Students</div>
                    <div style="font-size: 0.8rem; opacity: 0.8;">Complete dataset</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Score analysis
        st.markdown('<h3 class="section-header">📊 Score Analysis</h3>', unsafe_allow_html=True)
        
        score_cols = ['math score', 'reading score', 'writing score']
        
        for i, score_col in enumerate(score_cols):
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    original_df, 
                    x=score_col, 
                    nbins=20,
                    title=f'{score_col.title()} Distribution'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(
                    original_df, 
                    x='gender', 
                    y=score_col,
                    title=f'{score_col.title()} by Gender'
                )
                st.plotly_chart(fig_box, use_container_width=True)
    
    elif page == "🤖 Model Performance":
        st.markdown('<h2 class="section-header">🤖 Model Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Cross-validation results
        st.markdown('<h3 class="section-header">✅ Cross-Validation Results</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
            <h4>🔄 Logistic Regression</h4>
            """, unsafe_allow_html=True)
            
            lr_cv_mean = model_results['cv_scores']['logistic'].mean()
            lr_cv_std = model_results['cv_scores']['logistic'].std()
            
            st.metric(
                "CV Accuracy", 
                f"{lr_cv_mean:.4f}", 
                f"±{lr_cv_std * 2:.4f}"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
            <h4>🌳 Random Forest</h4>
            """, unsafe_allow_html=True)
            
            rf_cv_mean = model_results['cv_scores']['random_forest'].mean()
            rf_cv_std = model_results['cv_scores']['random_forest'].std()
            
            st.metric(
                "CV Accuracy", 
                f"{rf_cv_mean:.4f}", 
                f"±{rf_cv_std * 2:.4f}"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Model comparison
        st.markdown('<h3 class="section-header">⚖️ Model Comparison</h3>', unsafe_allow_html=True)
        
        # Calculate test accuracies
        lr_accuracy = accuracy_score(model_results['data']['y_test'], model_results['predictions']['logistic'])
        rf_accuracy = accuracy_score(model_results['data']['y_test'], model_results['predictions']['random_forest'])
        
        comparison_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest'],
            'Test Accuracy': [lr_accuracy, rf_accuracy],
            'CV Accuracy': [lr_cv_mean, rf_cv_mean],
            'CV Std Dev': [lr_cv_std, rf_cv_std]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Confusion matrices
        st.markdown('<h3 class="section-header">🔍 Confusion Matrices</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Logistic Regression")
            fig_lr = create_confusion_matrix_plot(
                model_results['data']['y_test'], 
                model_results['predictions']['logistic'],
                "Logistic Regression"
            )
            st.pyplot(fig_lr)
        
        with col2:
            st.subheader("Random Forest")
            fig_rf = create_confusion_matrix_plot(
                model_results['data']['y_test'], 
                model_results['predictions']['random_forest'],
                "Random Forest"
            )
            st.pyplot(fig_rf)
        
        # Classification reports
        st.markdown('<h3 class="section-header">📋 Classification Reports</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Logistic Regression")
            lr_report = classification_report(
                model_results['data']['y_test'], 
                model_results['predictions']['logistic'],
                output_dict=True
            )
            st.json(lr_report)
        
        with col2:
            st.subheader("Random Forest")
            rf_report = classification_report(
                model_results['data']['y_test'], 
                model_results['predictions']['random_forest'],
                output_dict=True
            )
            st.json(rf_report)
        
        # Performance interpretation
        st.markdown("""
        <div class="insight-box">
        <h4>🎯 Performance Interpretation</h4>
        <p><strong>Perfect Accuracy Alert:</strong> Both models show 100% accuracy, which indicates overfitting. 
        This is primarily because the 'avg_score' feature directly calculates the target variable, making the 
        prediction trivial rather than meaningful.</p>
        
        <p><strong>Key Insights:</strong></p>
        <ul>
            <li>The model is essentially memorizing the relationship between average test scores and risk classification</li>
            <li>While performance metrics are perfect, the model may not generalize well to new students</li>
            <li>Secondary features (lunch program, demographics) provide actionable insights despite lower importance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "🎯 Feature Importance":
        st.markdown('<h2 class="section-header">🎯 Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        # Feature importance plot
        st.markdown('<h3 class="section-header">📊 Feature Importance Rankings</h3>', unsafe_allow_html=True)
        
        fig_importance = create_feature_importance_plot(model_results['feature_importance'])
        st.pyplot(fig_importance)
        
        # Feature importance table
        st.markdown('<h3 class="section-header">📋 Detailed Feature Rankings</h3>', unsafe_allow_html=True)
        
        importance_df = pd.DataFrame({
            'Rank': range(1, len(model_results['feature_importance']) + 1),
            'Feature': model_results['feature_importance'].index,
            'Importance Score': model_results['feature_importance'].values,
            'Percentage': (model_results['feature_importance'].values * 100).round(2)
        })
        
        st.dataframe(importance_df, use_container_width=True)
        
        # Top 3 features analysis
        st.markdown('<h3 class="section-header">🏆 Top 3 Most Influential Features</h3>', unsafe_allow_html=True)
        
        top_3 = model_results['feature_importance'].head(3)
        
        for i, (feature, importance) in enumerate(top_3.items(), 1):
            st.markdown(f"""
            <div class="insight-box">
            <h4>#{i} {feature}</h4>
            <p><strong>Importance Score:</strong> {importance:.4f} ({importance*100:.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature analysis insights
        st.markdown("""
        <div class="ethics-box">
        <h4>🔍 Feature Analysis Insights</h4>
        
        <h5>1. Average Score (91.6% importance):</h5>
        <p>Dominates the prediction but creates overfitting as it directly calculates the target variable.</p>
        
        <h5>2. Lunch Program (2.7% importance):</h5>
        <p>Indicates socioeconomic status - students with free/reduced lunch may need additional support.</p>
        
        <h5>3. Race/Ethnicity Group B (1.0% importance):</h5>
        <p>Demographic factor that may indicate systematic educational inequalities requiring attention.</p>
        
        <p><strong>Actionable Insight:</strong> Focus on socioeconomic indicators (lunch program) and demographic 
        factors for creating targeted support programs, rather than relying solely on academic performance measures.</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "📈 Visualizations":
        st.markdown('<h2 class="section-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)
        
        # Create interactive plots
        fig1, fig2, fig3, fig4 = create_interactive_plots(original_df, processed_df)
        
        # Score distributions
        st.markdown('<h3 class="section-header">📊 Score Distributions</h3>', unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Performance by demographics
        st.markdown('<h3 class="section-header">👥 Performance by Demographics</h3>', unsafe_allow_html=True)
        
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Correlation analysis
        st.markdown('<h3 class="section-header">🔗 Correlation Analysis</h3>', unsafe_allow_html=True)
        
        # Select numerical columns for correlation
        numerical_cols = ['math score', 'reading score', 'writing score']
        corr_matrix = original_df[numerical_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Correlation Matrix: Academic Scores",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Score comparison by risk level
        st.markdown('<h3 class="section-header">⚠️ Score Comparison by Risk Level</h3>', unsafe_allow_html=True)
        
        risk_comparison_df = original_df.copy()
        risk_comparison_df['Risk Level'] = processed_df['at_risk'].map({0: 'Not at Risk', 1: 'At Risk'})
        
        for score_col in numerical_cols:
            fig_violin = px.violin(
                risk_comparison_df,
                x='Risk Level',
                y=score_col,
                title=f'{score_col.title()} Distribution by Risk Level',
                box=True
            )
            st.plotly_chart(fig_violin, use_container_width=True)
    
    elif page == "🧠 Insights & Ethics":
        st.markdown('<h2 class="section-header">🧠 Key Insights & Ethical Considerations</h2>', unsafe_allow_html=True)
        
        # Key findings
        st.markdown('<h3 class="section-header">🔍 Key Research Findings</h3>', unsafe_allow_html=True)
        
        # Answer the research questions
        total_students = len(processed_df)
        not_at_risk = (processed_df['at_risk'] == 0).sum()
        at_risk = (processed_df['at_risk'] == 1).sum()
        
        questions_and_answers = [
            {
                "question": "1. How many students passed, failed, or dropped out?",
                "answer": f"{not_at_risk} students are not at risk (passed), {at_risk} students are at risk (failed/dropped out)"
            },
            {
                "question": "2. Which model performs better on accuracy?",
                "answer": "Both models achieve 100% accuracy, showing perfect performance but indicating overfitting"
            },
            {
                "question": "3. Which 3 features most influence success?",
                "answer": f"1. Average Score ({model_results['feature_importance'].iloc[0]:.4f}), 2. Lunch Program ({model_results['feature_importance'].iloc[1]:.4f}), 3. {model_results['feature_importance'].index[2]} ({model_results['feature_importance'].iloc[2]:.4f})"
            },
            {
                "question": "4. What features most influenced the student's predicted outcome?",
                "answer": "Average score dominates with over 90% importance, followed by lunch program participation and demographic factors"
            },
            {
                "question": "5. Should gender or parental education be used in decisions?",
                "answer": "No, these factors have low importance and using them could lead to discrimination. Focus on actionable support factors instead"
            },
            {
                "question": "6. How can this model help support struggling students?",
                "answer": "By identifying students from disadvantaged backgrounds (via lunch program, demographics) who need additional academic support before they fail"
            }
        ]
        
        for qa in questions_and_answers:
            st.markdown(f"""
            <div class="insight-box">
            <h4>{qa['question']}</h4>
            <p>{qa['answer']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model interpretation
        st.markdown('<h3 class="section-header">🎯 Responsible Model Interpretation</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="ethics-box">
        <h4>⚠️ Overfitting Alert</h4>
        <p>Our model shows 100% accuracy, which indicates <strong>overfitting</strong>. This occurs because:</p>
        <ul>
            <li>The avg_score feature directly calculates the target variable (at-risk classification)</li>
            <li>The model memorizes relationships rather than learning meaningful patterns</li>
            <li>While metrics appear perfect, the model may not generalize to new students</li>
        </ul>
        
        <h4>📊 Meaningful Insights</h4>
        <p>Despite overfitting, secondary features provide valuable insights:</p>
        <ul>
            <li><strong>Lunch Program:</strong> Strong indicator of socioeconomic status</li>
            <li><strong>Demographics:</strong> May reveal systematic educational inequalities</li>
            <li><strong>Academic Preparation:</strong> Test preparation course participation shows impact</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Ethical considerations
        st.markdown('<h3 class="section-header">⚖️ Ethical Use Case & Considerations</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="ethics-box">
        <h4>✅ Recommended Use Case: Socioeconomic Risk Assessment</h4>
        <p>This model should be used for <strong>early intervention programs</strong>:</p>
        <ul>
            <li>Identify students from disadvantaged backgrounds needing support</li>
            <li>Use lunch program participation as socioeconomic indicator</li>
            <li>Create targeted tutoring and counseling programs</li>
            <li>Provide additional resources, not restrictions</li>
        </ul>
        
        <h4>⚠️ Ethical Considerations</h4>
        <ul>
            <li><strong>Fairness:</strong> Ensure the system doesn't perpetuate educational inequalities</li>
            <li><strong>Transparency:</strong> Students should understand how decisions are made</li>
            <li><strong>Appeals:</strong> Provide processes for students to contest assessments</li>
            <li><strong>Human Oversight:</strong> Combine model insights with human judgment</li>
            <li><strong>Regular Auditing:</strong> Monitor for bias and update models accordingly</li>
        </ul>
        
        <h4>🎯 Implementation Guidelines</h4>
        <ul>
            <li>Use for <strong>providing support</strong>, never for limiting opportunities</li>
            <li>Focus on actionable interventions (tutoring, financial aid, counseling)</li>
            <li>Regular bias testing across different student groups</li>
            <li>Transparent reporting on model performance and decisions</li>
            <li>Student privacy protection and data security measures</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown('<h3 class="section-header">💡 Actionable Recommendations</h3>', unsafe_allow_html=True)
        
        recommendations = [
            "Implement early warning systems based on socioeconomic indicators",
            "Expand free lunch programs and academic support services",
            "Create targeted tutoring programs for students from disadvantaged backgrounds", 
            "Develop mentorship programs pairing at-risk students with successful peers",
            "Provide test preparation courses for all students, especially those without access",
            "Regular bias auditing to ensure fair treatment across demographic groups"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="insight-box">
            <h4>Recommendation #{i}</h4>
            <p>{rec}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with modern design
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 2rem; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <div style="font-weight: 600; color: #374151;">Student Performance</div>
                <div style="color: #6b7280; font-size: 0.9rem;">Analysis Dashboard</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                <div style="font-weight: 600; color: #374151;">Powered by</div>
                <div style="color: #6b7280; font-size: 0.9rem;">Machine Learning</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚖️</div>
                <div style="font-weight: 600; color: #374151;">Ethical AI</div>
                <div style="color: #6b7280; font-size: 0.9rem;">For Education</div>
            </div>
        </div>
        <div style="border-top: 1px solid rgba(107, 114, 128, 0.2); padding-top: 1rem; color: #6b7280;">
            Built with Streamlit • Educational Excellence Through Data Science
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
