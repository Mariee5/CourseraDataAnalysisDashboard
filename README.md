# 📚 Coursera Course Success Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing course success factors on Coursera using machine learning and data analytics.

## 🎯 Project Overview

This project analyzes Coursera course data to identify key factors that drive course success, including ratings, review counts, partner categories, and skill requirements. The dashboard provides interactive visualizations and clustering analysis to help understand what makes courses successful on the platform.

## 👥 Team Members

- **Shobha Mary (2447247)** - Domain Selection, Data Collection and Streamlit
- **Yash Sharma (2447160)** - Data Preprocessing and EDA  
- **Maria Bobby (2447130)** - Modelling and Documentation
- **Anupama Chakraborty (2447212)** - Model Evaluation and Visualizations

## 🚀 Features

- **📊 Interactive Dashboard**: Modern Streamlit interface with gradient styling
- **🎛️ Dynamic Filtering**: Filter courses by partner category and level
- **📈 Data Visualizations**: Comprehensive charts for ratings, reviews, and partnerships
- **🔗 Correlation Analysis**: Identify relationships between course features
- **🎯 K-Means Clustering**: Segment courses based on performance metrics
- **📋 Statistical Insights**: Key performance indicators and summary statistics

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/Mariee5/CourseraDataAnalysisDashboard.git
cd CourseraDataAnalysisDashboard
```

2. Install required packages:
```bash
pip install -r coursera_requirements.txt
```

3. Run the dashboard:
```bash
streamlit run coursera_dashboard.py
```

## 📁 Files

- `coursera_dashboard.py` - Main Streamlit application
- `cleaned_coursera_data.csv` - Processed Coursera course dataset
- `Coursera.csv` - Original raw dataset (fallback)
- `coursera_requirements.txt` - Python dependencies
- `Team_04_Lab9&10_The_Insight_Quest.ipynb` - Original analysis notebook

## 📊 Dataset

The dataset contains Coursera course information with features including:
- **Course Details**: Title, Partner, Level, Duration
- **Performance Metrics**: Rating, Review Count, Popularity Score
- **Educational Content**: Skills, Certificate Type, Credit Eligibility
- **Categorizations**: Partner categories, Rating tiers, Duration groups

## 🎯 Key Insights

- **Course Success Factors**: Analyze what drives high ratings and engagement
- **Partner Analysis**: Compare performance across different educational partners
- **Skill Requirements**: Understand skill distribution and course complexity
- **Rating Patterns**: Identify trends in course ratings and reviews
- **Clustering Analysis**: Segment courses into distinct performance groups

## 📈 Dashboard Sections

1. **Dataset Overview**: Summary statistics and key metrics
2. **Interactive Filters**: Dynamic filtering by partner and level
3. **Visualizations**: 
   - Rating distribution analysis
   - Review count patterns
   - Partner category comparison
   - Correlation matrix
   - K-means clustering
4. **Performance Metrics**: Key indicators and trends

## 🎨 Design Features

- **Modern UI**: Clean, professional interface with gradient backgrounds
- **Responsive Layout**: Optimized for different screen sizes
- **Interactive Charts**: Matplotlib/Seaborn visualizations with custom styling
- **Color Scheme**: Professional blue gradient theme
- **User Experience**: Intuitive navigation and clear data presentation

## 🔧 Technical Details

- **Framework**: Streamlit for web application
- **Data Processing**: Pandas and NumPy for data manipulation
- **Visualization**: Matplotlib and Seaborn for charts
- **Machine Learning**: Scikit-learn for clustering analysis
- **Styling**: Custom CSS for modern appearance

## 🚀 Usage

1. Launch the dashboard using `streamlit run coursera_dashboard.py`
2. Use sidebar filters to explore specific course categories
3. Navigate through different visualization tabs
4. Adjust clustering parameters to explore course segments
5. Analyze correlation patterns and key insights

## 📋 Requirements

- Python 3.7+
- Streamlit 1.28.0+
- Pandas 2.0+
- NumPy 1.24+
- Matplotlib 3.7+
- Seaborn 0.12+
- Scikit-learn 1.3+

## 🎓 Domain

**Online Education / E-Learning Analytics**

This project focuses on understanding success patterns in online education, specifically analyzing factors that contribute to course popularity and student engagement on the Coursera platform.

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Data Source**: Coursera Course Catalog
- **Built with**: Streamlit, Pandas, Matplotlib, Scikit-learn
- **Inspired by**: Data-driven approaches to educational analytics

---

**Note**: This dashboard provides insights into course success patterns for educational and analytical purposes. Results should be interpreted in the context of the specific dataset and time period analyzed.
