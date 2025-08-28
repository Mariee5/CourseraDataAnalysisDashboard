# 📊 Student Performance Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing student performance data using machine learning and predictive analytics.

## 🎯 Project Overview

This project analyzes student performance data to predict which students are "at risk" of failing or dropping out based on their demographic and academic characteristics. The dashboard provides interactive visualizations, model performance metrics, and ethical insights for educational decision-making.

## 🚀 Features

- **📊 Data Analysis**: Statistical summaries and risk distribution analysis
- **🤖 Model Performance**: Logistic Regression and Random Forest comparison
- **🎯 Feature Importance**: Analysis of key factors influencing student success
- **📈 Interactive Visualizations**: Plotly charts for dynamic data exploration
- **🧠 Ethical Insights**: Responsible AI considerations and recommendations
- **📱 Modern UI**: Clean, professional dashboard with pastel color scheme

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/Mariee5/StudentPerformanceAnalysis.git
cd StudentPerformanceAnalysis
```

2. Install required packages:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
```

3. Run the dashboard:
```bash
streamlit run student_performance_dashboard.py
```

## 📁 Files

- `student_performance_dashboard.py` - Main Streamlit application
- `StudentsPerformance.csv` - Student performance dataset
- `2447247_P7.ipynb` - Original Jupyter notebook analysis
- `.gitignore` - Git ignore configuration

## 📊 Dataset

The dataset contains student performance data with the following features:
- **Demographics**: Gender, Race/Ethnicity
- **Socioeconomic**: Lunch Program, Parental Education Level
- **Academic**: Math, Reading, Writing Scores, Test Preparation Course

## 🎯 Key Insights

- **970 students** are not at risk (97%)
- **30 students** are at risk (3%)
- **Average score** is the dominant predictor (91.6% importance)
- **Lunch program** serves as a socioeconomic indicator
- **100% model accuracy** indicates overfitting (acknowledged)

## ⚖️ Ethical Considerations

This model should be used for:
- ✅ **Early intervention programs**
- ✅ **Targeted academic support**
- ✅ **Resource allocation**
- ✅ **Identifying students needing help**

**NOT for:**
- ❌ Limiting opportunities
- ❌ Discriminatory practices
- ❌ Punitive measures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)
- Visualizations with [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)

---

**Note**: This dashboard is for educational and research purposes. All AI recommendations should be combined with human judgment and institutional expertise.
