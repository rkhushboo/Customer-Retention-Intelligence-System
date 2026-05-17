import streamlit as st
from utils.visuals import load_css

st.set_page_config(page_title='About Project | Customer Retention Intelligence', page_icon='📌', layout='wide')
load_css('assets/style.css')

st.title('About the Project')
st.markdown('A modern retention intelligence system built to support banking teams with churn prediction, model explainability, and actionable outcomes.')

st.subheader('Project overview')
st.write(
    'This application combines exploratory data analysis, machine learning, and neural network modeling to help banks identify customers at risk of leaving. ' 
    'It is designed for deployment on Streamlit Cloud and includes polished UI, model comparison, and customer-level prediction reporting.'
)

st.subheader('Technologies used')
st.write(
    '''
- **Streamlit** for web app delivery and interactive dashboard design.
- **Pandas** and **NumPy** for data engineering.
- **Plotly** for production-quality charts.
- **scikit-learn** for model training and prediction pipelines.
- **TensorFlow / Keras** for ANN modeling and training.
    '''
)

st.subheader('ML algorithms')
st.write(
    '''
- **LightGBM / XGBoost** (as candidate traditional ML classifiers)
- **Random Forest** and **Gradient Boosting** in analysis explorations
- **Artificial Neural Network** with dropout and early stopping
    '''
)

st.subheader('Dataset source')
st.write('The dataset is sourced from standard bank customer churn analytics data and includes customer profile, account, and product usage features.')

st.subheader('Future improvements')
st.write(
    '''
- Add **real-time monitoring** and live score updating from production data.
- Extend the ANN with **customer embedding layers** for richer behavior capture.
- Add **A/B test management** for retention campaigns based on model segments.
- Integrate **Slack / email alerts** for high-risk customer triggers.
    '''
)
