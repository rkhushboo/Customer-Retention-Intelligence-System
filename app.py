import streamlit as st
from utils.visuals import load_css

st.set_page_config(
    page_title='Customer Retention Intelligence System',
    page_icon='💼',
    layout='wide',
    initial_sidebar_state='expanded'
)

load_css('assets/style.css')

st.markdown('<div class="hero-banner">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<h1>Customer Retention Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">A modern predictive dashboard for bank customer churn, blending business insight, interactive analysis and predictive confidence in one deployment-ready Streamlit application.</p>', unsafe_allow_html=True)
    st.markdown('<ul class="hero-bullets"><li>Explore dataset insights and model outcomes</li><li>Compare advanced ML and ANN predictions</li><li>Make real-time churn decisions with confidence</li></ul>', unsafe_allow_html=True)
    st.markdown('<a class="hero-cta" href="#prediction-studio">Go to Prediction Studio</a>', unsafe_allow_html=True)
with col2:
    st.image('https://images.unsplash.com/photo-1542744173-8e7e53415bb0?auto=format&fit=crop&w=900&q=80', caption='Intelligent customer retention with predictive analytics', use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('---')

st.markdown('## Navigate the application')
st.write('Use the sidebar to move between pages. Each page is designed for data exploration, model explanation, predictions, and business impact.')

with st.expander('What is inside this app?', expanded=True):
    st.markdown(
        '''
- **Home:** Landing experience with hero messaging and KPI highlights.
- **Dataset Overview:** Clean summary, search, filters and business feature context.
- **EDA & Visualization:** Interactive charts, filters, correlation analysis and story-driven insight.
- **Machine Learning Model:** Metrics, ROC, confusion matrix, feature importance and hyperparameter summary.
- **ANN Model:** Architecture breakdown, training curves, metrics and deep learning explanation.
- **Prediction Studio:** Customer input form, model comparison, risk meters and downloadable reports.
- **Business Insights:** Actionable recommendations for retention strategy.
- **About Project:** Tech stack, dataset source and future improvement plan.
        '''
    )
