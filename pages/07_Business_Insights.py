import streamlit as st
from utils.data_loader import load_churn_data
from utils.visuals import load_css

st.set_page_config(page_title='Business Insights | Customer Retention Intelligence', page_icon='💡', layout='wide')
load_css('assets/style.css')

st.title('Business Insights')
st.markdown('Actionable recommendations and customer retention strategies derived from churn analytics.')

df = load_churn_data()

churn_by_geo = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False)
churn_by_products = df.groupby('NumOfProducts')['Exited'].mean().sort_values(ascending=False)
churn_by_active = df.groupby('IsActiveMember')['Exited'].mean().sort_values(ascending=False)

st.subheader('High-risk customer segments')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Top churn geography', churn_by_geo.index[0])
    st.write(f"{churn_by_geo.iloc[0]*100:.1f}% churn rate")
with col2:
    st.metric('Customers with most churn', f'{int(churn_by_products.index[0])} products')
    st.write(f"{churn_by_products.iloc[0]*100:.1f}% churn rate")
with col3:
    st.metric('Inactive members churn rate', f"{churn_by_active.iloc[0]*100:.1f}%")
    st.write('Inactive customers are significantly more likely to churn.')

st.markdown('---')
st.header('Why customers churn')
st.write(
    '''
- Customers with low product engagement and inactive accounts are most vulnerable.
- High balance customers who are not active may leave due to poor communication.
- Specific geographies show higher churn, suggesting localized churn campaigns.
    '''
)

st.header('Retention strategy recommendations')
st.markdown(
    '''
- **Target high-risk geographies** with personalized offers and dedicated service teams.
- **Engage inactive members** through product education, loyalty campaigns, and proactive service outreach.
- **Monitor balance-heavy customers** closely and create premium relationship management touchpoints.
- **Use prediction scores** to segment customers into low, medium, and high risk, then deploy tiered retention interventions.
    '''
)

with st.expander('Strategic recommendation cards', expanded=True):
    st.markdown(
        '<div class="metric-card"><h4>Retention Offer Design</h4><p>Use churn probability to deliver offers only to customers who are both high-risk and high-value.</p></div>'
        '<div class="metric-card"><h4>Cross-sell and loyalty</h4><p>Offer additional products to customers with low churn probability but high balance to deepen loyalty.</p></div>'
        '<div class="metric-card"><h4>Operational alignment</h4><p>Align marketing, customer success, and risk teams around the same churn risk signals for real-time action.</p></div>',
        unsafe_allow_html=True
    )
