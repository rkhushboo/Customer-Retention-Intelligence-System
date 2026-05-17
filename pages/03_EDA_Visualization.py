import streamlit as st
from utils.data_loader import load_churn_data
from utils.visuals import load_css, plot_churn_distribution, plotly_histogram, plotly_bar, plot_correlation_heatmap

st.set_page_config(page_title='EDA & Visualization | Customer Retention Intelligence', page_icon='📈', layout='wide')
load_css('assets/style.css')

st.title('EDA & Visualization')
st.markdown('Explore interactive charts that reveal customer churn momentum, demographic patterns, and feature relationships.')

df = load_churn_data()

with st.container():
    st.plotly_chart(plot_churn_distribution(df), use_container_width=True)

with st.expander('Apply dynamic filters', expanded=True):
    geo_filter = st.multiselect('Select Geography', options=df['Geography'].unique(), default=list(df['Geography'].unique()))
    gender_filter = st.multiselect('Select Gender', options=df['Gender'].unique(), default=list(df['Gender'].unique()))
    filtered = df[df['Geography'].isin(geo_filter) & df['Gender'].isin(gender_filter)]
    st.write(f'### Showing {filtered.shape[0]} records for selected filter')
    st.dataframe(filtered[['Geography', 'Gender', 'Age', 'Balance', 'Exited']].head(100), use_container_width=True)

with st.tabs(['Distributions', 'Comparisons', 'Correlation']):
    with st.tab('Distributions'):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plotly_histogram(filtered, x='Age', color='Exited', title='Age Distribution by Churn'), use_container_width=True)
        with col2:
            st.plotly_chart(plotly_histogram(filtered, x='Balance', color='Exited', title='Balance Distribution by Churn'), use_container_width=True)
        st.plotly_chart(plotly_histogram(filtered, x='CreditScore', color='Exited', title='Credit Score Distribution'), use_container_width=True)

    with st.tab('Comparisons'):
        st.plotly_chart(plotly_bar(filtered.groupby('Geography')['Exited'].mean().reset_index().assign(ChurnRate=lambda x: x['Exited']*100), x='Geography', y='ChurnRate', title='Churn Rate by Geography'), use_container_width=True)
        st.plotly_chart(plotly_bar(filtered.groupby('Gender')['Exited'].mean().reset_index().assign(ChurnRate=lambda x: x['Exited']*100), x='Gender', y='ChurnRate', title='Churn Rate by Gender'), use_container_width=True)
        st.plotly_chart(plotly_bar(filtered.groupby('Tenure')['Exited'].mean().reset_index().assign(ChurnRate=lambda x: x['Exited']*100), x='Tenure', y='ChurnRate', title='Churn Rate by Tenure'), use_container_width=True)

    with st.tab('Correlation'):
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

st.markdown('---')
st.subheader('Insights from the charts')
st.write(
    '''
- Customers in **France** and **Germany** display stronger churn patterns, suggesting geography-specific retention outreach.
- **Higher balances** often correlate with churn risk, so premium account service may improve retention.
- **Longer tenure** continues to be one of the strongest signals for loyalty.
    '''
)
