import streamlit as st
from utils.data_loader import load_churn_data
from utils.visuals import load_css

st.set_page_config(page_title='Dataset Overview | Customer Retention Intelligence', page_icon='📊', layout='wide')
load_css('assets/style.css')

df = load_churn_data()

st.title('Dataset Overview')
st.markdown('Understand the structure of the churn dataset, inspect feature meaning, and confirm quality before modeling.')

col1, col2 = st.columns(2)
with col1:
    st.metric('Rows', df.shape[0])
    st.metric('Columns', df.shape[1])
    st.metric('Churn Rate', f"{df['Exited'].mean()*100:.1f}%")
with col2:
    missing = df.isna().sum().sum()
    st.metric('Missing Values', f'{missing}')
    st.metric('Retained Rate', f"{(1 - df['Exited'].mean())*100:.1f}%")
    st.metric('Unique Geography', df['Geography'].nunique())

with st.expander('Dataset sample and filter controls', expanded=True):
    query = st.text_input('Search feature values', value='')
    if query:
        filtered = df[df.apply(lambda row: row.astype(str).str.contains(query, case=False, na=False).any(), axis=1)]
    else:
        filtered = df
    st.dataframe(filtered.head(100), use_container_width=True)

with st.expander('Feature list and data types', expanded=True):
    metadata = df.dtypes.reset_index()
    metadata.columns = ['Feature', 'Data Type']
    st.dataframe(metadata, use_container_width=True)

with st.expander('Statistical summary', expanded=False):
    st.dataframe(df.describe().T, use_container_width=True)

st.markdown('---')

st.header('Target variable and business context')
st.markdown(
    '**Exited** is the binary target label. Customers with `1` have churned, which represents leaving the bank. ' 
    'Our goal is to identify those customers early and preserve revenue through targeted retention actions.'
)

st.subheader('Business meaning of select features')
st.write(
    '''
- **CreditScore:** A strong predictor of credit risk and loyalty.
- **Geography:** Location-specific retention strategies can differ by country.
- **Gender:** Demographic signal that complements behavioral features.
- **Balance:** Customers with larger balances may require proactive service to stay.
- **NumOfProducts:** Product engagement often correlates with loyalty.
- **IsActiveMember:** Dormant customers are more likely to churn.
    '''
)
