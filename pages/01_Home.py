import streamlit as st
from utils.data_loader import load_churn_data
from utils.model_utils import load_preprocessor, load_ml_model, load_ann_model
from utils.visuals import load_css, render_metric_cards

st.set_page_config(page_title='Home | Customer Retention Intelligence', page_icon='🏠', layout='wide')
load_css('assets/style.css')

st.title('Customer Retention Intelligence System')
st.markdown('## Business-facing churn analytics for banking teams')

with st.container():
    df = load_churn_data()
    total_customers = len(df)
    churn_rate = df['Exited'].mean()
    retention_rate = 1 - churn_rate

    model_accuracy = 'Pending model build'
    ann_accuracy = 'Pending model build'
    ml_name = 'Best ML Model'
    try:
        preprocessor = load_preprocessor()
        ml_model = load_ml_model()
        ann_model = load_ann_model()
        X = preprocessor.transform(df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited']))
        y = df['Exited']
        ml_acc = ml_model.score(X, y)
        ann_acc = ann_model.evaluate(X, y, verbose=0)[1]
        model_accuracy = f'{ml_acc*100:.1f}%'
        ann_accuracy = f'{ann_acc*100:.1f}%'
        ml_name = ml_model.__class__.__name__
    except Exception:
        pass

    cards = [
        {'label': 'Total Customers', 'value': f'{total_customers:,}', 'description': 'Bank customers analyzed in the dataset'},
        {'label': 'Churn Rate', 'value': f'{churn_rate*100:.1f}%', 'description': 'Percentage of customers who left'},
        {'label': 'Retention Rate', 'value': f'{retention_rate*100:.1f}%', 'description': 'Percentage of customers retained'},
        {'label': ml_name, 'value': model_accuracy, 'description': 'Latest model accuracy on the full dataset'},
        {'label': 'ANN Accuracy', 'value': ann_accuracy, 'description': 'Deep learning model performance estimate'},
    ]
    render_metric_cards(cards)

st.markdown('---')
st.header('Why proactive customer retention matters')
st.markdown(
    'Financial institutions can improve lifetime value, reduce acquisition costs, and increase customer satisfaction ' 
    'by understanding churn patterns. This application is built to surface those signals and turn them into action.'
)

st.markdown('---')
col1, col2 = st.columns([1.4, 1])
with col1:
    st.subheader('Project highlights')
    st.markdown(
        '''
- **Interactive prediction studio** for live customer scenario testing.
- **Model comparison** with ML and ANN outputs side by side.
- **Business insights** that translate churn risk to retention strategy.
- **Modern dashboard styling** with deploy-ready architecture.
        '''
    )
with col2:
    st.subheader('Project workflow')
    st.image('https://images.unsplash.com/photo-1522202176988-66273c2fd55f?auto=format&fit=crop&w=900&q=80', caption='Analytics workflow and business intelligence', use_column_width=True)
