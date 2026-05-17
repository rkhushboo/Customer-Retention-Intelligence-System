import streamlit as st
from utils.data_loader import build_user_input
from utils.model_utils import load_preprocessor, load_ml_model, load_ann_model, predict_probability, format_prediction_result, generate_prediction_report
from utils.visuals import load_css, create_progress_card
from utils.config import GEOGRAPHY_OPTIONS, GENDER_MAP

st.set_page_config(page_title='Prediction Studio | Customer Retention Intelligence', page_icon='🧪', layout='wide')
load_css('assets/style.css')

st.title('Prediction Studio')
st.markdown('Enter a customer profile and compare churn predictions from the best traditional model and the best ANN.')

try:
    preprocessor = load_preprocessor()
    ml_model = load_ml_model()
    ann_model = load_ann_model()

    with st.form('prediction_form'):
        st.subheader('Customer profile inputs')
        c1, c2 = st.columns(2)
        user_values = {}
        with c1:
            user_values['CreditScore'] = st.slider('Credit Score', min_value=300, max_value=850, value=650, step=1)
            user_values['Geography'] = st.selectbox('Geography', GEOGRAPHY_OPTIONS)
            user_values['Gender'] = st.selectbox('Gender', list(GENDER_MAP.keys()))
            user_values['Age'] = st.number_input('Age', min_value=18, max_value=100, value=41, step=1)
            user_values['Tenure'] = st.slider('Tenure (years)', min_value=0, max_value=10, value=3, step=1)
        with c2:
            user_values['Balance'] = st.number_input('Balance', min_value=0.0, max_value=250000.0, value=72000.0, step=100.0, format='%.2f')
            user_values['NumOfProducts'] = st.selectbox('Number of Products', [1, 2, 3, 4])
            user_values['HasCrCard'] = st.radio('Has Credit Card', [1, 0], index=0, format_func=lambda x: 'Yes' if x == 1 else 'No')
            user_values['IsActiveMember'] = st.radio('Is Active Member', [1, 0], index=1, format_func=lambda x: 'Yes' if x == 1 else 'No')
            user_values['EstimatedSalary'] = st.number_input('Estimated Salary', min_value=10000.0, max_value=200000.0, value=100000.0, step=500.0, format='%.2f')

        submitted = st.form_submit_button('Predict Customer Churn')

    if submitted:
        payload = build_user_input(user_values)
        X = preprocessor.transform(payload)
        ml_proba = predict_probability(ml_model, X)[0]
        ann_proba = predict_probability(ann_model, X)[0]
        ml_result = format_prediction_result('Best ML Model', ml_proba)
        ann_result = format_prediction_result('Best ANN Model', ann_proba)

        st.markdown('### Prediction comparison')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-card'><h4>Traditional ML prediction</h4><p><strong>{'Churn' if ml_result['prediction'] else 'Stay'}</strong></p><p>Probability: {ml_result['probability']:.2%}</p><p>Risk: <span style='color:{ml_result['risk']['color']};'>{ml_result['risk']['label']}</span></p></div>", unsafe_allow_html=True)
            create_progress_card('ML Confidence', ml_result['probability'], ml_result['risk']['color'])
        with col2:
            st.markdown(f"<div class='metric-card'><h4>ANN prediction</h4><p><strong>{'Churn' if ann_result['prediction'] else 'Stay'}</strong></p><p>Probability: {ann_result['probability']:.2%}</p><p>Risk: <span style='color:{ann_result['risk']['color']};'>{ann_result['risk']['label']}</span></p></div>", unsafe_allow_html=True)
            create_progress_card('ANN Confidence', ann_result['probability'], ann_result['risk']['color'])

        if ml_result['prediction'] != ann_result['prediction']:
            st.warning('The models disagree. This can happen when the ANN captures non-linear interactions differently from the tree-based model. Investigate the profile for balanced retention actions.')
        else:
            st.success('Both models agree on the predicted customer outcome.')

        report = generate_prediction_report(user_values, ml_result, ann_result)
        st.download_button('Download Prediction Report', data=report, file_name='churn_prediction_report.txt', mime='text/plain')

except FileNotFoundError as err:
    st.error(str(err))
    st.info('Please place model files in the models folder or run scripts/train_models.py to generate them.')
except Exception as err:
    st.error(f'Unexpected error: {err}')
