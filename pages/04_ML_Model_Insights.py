import streamlit as st
from sklearn.model_selection import train_test_split
from utils.data_loader import load_churn_data, preprocess_features
from utils.model_utils import load_ml_model, load_preprocessor, evaluate_model
from utils.visuals import load_css, plot_confusion_matrix, plot_roc_curve, plot_feature_importance

st.set_page_config(page_title='Machine Learning Model | Customer Retention Intelligence', page_icon='🤖', layout='wide')
load_css('assets/style.css')

st.title('Machine Learning Model Insights')
st.markdown('Performance metrics, confusion matrix, and explainability for the bank churn classifier.')

try:
    df = load_churn_data()
    X_raw = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
    y = df['Exited']
    preprocessor = load_preprocessor()
    ml_model = load_ml_model()
    X = preprocess_features(df, preprocessor)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    metrics = evaluate_model(ml_model, X_test, y_test)

    col1, col2, col3 = st.columns(3)
    col1.metric('Accuracy', f"{metrics['accuracy']*100:.2f}%")
    col2.metric('Precision', f"{metrics['precision']*100:.2f}%")
    col3.metric('Recall', f"{metrics['recall']*100:.2f}%")
    st.metric('F1 Score', f"{metrics['f1_score']*100:.2f}%")
    st.metric('ROC-AUC', f"{metrics['roc_auc']*100:.2f}%")

    st.subheader('Confusion Matrix')
    st.plotly_chart(plot_confusion_matrix(metrics['confusion_matrix']), use_container_width=True)

    st.subheader('ROC Curve')
    y_prob = ml_model.predict_proba(X_test)[:, 1] if hasattr(ml_model, 'predict_proba') else ml_model.predict(X_test)
    st.plotly_chart(plot_roc_curve(y_test, y_prob), use_container_width=True)

    if hasattr(ml_model, 'feature_importances_'):
        feature_names = list(preprocessor.get_feature_names_out())
        st.subheader('Feature Importance')
        st.plotly_chart(plot_feature_importance(feature_names, ml_model.feature_importances_), use_container_width=True)

    st.markdown('---')
    st.header('Model explanation and hyperparameter summary')
    st.write(f'- **Model type:** {ml_model.__class__.__name__}')
    if hasattr(ml_model, 'get_params'):
        params = ml_model.get_params()
        st.write('**Selected hyperparameters:**')
        st.write({k: params[k] for k in ['n_estimators', 'learning_rate', 'max_depth'] if k in params})
    st.write(
        'This model is trained on balanced features with a standard scaling pipeline and geography one-hot encoding. ' 
        'It is built for strong classification accuracy while preserving interpretability through feature importance analysis.'
    )
except FileNotFoundError as err:
    st.error(str(err))
    st.info('Run the model build script or place the published model files in the models/ directory.')
except Exception as err:
    st.error(f'Error loading ML model: {err}')
