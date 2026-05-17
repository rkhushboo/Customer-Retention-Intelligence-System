import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from utils.data_loader import load_churn_data, preprocess_features
from utils.model_utils import load_ann_model, load_ann_history, load_preprocessor
from utils.visuals import load_css, plot_confusion_matrix, plot_roc_curve

st.set_page_config(page_title='ANN Model | Customer Retention Intelligence', page_icon='🧠', layout='wide')
load_css('assets/style.css')

st.title('ANN Model Insights')
st.markdown('Deep neural network performance, training behavior, and model lifecycle explanation.')

try:
    df = load_churn_data()
    preprocessor = load_preprocessor()
    ann_model = load_ann_model()
    history = load_ann_history()

    X = preprocess_features(df, preprocessor)
    y = df['Exited']
    y_prob = ann_model.predict(X).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_prob),
        'confusion_matrix': confusion_matrix(y, y_pred)
    }

    col1, col2, col3 = st.columns(3)
    col1.metric('Accuracy', f"{metrics['accuracy']*100:.2f}%")
    col2.metric('Precision', f"{metrics['precision']*100:.2f}%")
    col3.metric('Recall', f"{metrics['recall']*100:.2f}%")
    st.metric('F1 Score', f"{metrics['f1_score']*100:.2f}%")
    st.metric('ROC-AUC', f"{metrics['roc_auc']*100:.2f}%")

    st.subheader('ANN Architecture Summary')
    st.write('The ANN is built with dense layers, dropout regularization and sigmoid output for customer churn probability.')
    st.markdown(
        '''
- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **EarlyStopping:** Enabled to avoid overfitting
- **ModelCheckpoint:** Saves the best validation model automatically
- **KerasTuner:** Useful for hyperparameter search when tuning is active
        '''
    )

    if history:
        import plotly.graph_objects as go
        if 'accuracy' in history and 'val_accuracy' in history:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['accuracy'], name='Train Accuracy', line=dict(color='#0D6EFD')))
            fig.add_trace(go.Scatter(y=history['val_accuracy'], name='Validation Accuracy', line=dict(color='#20c997')))
            fig.update_layout(title='ANN Accuracy Curve', xaxis_title='Epoch', yaxis_title='Accuracy', template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

        if 'loss' in history and 'val_loss' in history:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=history['loss'], name='Train Loss', line=dict(color='#ffc107')))
            fig2.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss', line=dict(color='#dc3545')))
            fig2.update_layout(title='ANN Loss Curve', xaxis_title='Epoch', yaxis_title='Loss', template='plotly_dark')
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader('Confusion Matrix and ROC')
    st.plotly_chart(plot_confusion_matrix(metrics['confusion_matrix']), use_container_width=True)
    st.plotly_chart(plot_roc_curve(y, y_prob), use_container_width=True)

    st.markdown('---')
    st.write(
        'EarlyStopping stops training when validation loss stops improving, which helps prevent overfitting. '
        'ModelCheckpoint saves the best performing model weights. KerasTuner automates tuning of layer sizes, activation functions, optimizers and dropout rates.'
    )
except FileNotFoundError as err:
    st.error(str(err))
    st.info('Place the trained ANN model into models/best_ann_model.keras and history into models/ann_history.json.')
except Exception as err:
    st.error(f'Error loading ANN model: {err}')
