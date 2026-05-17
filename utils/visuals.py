import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay


def load_css(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning('Custom style file not found, using default theme.')


def plotly_histogram(df: pd.DataFrame, x: str, color: str = None, title: str = None):
    fig = px.histogram(df, x=x, color=color, barmode='overlay', template='plotly_dark', opacity=0.85)
    fig.update_layout(title=title or f'{x} Distribution', legend_title_text=color or '')
    fig.update_traces(marker_line_width=0.5, marker_line_color='white')
    return fig


def plotly_bar(df: pd.DataFrame, x: str, y: str, color: str = None, title: str = None):
    fig = px.bar(df, x=x, y=y, color=color, template='plotly_dark', text_auto=True)
    fig.update_layout(title=title or f'{x} vs {y}', xaxis_title=x, yaxis_title=y)
    return fig


def plot_churn_distribution(df: pd.DataFrame):
    counts = df['Exited'].value_counts().rename(index={0: 'Retained', 1: 'Churned'})
    fig = px.pie(values=counts.values, names=counts.index, color=counts.index,
                 color_discrete_map={'Retained': '#28a745', 'Churned': '#dc3545'}, hole=0.35,
                 template='plotly_dark')
    fig.update_layout(title='Customer Churn Distribution')
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    corr = df.select_dtypes(include=['number']).corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='thermal', template='plotly_dark')
    fig.update_layout(title='Correlation Heatmap for Numeric Features')
    return fig


def plot_feature_importance(feature_names, importances):
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h', template='plotly_dark', title='Feature Importance')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def plot_confusion_matrix(matrix, labels=['Retained', 'Churned']):
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale='blues',
        hoverongaps=False,
        text=matrix,
        texttemplate='%{text}',
    ))
    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted Label', yaxis_title='True Label')
    return fig


def plot_roc_curve(y_true, y_score):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#17becf', width=3), name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='white', width=1, dash='dash'), showlegend=False))
    fig.update_layout(title=f'ROC Curve (AUC = {roc_auc:.3f})', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', template='plotly_dark')
    return fig


def create_progress_card(label: str, value: float, color: str):
    text = f"{value*100:.1f}%"
    st.markdown(
        f"<div class='progress-card'><h4>{label}</h4><div class='progress-meter'><div style='width: {value*100:.1f}%; background: {color};'></div></div><span>{text}</span></div>",
        unsafe_allow_html=True,
    )


def render_metric_cards(cards: list[dict]):
    columns = st.columns(len(cards))
    for col, card in zip(columns, cards):
        with col:
            st.markdown(f"<div class='metric-card'><span class='metric-label'>{card['label']}</span><h3>{card['value']}</h3><p>{card['description']}</p></div>", unsafe_allow_html=True)
