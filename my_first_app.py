
"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


st.title('Predicting Turbofan Engine Degradation')

st_df = pd.read_csv('st_df.csv')

st.subheader('Predicted Values Vs. True Values')

fig = go.Figure()
st_df = st_df.sort_values(by='true_values', ascending=False)
fig.add_trace(go.Scatter(x=st_df['true_values'], y=st_df['features'],
                    mode='lines',
                    name='True RUL'))
st_df = st_df.sort_values(by='lstm_pred', ascending=False)
fig.add_trace(go.Scatter(x=st_df['lstm_pred'], y=st_df['features'],
                    mode='lines',
                    name='LSTM'))
st_df = st_df.sort_values(by='pred_br', ascending=False)
fig.add_trace(go.Scatter(x=st_df['pred_br'], y=st_df['features'],
                    mode='lines',
                    name='Bayesian Ridge'))
st_df = st_df.sort_values(by='gru', ascending=False)
fig.add_trace(go.Scatter(x=st_df['gru'], y=st_df['features'],
                    mode='lines',
                    name='GRU'))
st_df = st_df.sort_values(by='dcnn', ascending=False)
fig.add_trace(go.Scatter(x=st_df['dcnn'], y=st_df['features'],
                    mode='lines',
                    name='DCNN'))
st_df = st_df.sort_values(by='lr_pred', ascending=False)
fig.add_trace(go.Scatter(x=st_df['lr_pred'], y=st_df['features'],
                    mode='lines',
                    name='Linear Regression'))
st_df = st_df.sort_values(by='re_pred', ascending=False)
fig.add_trace(go.Scatter(x=st_df['re_pred'], y=st_df['features'],
                    mode='lines',
                    name='Random Effects'))
fig.update_layout(
    xaxis_title="RUL (In Cycles)",
    yaxis_title="Sensor Features",
    legend_title="Model Name",
    font=dict(
        size=16
    )
)

st.plotly_chart(fig)

st.write("Note: Y-axis represents all 21 sensor features compressed into a single dimension using PCA. However, we have used all 21 features to train the models.") 

st.subheader('Model Performance Metrics')

st.write("LSTM has the best performance while Linear Regression and Random Effects has the worst.") 

mae_df = pd.read_csv('mae.csv')

rmse_df = pd.read_csv('rmse.csv')

fig = go.Figure()
fig.add_trace(go.Bar(
    x=mae_df['model'],
    y=mae_df['mae'],
    name='MAE',
    marker_color='indianred'
))

fig.add_trace(go.Bar(
    x=rmse_df['model'],
    y=rmse_df['rmse'],
    name='RMSE',
    marker_color='lightsalmon'
))

fig.update_layout(barmode='group', xaxis_tickangle=-45)

st.plotly_chart(fig)
