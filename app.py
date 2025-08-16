import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
import joblib

st.title('Online Machine Healh Monitoring System')
status = 'Upload data to start Prediction'
st.write(status)
model = load_model("turbofan_engine_model_updated.h5")

def create_sequences_per_unit(df, time_steps=30):
    Xs, ys = [], []
    for unit in df['unit_number'].unique():
        unit_df = df[df['unit_number'] == unit]
        unit_X = scaler.fit_transform(unit_df.drop(['unit_number', 'RUL', 'time_in_cycles'], axis=1))
        unit_y = unit_df['RUL'].values
        for i in range(len(unit_X) - time_steps):
            Xs.append(unit_X[i:i+time_steps])
            ys.append(unit_y[i+time_steps])
    return np.array(Xs), np.array(ys)

st.sidebar.header("Upload Sensor Data File")
uploaded_file = st.sidebar.file_uploader("Upload TXT file", type=["txt"])
if st.sidebar.button('Start Prediction'):
    status = st.success('Data Uploaded Successfully')
    col_names = ['unit_number', 'time_in_cycles'] + [f'op_setting_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
    df = pd.read_csv(uploaded_file, sep='\s+', header=None, names=col_names)

    max_cycle = df.groupby('unit_number')['time_in_cycles'].max()
    df['RUL'] = df.apply(lambda row: max_cycle[row.unit_number] - row.time_in_cycles, axis=1)
    df = df.merge(max_cycle, on='unit_number', suffixes=('', '_max'))
    df.drop('time_in_cycles_max', axis=1, inplace=True)

    df_train_y = df['RUL']
    df_train_X = df.drop(['unit_number', 'RUL', 'time_in_cycles'], axis=1)

    scaler = MinMaxScaler()

    scaled_X = scaler.fit_transform(df_train_X)

    X_seq, y_seq = create_sequences_per_unit(df, time_steps=30)

    st.subheader("Uploaded Sensor Data")
    st.write(df.head())

    y_pred_scaled = model.predict(X_seq)
    # y_pred = scaler_y.inverse_transform(y_pred_scaled)
    result_df = pd.DataFrame({
            "Predicted_RUL": y_pred_scaled.flatten()
        })
    st.subheader("Prediction Results")
    st.write(result_df)
