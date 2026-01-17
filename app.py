import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load pollutant data
df_pollutants = pd.read_csv("pollutants.csv")  # 替换成你的 CSV 路径

st.markdown("<h1 style='text-align: center;'>Catalyst Preparation Guidance for Better Catalytic Performance</h1>", unsafe_allow_html=True)

# --- Column layout ---
col1, col2, col3, col4 = st.columns(4)

# --- Column 1: Prep temp & doping ---
with col1:
    prep_temp = st.number_input("Prep temp (°C)", value=600, step=10)
    doped_metal = st.selectbox("Doped with metals?", [0, 1])
    doped_nonmetal = st.selectbox("Doped with non-metals?", [0, 1])

# --- Column 2: Pollutant & conc ---
with col2:
    selected_pollutant = st.selectbox("Select pollutant", df_pollutants['Pollutant'].tolist())
    pollutant_row = df_pollutants[df_pollutants['Pollutant'] == selected_pollutant].iloc[0]

    pollutant_conc = st.number_input("Pollutant conc.(mg/L)", value=10, step=1)
    catalyst_conc = st.number_input("Catalyst conc.(mg/L)", value=200, step=1)
    pms_conc = st.number_input("PMS conc.(mg/L)", value=200, step=1)
    ph_value = st.number_input("pH", value=7.0, step=0.1)

# --- Column 3: Co inputs ---
with col3:
    co_values = [st.number_input(f"Co(wt%) {i + 1}", value=0.05 + 0.01 * i, step=0.01) for i in range(4)]

# --- Column 4: SSA inputs ---
with col4:
    sa_values = [st.number_input(f"SSA {i + 1}", value=300 + i * 100, step=1) for i in range(4)]

st.markdown("---")

# --- Centered Predict Button ---
col_left, col_center, col_right = st.columns([1.5, 1, 1])  # 三列布局，中间列宽度大
with col_center:
    predict_clicked = st.button("Predict")

# --- Prediction Logic ---
if predict_clicked:
    E = pollutant_row['E']
    HD = pollutant_row['HD']
    CP = pollutant_row['CP']

    best_pred = -np.inf
    best_combo = None

    for co in co_values:
        for sa in sa_values:
            cat_input = np.array([[doped_metal, doped_nonmetal]])
            num_input = np.array([[E, HD, CP, prep_temp,
                                   co, sa, pollutant_conc, catalyst_conc,
                                   pms_conc, ph_value]])
            num_input_scaled = scaler.transform(num_input)
            full_input = np.hstack((cat_input, num_input_scaled))
            pred = model.predict(full_input)[0]

            if pred > best_pred:
                best_pred = pred
                best_combo = (co, sa)

    # 输出结果在新行，保持默认布局
    st.success(
        f"Best combination: Co(wt%) = {best_combo[0]}, Surface Area = {best_combo[1]}, Predicted = {best_pred:.2f}")



