import streamlit as st
import joblib
import pandas as pd
import os

# Load files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "bank_model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))

st.title("🏦 Bank Marketing Prediction")
st.write("Predict if customer will subscribe")

# Helper
def get_classes(col):
    return encoders[col].classes_

# ----------- INPUTS -----------

age = st.number_input("Age", 18, 100)

job = st.selectbox("Job", get_classes('job'))
marital = st.selectbox("Marital", get_classes('marital'))
education = st.selectbox("Education", get_classes('education'))
default = st.selectbox("Default", get_classes('default'))
housing = st.selectbox("Housing", get_classes('housing'))
loan = st.selectbox("Loan", get_classes('loan'))

contact = st.selectbox("Contact", get_classes('contact'))
month = st.selectbox("Month", get_classes('month'))
day_of_week = st.selectbox("Day of Week", get_classes('day_of_week'))

duration = st.number_input("Call Duration", value=0.0)
campaign = st.number_input("Campaign Contacts", value=1)
pdays = st.number_input("Pdays", value=-1)
previous = st.number_input("Previous Contacts", value=0)

poutcome = st.selectbox("Previous Outcome", get_classes('poutcome'))

# Economic indicators
emp_var_rate = st.number_input("Employment Variation Rate", value=0.0)
cons_price_idx = st.number_input("Consumer Price Index", value=0.0)
cons_conf_idx = st.number_input("Consumer Confidence Index", value=0.0)
euribor3m = st.number_input("Euribor 3 Month Rate", value=0.0)
nr_employed = st.number_input("Number of Employees", value=0.0)

# ----------- INPUT DICT -----------

input_dict = {
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'month': month,
    'day_of_week': day_of_week,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome,
    'emp.var.rate': emp_var_rate,
    'cons.price.idx': cons_price_idx,
    'cons.conf.idx': cons_conf_idx,
    'euribor3m': euribor3m,
    'nr.employed': nr_employed
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Encode categorical
for col in input_df.columns:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

# Align columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Customer WILL Subscribe")
    else:
        st.error("❌ Customer will NOT Subscribe")