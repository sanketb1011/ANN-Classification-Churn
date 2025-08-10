import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)


st.title("Bank Customer Churn Prediction")

geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', step=100.0)
credit_score = st.number_input('Credit Score', step=1.0)
estimated_salary = st.number_input('Estimated Salary', step=100.0)
tenure = st.slider('Tenure', 0, 10)
number_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


geo_encoded = one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder.get_feature_names_out(['Geography']))

gender_encoded = label_encoder_gender.transform([gender])[0]


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine with one-hot encoded geography
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale input
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display result
st.write(f"Prediction Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write("The customer is **likely to leave** the bank.")
else:
    st.write("The customer is **likely to stay** with the bank.")
