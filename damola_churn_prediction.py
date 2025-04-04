import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load the trained churn prediction model
model = pickle.load(open(r"churn_model.pkl", 'rb'))

# Set up the title and description of the app
st.title("Bank Churn Prediction App")
st.write("Enter customer details to predict if a customer is likely to churn.")

# Sidebar: Collect user input features
st.sidebar.header("Customer Input Features")

def user_input_features():
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)
    geography = st.sidebar.selectbox("Geography", ("France", "Germany", "Spain"))
    gender = st.sidebar.selectbox("Gender", ("Female", "Male"))
    age = st.sidebar.slider("Age", 18, 100, 30)
    tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
    balance = st.sidebar.number_input("Account Balance", value=0.0)
    num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.sidebar.selectbox("Has Credit Card", ("Yes", "No"))
    is_active_member = st.sidebar.selectbox("Is Active Member", ("Yes", "No"))
    estimated_salary = st.sidebar.number_input("Estimated Salary", value=0.0)
    
    # Create a dictionary for numeric features (the ones used directly)
    data = {
        "CreditScore": credit_score,
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active_member == "Yes" else 0,
        "EstimatedSalary": estimated_salary
    }
    
    # Convert to a DataFrame
    input_df = pd.DataFrame(data, index=[0])
    
    # Process Geography using one-hot encoding similar to pd.get_dummies
    # Assume your training data created these dummy columns:
    geo_dict = {"Geography_France": 0, "Geography_Germany": 0, "Geography_Spain": 0}
    geo_key = f"Geography_{geography}"
    geo_dict[geo_key] = 1
    geo_df = pd.DataFrame([geo_dict])
    
    
    # Combine numeric features and dummy variables for Geography
    input_df = pd.concat([input_df, geo_df], axis=1)
    
    # (Optional) Reindex to match the exact order of training features if necessary:
    # training_columns = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    #                     "HasCrCard", "IsActiveMember", "EstimatedSalary", "Gender",
    #                     "Geography_France", "Geography_Germany", "Geography_Spain"]
    # input_df = input_df.reindex(columns=training_columns, fill_value=0)
    
    return input_df

# Get the user input features DataFrame
input_features = user_input_features()

# Display the input features
st.subheader("Customer Input Features")
st.write(input_features)

# Predict churn when the user clicks the button
if st.button("Predict Churn"):
    # Make a prediction using the loaded model
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)
    
    # Interpret the prediction (assume 1 means churn, 0 means not churn)
    churn_label = "Churn" if prediction[0] == 1 else "Not Churn"
    
    st.subheader("Prediction")
    st.write(f"The customer is predicted to: **{churn_label}**")
    
    st.subheader("Prediction Probabilities")
    st.write(prediction_proba)