

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.model_selection import train_test_split

# Load the Telco Customer Churn dataset
df = pd.read_csv("Updated_churn_dataset.csv")
df2 = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Add a title and description to the app
st.title("Ameerah's Churn prediction model")
st.write("""
This app uses **LogisticRegression** to predict the likelihood of a customer churning
based on input features.
""")

# Display the dataset
st.write("### Telco Customer Churn dataset", df2)

# Create sliders for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    # Create a form to collect user input
    with st.sidebar.form('user_input'):
        contract_type = st.selectbox('Contract Type', ['One year', 'Two year', 'Month-to-month'])
        has_streaming_tv = st.checkbox('Streaming TV')
        has_tech_support = st.checkbox('Tech Support')
        has_streaming_movies = st.checkbox('Streaming Movies')
        payment_method = st.selectbox('Payment Method', ['Bank transfer (automatic)', 'Mailed check', 'Credit card (automatic)', 'Electronic check'])
        has_online_security = st.checkbox('Online Security')
        has_device_protection = st.checkbox('Device Protection')
        has_online_backup = st.checkbox('Online Backup')
        has_multiple_lines = st.checkbox('Multiple Lines')
        dependents = st.checkbox('Dependents')
        is_senior_citizen = st.checkbox('Senior Citizen')
        internet_service = st.selectbox('Internet Service', ['Fiber optic', 'Other'])
        has_partner = st.checkbox('Partner')
        has_paperless_billing = st.checkbox('Paperless Billing')
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        monthly_charges = st.number_input('Monthly Charges', min_value=0.0, step=0.1)
        tenure = st.number_input('Tenure', min_value=0, step=1)

        # Create a submit button
        submitted = st.form_submit_button('Submit')

        # Process the user input when the submit button is clicked
        if submitted:
            input_data = {
                'Contract Type': [contract_type],
                'Streaming TV': [has_streaming_tv],
                'Tech Support': [has_tech_support],
                'Streaming Movies': [has_streaming_movies],
                'Payment Method': [payment_method],
                'Online Security': [has_online_security],
                'Device Protection': [has_device_protection],
                'Online Backup': [has_online_backup],
                'Multiple Lines': [has_multiple_lines],
                'Dependents': [dependents],
                'Senior Citizen': [is_senior_citizen],
                'Internet Service': [internet_service],
                'Partner': [has_partner],
                'Paperless Billing': [has_paperless_billing],
                'Gender': [gender],
                'Monthly Charges': [monthly_charges],
                'Tenure': [tenure]
            }
            return pd.DataFrame(input_data)  # Return the DataFrame
    return None  # Return None if not submitted

# Call the user input function
input_df = user_input_features()

if input_df is not None:
    # Preprocessing: encode categorical variables to match training data
    input_df_encoded = pd.get_dummies(input_df)

    # Ensure that the input DataFrame has the same columns as the training data
    X = df.drop(columns=['Churn','customerID','TotalCharges'])
    X_encoded = pd.get_dummies(X)

    # Align the columns of the input_df_encoded to match X_encoded
    input_df_encoded = input_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Split the dataset into features (X) and target (Y)
    Y = df['Churn']
   

    # Train the model using the entire dataset
    clf = LogisticRegression()
    clf.fit(X_encoded, Y)

    # Display the user input
    st.write(input_df)

    # Make predictions
    prediction = clf.predict(input_df_encoded)
    prediction_proba = clf.predict_proba(input_df_encoded)

    st.subheader('Prediction')
    st.write('Churn' if prediction[0] == 1 else 'Not Churn')

    st.subheader('Prediction Probability')
    st.write('Churn Probability: {:.2f}%'.format(prediction_proba[0][1] * 100))

