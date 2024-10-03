import streamlit as st
st.title("Simple User Input Form")

# Create text input for name
name = st.text_input("Enter your name")

# Create slider for age
age = st.slider("Select your age", 18, 100)

# Create a select box for favorite number
favorite_number = st.selectbox("Select your favorite number", [1, 2, 3, 4, 5])

# Display the input back to the user
st.write(f"Hello, {name}! You are {age} years old and your favorite number is {favorite_number}.")
