
import streamlit as st
# Slider widget
x = st.slider("Select a value", 0, 100)
st.write(f"You selected: {x}")

# Text input widget
name = st.text_input("Enter your name")
st.write(f"Hello, {name}!")

# Button widget
if st.button('Click me'):
    st.write("Button clicked!")
    