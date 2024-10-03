import streamlit as st
import pandas as pd
# Sample data
data = {
    'Name': ['Mai', 'Azula', 'Toph'],
    'Age': [14, 15, 12],
    'Bending Style': ['Non Bender', 'Fire bender', 'Earth Bender']
}

df = pd.DataFrame(data)

st.write("### DataFrame Example")
st.dataframe(df)  # Display the dataframe

    
