
#### Step 1: Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


#### Step 2: Load the Data
# We will load the famous Iris dataset using `sklearn` and display it in our app.


# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add a column for the target variable (species)
df['species'] = iris.target

# Map the target integers to the actual class names
df['species_name'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Add a title and description to the app
st.title("Iris Flower Classification")
st.write("""
This app uses **Random Forest Classifier** to predict the type of Iris flower based on input features.
""")

# Display the dataset
st.write("### Iris Dataset", df)


#### Step 3: Create User Inputs (Widgets)

# Use Streamlit's widgets to allow users to input values for the model features.


# Create sliders for user input
st.sidebar.header('User Input Parameters')

def user_input_features():


    # Slider for Sepal Length
    # Set the label, minimum value, maximum value, and default value (mean) for the slider
    sepal_length = st.sidebar.slider(
        'Sepal length',  # Label displayed for the slider
        float(df['sepal length (cm)'].min()),  # Minimum value from the dataset
        float(df['sepal length (cm)'].max()),  # Maximum value from the dataset
        float(df['sepal length (cm)'].mean())  # Default value set to the mean of sepal lengths
    )

    # Slider for Sepal Width
    # Similar to the sepal length slider, but for sepal width
    sepal_width = st.sidebar.slider(
        'Sepal width',  # Label for the slider
        float(df['sepal width (cm)'].min()),  # Minimum value from the dataset
        float(df['sepal width (cm)'].max()),  # Maximum value from the dataset
        float(df['sepal width (cm)'].mean())  # Default value set to the mean of sepal widths
    )

    # Slider for Petal Length
    # Slider to input petal length with appropriate labels and values
    petal_length = st.sidebar.slider(
        'Petal length',  # Label for the slider
        float(df['petal length (cm)'].min()),  # Minimum value from the dataset
        float(df['petal length (cm)'].max()),  # Maximum value from the dataset
        float(df['petal length (cm)'].mean())  # Default value set to the mean of petal lengths
    )

    # Slider for Petal Width
    # Slider for petal width with similar structure to previous sliders
    petal_width = st.sidebar.slider(
        'Petal width',  # Label for the slider
        float(df['petal width (cm)'].min()),  # Minimum value from the dataset
        float(df['petal width (cm)'].max()),  # Maximum value from the dataset
        float(df['petal width (cm)'].mean())  # Default value set to the mean of petal widths
    )

    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


#### Step 4: Train the Model and Make Predictions

# We'll use a Random Forest Classifier to train the model and predict the class based on user inputs.

# Train the model using the entire Iris dataset
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # Convert to DataFrame
Y = iris.target
clf = RandomForestClassifier()
clf.fit(X, Y)

# Make predictions
prediction = clf.predict(input_df)
prediction_proba = clf.predict_proba(input_df)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)


#### Step 5: Visualizing Results

# You can use Streamlit's `st.line_chart()` or `st.bar_chart()` to show charts or integrate Matplotlib/Plotly for custom visualizations.


# Visualization Example: Show the feature importance
st.subheader('Feature Importance')
import matplotlib.pyplot as plt

importance = clf.feature_importances_
features = iris.feature_names
plt.barh(features, importance)
st.pyplot(plt)

