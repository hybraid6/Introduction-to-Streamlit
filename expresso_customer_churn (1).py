#### Import the libraries
import pandas as pd
import streamlit as st
import numpy as np
import gdown
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

st.title('Expresso Customers Churn Prediction')
st.write('This model use LogisticRegression to make prediction')

# Create user input (widgets)
st.sidebar.header("Input features for prediction")

# Load the dataset
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1h_dzCN7rbOspMJcM-N_GyPhJWpLxsAZr"
    output = "expresso_dataset.csv"
    gdown.download(url, output, quiet=False)
    data = pd.read_csv(output)
    data.columns = data.columns.str.strip()
    return data
data = load_data()
expresso_df = data.drop(columns = ['user_id', 'MONTANT', 'DATA_VOLUME', 'ARPU_SEGMENT', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2'])
expresso_df = expresso_df.dropna(thresh=int(0.7 * expresso_df.shape[1]), axis=0)
expresso_df['REGION'] = expresso_df['REGION'].fillna(expresso_df['REGION'].mode()[0])
expresso_df['FREQUENCE_RECH'] = expresso_df['FREQUENCE_RECH'].fillna(expresso_df['FREQUENCE_RECH'].mean())
expresso_df.dropna(inplace = True, subset = ['REVENUE'])
expresso_df.dropna(inplace = True, subset = ['FREQUENCE'])
expresso_df['ON_NET'] = expresso_df['ON_NET'].fillna(expresso_df['ON_NET'].mean())
expresso_df['TOP_PACK'] = expresso_df['TOP_PACK'].fillna(expresso_df['TOP_PACK'].mode()[0])
expresso_df['FREQ_TOP_PACK'] = expresso_df['FREQ_TOP_PACK'].fillna(expresso_df['FREQ_TOP_PACK'].median())
expresso_df['Churn_name'] = np.where(expresso_df['CHURN'] == 0, 'No', 'Yes')

# Display the dataset
st.write('Expresso Customers Churn', expresso_df.head(50))

# The features
region = st.sidebar.selectbox('Region', expresso_df['REGION'].unique())
tenure = st.sidebar.selectbox('Tenure', expresso_df['TENURE'].unique())
freq_rech = st.sidebar.slider('Frequence Reach',
                              float(expresso_df['FREQUENCE_RECH'].min()),
                              float(expresso_df['FREQUENCE_RECH'].mean()),
                              float(expresso_df['FREQUENCE_RECH'].max()))
revenue = st.sidebar.number_input('Enter revenue', min_value = 0)
freq = st.sidebar.slider('Frequence',
                        float(expresso_df['FREQUENCE'].min()),
                        float(expresso_df['FREQUENCE'].mean()),
                        float(expresso_df['FREQUENCE'].max()))
on_net = st.sidebar.slider('ON_NET',
                            float(expresso_df['ON_NET'].min()),
                            float(expresso_df['ON_NET'].mean()),
                            float(expresso_df['ON_NET'].max()))
regular = st.sidebar.slider('Regularity',
                              float(expresso_df['REGULARITY'].min()),
                              float(expresso_df['REGULARITY'].mean()),
                              float(expresso_df['REGULARITY'].max()))
freq_top_pack = st.sidebar.slider('Freq Top Pack',
                              float(expresso_df['FREQ_TOP_PACK'].min()),
                              float(expresso_df['FREQ_TOP_PACK'].mean()),
                              float(expresso_df['FREQ_TOP_PACK'].max()))
top_pack = st.sidebar.selectbox('Top Pack', expresso_df['TOP_PACK'].unique())
# Select the columns to be used for modeling
express_df = expresso_df[['REGION', 'TENURE', 'FREQUENCE_RECH', 'REVENUE', 'FREQUENCE', 'ON_NET', 'REGULARITY', 'FREQ_TOP_PACK', 'TOP_PACK', 'CHURN']]
# Encode the categorical features
encode_data = pd.get_dummies(express_df, columns = express_df.select_dtypes(include = 'object').columns)

# split the data into X and y and also into train and test
# Features (X) and target(y)
X = encode_data.drop(columns = 'CHURN')
y = encode_data['CHURN']

# Train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fit smote data
@st.cache_data
def train_model(X, y):
    # Instantiate SMOTE
    smote = SMOTE(k_neighbors = 3, random_state = 42)
    # Fit and resample X_train, y_train
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    # Instantiate LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_smote, y_smote)
    return logreg
train = train_model(X_train, y_train)

# Prepare a dataframe
user_input = pd.DataFrame({
    'REGION': [region],
    'TENURE': [tenure],
    'FREQUENCE_RECH': [freq_rech],
    'REVENUE': [revenue],
    'FREQUENCE': [freq],
    'ON_NET': [on_net],
    'REGULARITY': [regular],
    'FREQ_TOP_PACK': [freq_top_pack],
    'TOP_PACK': top_pack  # or allow user to choose
})

# Encode user_input to match training data
user_input_encoded = pd.get_dummies(user_input, columns = user_input.select_dtypes(include = 'object').columns)
user_input_encoded = user_input_encoded.reindex(columns = X.columns, fill_value = 0)

# Predict user_input
user_prediction = train.predict(user_input_encoded)[0]
label = 'No' if user_prediction == 0 else 'Yes'
st.subheader('User Prediction')
st.success(f'The probability that the customer is likely to churn is {label}')
