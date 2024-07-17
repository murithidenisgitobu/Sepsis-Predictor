import streamlit as st
import requests

# Page title
st.set_page_config(page_title='Sepsis Prediction Application', layout='wide', page_icon='🏠')

# Header
st.title('Sepsis Prediction Application')

# Function to capture user inputs
def features():
    with st.form("sepsis_form"):
        col1, col2 = st.columns(2)

        with col1:
            PRG = st.number_input('Plasma glucose (PRG)', min_value=0, max_value=20, step=1)
            PL = st.number_input('Blood Work Result-1 (mu U/ml) (PL)', min_value=0, max_value=200, step=1)
            PR = st.number_input('Blood Pressure (mm Hg) (PR)', min_value=0, max_value=125, step=1)
            SK = st.number_input('Blood Work Result-2 (mm) (SK)', min_value=0, max_value=100, step=1)
            TS = st.number_input('Blood Work Result-3 (mu U/ml) (TS)', min_value=0, max_value=900, step=1)

        with col2:
            M11 = st.number_input('Body mass index (weight in kg/(height in m)^2) (M11)', min_value=0.0, max_value=68.0)
            BD2 = st.number_input('Blood Work Result-4 (mu U/ml) (BD2)', min_value=0.0, max_value=2.5)
            Age = st.number_input("Patient's age (Age)", min_value=0, max_value=100, step=1)
            Insurance = st.selectbox('Insurance', ['No', 'Yes'])

        submit_button = st.form_submit_button("Submit")

    return submit_button, PRG, PL, PR, SK, TS, M11, BD2, Age, Insurance

# Capture form data
submit_button, PRG, PL, PR, SK, TS, M11, BD2, Age, Insurance = features()

# Model selection
model_choice = st.radio(
    "Choose Model for Prediction:",
    ('Random Forest', 'XGBoost')
)

# API URLs
api_urls = {
    "Random Forest": "http://127.0.0.1:8000/random_forest_prediction",
    "XGBoost": "http://127.0.0.1:8000/xgboost_prediction"
}

# Make prediction
if submit_button:
    data = {
        'PRG': PRG,
        'PL': PL,
        'PR': PR,
        'SK': SK,
        'TS': TS,
        'M11': M11,
        'BD2': BD2,
        'Age': Age,
        'Insurance': Insurance
    }
    
    url = api_urls[model_choice]
    response = requests.post(url, json=data)

    if response.status_code == 200:
        prediction = response.json()['Sepsis Prediction']

        if prediction == 'Positive':
            st.success(f'The patient will develop sepsis')
        
        else:
            st.success(f'The patient will not develop sepsis')
    else:
        st.error(f'Error: {response.json()["detail"]}')
