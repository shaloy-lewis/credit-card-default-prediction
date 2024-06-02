import streamlit as st
import pandas as pd
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

# Initialize prediction pipeline
predict_pipeline = PredictPipeline()

# Streamlit app title
st.title("Credit Default Prediction")

# Input fields for user data
st.header("Enter the details:")

LIMIT_BAL = st.number_input('Limit Balance', min_value=0.0)
AGE = st.number_input('Age', min_value=0)
BILL_AMT1 = st.number_input('Bill Amount 1', min_value=0.0)
BILL_AMT2 = st.number_input('Bill Amount 2', min_value=0.0)
BILL_AMT3 = st.number_input('Bill Amount 3', min_value=0.0)
BILL_AMT4 = st.number_input('Bill Amount 4', min_value=0.0)
BILL_AMT5 = st.number_input('Bill Amount 5', min_value=0.0)
BILL_AMT6 = st.number_input('Bill Amount 6', min_value=0.0)
PAY_AMT1 = st.number_input('Payment Amount 1', min_value=0.0)
PAY_AMT2 = st.number_input('Payment Amount 2', min_value=0.0)
PAY_AMT3 = st.number_input('Payment Amount 3', min_value=0.0)
PAY_AMT4 = st.number_input('Payment Amount 4', min_value=0.0)
PAY_AMT5 = st.number_input('Payment Amount 5', min_value=0.0)
PAY_AMT6 = st.number_input('Payment Amount 6', min_value=0.0)
EDUCATION = st.selectbox('Education', options=['graduate_school', 'university', 'high_school', 'others'])
MARRIAGE = st.selectbox('Marriage', options=['married', 'single', 'others'])
SEX = st.selectbox('Sex', options=['male', 'female'])
PAY_0 = st.selectbox('Repayment Status in September', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])
PAY_2 = st.selectbox('Repayment Status in August', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])
PAY_3 = st.selectbox('Repayment Status in July', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])
PAY_4 = st.selectbox('Repayment Status in June', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])
PAY_5 = st.selectbox('Repayment Status in May', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])
PAY_6 = st.selectbox('Repayment Status in April', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])

# Button to submit data and get predictions
if st.button('Predict'):
    # Create a CustomData instance
    custom_data = CustomData(
        LIMIT_BAL=LIMIT_BAL,
        AGE=AGE,
        BILL_AMT1=BILL_AMT1,
        BILL_AMT2=BILL_AMT2,
        BILL_AMT3=BILL_AMT3,
        BILL_AMT4=BILL_AMT4,
        BILL_AMT5=BILL_AMT5,
        BILL_AMT6=BILL_AMT6,
        PAY_AMT1=PAY_AMT1,
        PAY_AMT2=PAY_AMT2,
        PAY_AMT3=PAY_AMT3,
        PAY_AMT4=PAY_AMT4,
        PAY_AMT5=PAY_AMT5,
        PAY_AMT6=PAY_AMT6,
        EDUCATION=EDUCATION,
        MARRIAGE=MARRIAGE,
        SEX=SEX,
        PAY_0=PAY_0,
        PAY_2=PAY_2,
        PAY_3=PAY_3,
        PAY_4=PAY_4,
        PAY_5=PAY_5,
        PAY_6=PAY_6
    )

    # Convert the user inputs to a DataFrame
    input_df = custom_data.get_data_as_dataframe()

    # Get predictions
    predictions = predict_pipeline.predict(input_df)

    # Display the prediction results
    st.subheader("Prediction Results:")
    st.write(f"Probability of default: {predictions[0][1]:.4f}")
