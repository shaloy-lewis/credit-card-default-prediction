import streamlit as st
import pandas as pd
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

# Initialize prediction pipeline
predict_pipeline = PredictPipeline()

# Streamlit app title
st.title("Credit Default Prediction")

page = st.sidebar.selectbox('Page Navigation', ["Problem statement", "Predictor",])

st.sidebar.markdown("""---""")
st.sidebar.write("Created by [Shaloy Lewis](https://www.linkedin.com/in/shaloy-lewis/)")

if page=="Problem statement":
    st.write("""Predicting credit card default is essential for effective risk management, optimal credit allocation, customer retention, economic stability, and regulatory compliance. Accurate prediction of default probability can significantly enhance these areas. This machine learning model aims to address this critical issue by predicting the likelihood of default using member demographics and credit history as inputs.

The generated risk score can be utilized by lenders for informed decision-making, thereby mitigating financial risks and fostering the stability and growth of the financial sector. This model assists in optimizing the risk-adjusted return on capital, improving portfolio quality, and ensuring adherence to Basel III regulatory standards.
             """)

else:
    # Input fields for user data
    st.subheader("Enter member demographics:")
    AGE = st.number_input('Age', min_value=18, max_value=80)
    EDUCATION = st.selectbox('Education', options=['graduate_school', 'university', 'high_school', 'others'])
    MARRIAGE = st.selectbox('Marriage', options=['married', 'single', 'others'])
    SEX = st.selectbox('Sex', options=['male', 'female'])

    st.subheader("Enter member credit details:")
    LIMIT_BAL = st.number_input('Balance', min_value=20000, max_value=500000,step=1000)
    
    # Credit history for the past 6 months
    st.subheader("Credit history for the past 6 months:")
    col1 = st.columns(3)
    BILL_AMT1 = col1[0].number_input('Bill Amount 1', min_value=0, max_value=300000,step=1000)
    PAY_AMT1 = col1[1].number_input('Payment Amount 1', min_value=0, max_value=300000,step=1000)
    PAY_0 = col1[2].selectbox('Repayment Status 1', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])

    col2 = st.columns(3)
    BILL_AMT2 = col2[0].number_input('Bill Amount 2', min_value=0, max_value=300000,step=1000)
    PAY_AMT2 = col2[1].number_input('Payment Amount 2', min_value=0, max_value=300000,step=1000)
    PAY_2 = col2[2].selectbox('Repayment Status 2', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])

    col3 = st.columns(3)
    BILL_AMT3 = col3[0].number_input('Bill Amount 3', min_value=0, max_value=300000,step=1000)
    PAY_AMT3 = col3[1].number_input('Payment Amount 3', min_value=0, max_value=300000,step=1000)
    PAY_3 = col3[2].selectbox('Repayment Status 3', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])

    col4 = st.columns(3)
    BILL_AMT4 = col4[0].number_input('Bill Amount 4', min_value=0, max_value=300000,step=1000)
    PAY_AMT4 = col4[1].number_input('Payment Amount 4', min_value=0, max_value=300000,step=1000)
    PAY_4 = col4[2].selectbox('Repayment Status 4', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])

    col5 = st.columns(3)
    BILL_AMT5 = col5[0].number_input('Bill Amount 5', min_value=0, max_value=300000,step=1000)
    PAY_AMT5 = col5[1].number_input('Payment Amount 5', min_value=0, max_value=300000,step=1000)
    PAY_5 = col5[2].selectbox('Repayment Status 5', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])

    col6 = st.columns(3)
    BILL_AMT6 = col6[0].number_input('Bill Amount 6', min_value=0, max_value=300000,step=1000)
    PAY_AMT6 = col6[1].number_input('Payment Amount 6', min_value=0, max_value=300000,step=1000)
    PAY_6 = col6[2].selectbox('Repayment Status 6', options=['bill_paid', 'bill_payment_delay', 'revolving_credit'])
    
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
