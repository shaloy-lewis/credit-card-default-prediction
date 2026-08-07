import streamlit as st

from credit_risk.pipeline.prediction_pipeline import CustomData, PredictPipeline

# Initialize prediction pipeline
predict_pipeline = PredictPipeline()

# Streamlit app title
st.title("Credit Default Prediction")

page = st.sidebar.selectbox(
    "Page Navigation",
    [
        "Problem statement",
        "Predictor",
    ],
)

st.sidebar.markdown("""---""")
st.sidebar.write("Created by [Shaloy Lewis](https://www.linkedin.com/in/shaloy-lewis/)")

if page == "Problem statement":
    st.write(
        """This demonstration estimates next-month default risk for existing
credit-card accounts using the preceding six months of account history. The
proposed product uses calibrated scores to prioritise a capacity-constrained,
human-reviewed support or collections queue after each monthly billing cycle.

The public UCI data is a historical Taiwanese sample and does not establish
validity for Indian customers, current lender portfolios, realised business
impact, or regulatory compliance. This demonstration must not be used to approve
or deny credit, change account terms, or initiate an adverse action."""
    )

else:
    # Input fields for user data
    st.subheader("Enter member demographics:")
    col = st.columns(4)
    AGE = col[0].number_input("Age", min_value=18, max_value=80)
    EDUCATION = col[1].selectbox(
        "Education", options=["graduate_school", "university", "high_school", "others"]
    )
    MARRIAGE = col[2].selectbox("Marriage", options=["married", "single", "others"])
    SEX = col[3].selectbox("Sex", options=["male", "female"])

    st.subheader("Enter member credit details:")
    LIMIT_BAL = st.number_input("Balance", min_value=20000, max_value=500000, step=1000)

    # Credit history for the past 6 months
    st.subheader("Credit history for the past 6 months:")
    col1 = st.columns(3)
    BILL_AMT1 = col1[0].number_input("Bill Amount 1", min_value=0, max_value=300000, step=1000)
    PAY_AMT1 = col1[1].number_input("Payment Amount 1", min_value=0, max_value=300000, step=1000)
    PAY_0 = col1[2].selectbox(
        "Repayment Status 1", options=["bill_paid", "bill_payment_delay", "revolving_credit"]
    )

    col2 = st.columns(3)
    BILL_AMT2 = col2[0].number_input("Bill Amount 2", min_value=0, max_value=300000, step=1000)
    PAY_AMT2 = col2[1].number_input("Payment Amount 2", min_value=0, max_value=300000, step=1000)
    PAY_2 = col2[2].selectbox(
        "Repayment Status 2", options=["bill_paid", "bill_payment_delay", "revolving_credit"]
    )

    col3 = st.columns(3)
    BILL_AMT3 = col3[0].number_input("Bill Amount 3", min_value=0, max_value=300000, step=1000)
    PAY_AMT3 = col3[1].number_input("Payment Amount 3", min_value=0, max_value=300000, step=1000)
    PAY_3 = col3[2].selectbox(
        "Repayment Status 3", options=["bill_paid", "bill_payment_delay", "revolving_credit"]
    )

    col4 = st.columns(3)
    BILL_AMT4 = col4[0].number_input("Bill Amount 4", min_value=0, max_value=300000, step=1000)
    PAY_AMT4 = col4[1].number_input("Payment Amount 4", min_value=0, max_value=300000, step=1000)
    PAY_4 = col4[2].selectbox(
        "Repayment Status 4", options=["bill_paid", "bill_payment_delay", "revolving_credit"]
    )

    col5 = st.columns(3)
    BILL_AMT5 = col5[0].number_input("Bill Amount 5", min_value=0, max_value=300000, step=1000)
    PAY_AMT5 = col5[1].number_input("Payment Amount 5", min_value=0, max_value=300000, step=1000)
    PAY_5 = col5[2].selectbox(
        "Repayment Status 5", options=["bill_paid", "bill_payment_delay", "revolving_credit"]
    )

    col6 = st.columns(3)
    BILL_AMT6 = col6[0].number_input("Bill Amount 6", min_value=0, max_value=300000, step=1000)
    PAY_AMT6 = col6[1].number_input("Payment Amount 6", min_value=0, max_value=300000, step=1000)
    PAY_6 = col6[2].selectbox(
        "Repayment Status 6", options=["bill_paid", "bill_payment_delay", "revolving_credit"]
    )

    # Button to submit data and get predictions
    if st.button("Predict"):
        # Each select box has non-empty options and therefore returns a string.
        assert EDUCATION is not None
        assert MARRIAGE is not None
        assert SEX is not None
        assert PAY_0 is not None
        assert PAY_2 is not None
        assert PAY_3 is not None
        assert PAY_4 is not None
        assert PAY_5 is not None
        assert PAY_6 is not None

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
            PAY_6=PAY_6,
        )

        # Convert the user inputs to a DataFrame
        input_df = custom_data.get_data_as_dataframe()

        # Get predictions
        predictions = predict_pipeline.predict(input_df)

        # Display the prediction results
        st.subheader("Prediction Results:")
        st.write(f"Probability of default: {predictions[0][1]:.4f}")
