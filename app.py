from typing import cast

import pandas as pd
import streamlit as st

from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.serving import SelectedPredictPipeline

predict_pipeline = SelectedPredictPipeline()

st.title("Credit Risk Early-Warning Demo")

page = st.sidebar.selectbox("Page Navigation", ["Problem statement", "Predictor"])
st.sidebar.markdown("""---""")
st.sidebar.write("Created by [Shaloy Lewis](https://www.linkedin.com/in/shaloy-lewis/)")

if page == "Problem statement":
    st.write(
        """This demonstration estimates next-month default risk for existing
credit-card accounts using the preceding six months of operational account
history. Scores can support a capacity-constrained, human-reviewed service or
collections queue after each monthly billing cycle.

The public UCI data is a historical Taiwanese sample and does not establish
validity for Indian customers, current lender portfolios, realised business
impact, or regulatory compliance. This demonstration must not be used to approve
or deny credit, change account terms, or initiate an adverse action."""
    )
else:
    st.subheader("Current credit facility")
    credit_limit = st.number_input(
        "Credit limit (NTD)", min_value=1, max_value=2_000_000, value=100_000, step=1_000
    )

    st.subheader("Six-month operational history")
    st.caption("Repayment status uses the governed source codes from -2 through 9.")
    values: dict[str, int] = {"credit_limit_ntd": int(credit_limit)}
    for lag in range(6):
        columns = st.columns(3)
        values[f"repayment_status_lag_{lag}"] = int(
            cast(
                int,
                columns[0].selectbox(
                    f"Repayment status lag {lag}",
                    options=list(range(-2, 10)),
                    index=2,
                    key=f"status_{lag}",
                ),
            )
        )
        values[f"bill_amount_ntd_lag_{lag}"] = int(
            columns[1].number_input(
                f"Bill amount lag {lag}",
                min_value=-1_000_000,
                max_value=2_000_000,
                value=4_000,
                step=1_000,
                key=f"bill_{lag}",
            )
        )
        values[f"payment_amount_ntd_lag_{lag}"] = int(
            columns[2].number_input(
                f"Payment amount lag {lag}",
                min_value=0,
                max_value=2_000_000,
                value=1_500,
                step=500,
                key=f"payment_{lag}",
            )
        )

    if st.button("Predict"):
        features = pd.DataFrame([values], columns=PREDICTOR_COLUMNS)
        probability, band = predict_pipeline.predict(features)
        st.subheader("Prediction result")
        st.metric("Probability of default", f"{probability:.4f}")
        st.write(f"Risk band: **{band}**")
        st.caption("The band is a ranking aid, not an automated credit decision.")
