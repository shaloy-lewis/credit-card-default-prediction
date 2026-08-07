"""Characterisation tests for the existing feature transformations."""

import pandas as pd

from credit_risk.utils.utils import preprocess_data


def test_preprocess_data_preserves_legacy_feature_contract() -> None:
    raw = pd.DataFrame(
        {
            "MARRIAGE": [0],
            "SEX": [2],
            "EDUCATION": [5],
            "PAY_0": [2],
            "PAY_2": [0],
            "PAY_3": [-1],
            "PAY_4": [1],
            "PAY_5": [-2],
            "PAY_6": [0],
            "BILL_AMT1": [100.0],
            "BILL_AMT2": [200.0],
            "BILL_AMT3": [300.0],
            "BILL_AMT4": [400.0],
            "BILL_AMT5": [500.0],
            "BILL_AMT6": [600.0],
        }
    )

    transformed = preprocess_data(raw.copy())

    assert transformed.loc[0, "MARRIAGE"] == "others"
    assert transformed.loc[0, "SEX"] == "female"
    assert transformed.loc[0, "EDUCATION"] == "others"
    assert transformed.loc[0, "PAY_0"] == "bill_payment_delay"
    assert transformed.loc[0, "PAY_2"] == "revolving_credit"
    assert transformed.loc[0, "PAY_3"] == "bill_paid"
    assert transformed.loc[0, "BILL_AMT_AVG_6M"] == 350.0
    assert not any(
        column.startswith("BILL_AMT") and column != "BILL_AMT_AVG_6M" for column in transformed
    )
