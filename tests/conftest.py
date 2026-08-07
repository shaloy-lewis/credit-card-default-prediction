"""Shared fixtures for unit and integration tests."""

import pytest


@pytest.fixture()
def readme_prediction_payload() -> dict[str, int | float | str]:
    """Return the documented request used to guard legacy artifact compatibility."""
    return {
        "LIMIT_BAL": 1_000_000,
        "AGE": 29,
        "BILL_AMT1": 4_000,
        "BILL_AMT2": 4_000,
        "BILL_AMT3": 4_000,
        "BILL_AMT4": 4_000,
        "BILL_AMT5": 4_000,
        "BILL_AMT6": 4_000,
        "PAY_AMT1": 1_500,
        "PAY_AMT2": 1_500,
        "PAY_AMT3": 1_500,
        "PAY_AMT4": 1_500,
        "PAY_AMT5": 1_500,
        "PAY_AMT6": 1_500,
        "EDUCATION": "graduate_school",
        "MARRIAGE": "married",
        "SEX": "female",
        "PAY_0": "bill_payment_delay",
        "PAY_2": "revolving_credit",
        "PAY_3": "bill_paid",
        "PAY_4": "bill_paid",
        "PAY_5": "bill_paid",
        "PAY_6": "bill_paid",
    }
