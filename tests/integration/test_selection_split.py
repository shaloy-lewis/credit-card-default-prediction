"""Self-contained integration proof for the frozen train/validation view."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from credit_risk.modeling.selection_contracts import load_selection_config
from credit_risk.modeling.selection_workflow import build_selection_split


def test_selection_split_is_disjoint_complete_and_never_exposes_test_rows() -> None:
    config = load_selection_config()
    account_ids = pd.Index(np.arange(1, 24_001), name="account_id")
    target = pd.Series(
        [0] * 14_953 + [1] * 4_247 + [0] * 3_738 + [1] * 1_062,
        index=account_ids,
        dtype="int8",
    )
    governed = SimpleNamespace(
        X=pd.DataFrame({"fixture": np.zeros(24_000, dtype="int8")}, index=account_ids),
        y=target,
        assignments=pd.DataFrame({"cv_fold_r0": [1] * 19_200 + [0] * 4_800}, index=account_ids),
    )

    split = build_selection_split(governed, config)
    train_ids = set(split.X_train.index)
    validation_ids = set(split.X_validation.index)

    assert len(train_ids) == 19200
    assert len(validation_ids) == 4800
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == set(account_ids)
    assert split.y_train.value_counts().sort_index().to_dict() == {
        0: 14953,
        1: 4247,
    }
    assert split.y_validation.value_counts().sort_index().to_dict() == {
        0: 3738,
        1: 1062,
    }
    assert "partition" not in split.X_train.columns
    assert "partition" not in split.X_validation.columns
