"""Read-only integration proof for the frozen train/validation view."""

from __future__ import annotations

from credit_risk.modeling.dataset import load_governed_development_data


def test_selection_split_is_disjoint_complete_and_never_exposes_test_rows() -> None:
    governed = load_governed_development_data()
    validation = governed.assignments["cv_fold_r0"].eq(0)
    train_ids = set(governed.X.index[~validation])
    validation_ids = set(governed.X.index[validation])

    assert len(train_ids) == 19200
    assert len(validation_ids) == 4800
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == set(governed.X.index)
    assert governed.y.loc[list(train_ids)].value_counts().sort_index().to_dict() == {
        0: 14953,
        1: 4247,
    }
    assert governed.y.loc[list(validation_ids)].value_counts().sort_index().to_dict() == {
        0: 3738,
        1: 1062,
    }
    assert len(governed.X) == 24000
    assert "partition" not in governed.X.columns
