"""Cell-level holdout evaluation, RUL regression, and anomaly ranking."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from cellscope.data import ANOMALY_FEATURES, FEATURES, make_demo_data


@dataclass
class CellScopeModel:
    rul_model: HistGradientBoostingRegressor
    anomaly_model: IsolationForest
    evaluation: dict[str, float]
    train_cells: list[str]
    test_cells: list[str]


def fit_demo_model(seed: int = 17) -> tuple[CellScopeModel, pd.DataFrame]:
    data = make_demo_data(seed=31)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_index, test_index = next(splitter.split(data, groups=data["cell_id"]))
    train = data.iloc[train_index].copy()
    test = data.iloc[test_index].copy()

    rul_model = HistGradientBoostingRegressor(
        learning_rate=0.07, max_iter=150, max_leaf_nodes=18,
        l2_regularization=1.5, random_state=seed,
    )
    rul_model.fit(train[FEATURES], train["true_remaining_cycles"])
    test_prediction = np.maximum(0, rul_model.predict(test[FEATURES]))
    rul_mae = float(mean_absolute_error(test["true_remaining_cycles"], test_prediction))

    anomaly_model = IsolationForest(
        n_estimators=180, contamination=0.015, random_state=seed, n_jobs=-1
    )
    normal_train = train.loc[~train["anomaly_truth"]]
    anomaly_model.fit(normal_train[ANOMALY_FEATURES])
    anomaly_prediction = anomaly_model.predict(test[ANOMALY_FEATURES]) == -1
    anomaly_truth = test["anomaly_truth"].to_numpy(dtype=bool)
    precision = float(precision_score(anomaly_truth, anomaly_prediction, zero_division=0))
    recall = float(recall_score(anomaly_truth, anomaly_prediction, zero_division=0))

    model = CellScopeModel(
        rul_model=rul_model,
        anomaly_model=anomaly_model,
        evaluation={
            "heldout_rul_mae_cycles": rul_mae,
            "anomaly_precision": precision,
            "anomaly_recall": recall,
        },
        train_cells=sorted(train["cell_id"].astype(str).unique()),
        test_cells=sorted(test["cell_id"].astype(str).unique()),
    )
    data["estimated_remaining_cycles"] = np.maximum(
        0, rul_model.predict(data[FEATURES])
    ).round(1)
    data["anomaly_score"] = -anomaly_model.decision_function(data[ANOMALY_FEATURES])
    data["anomaly_flag"] = anomaly_model.predict(data[ANOMALY_FEATURES]) == -1
    return model, data


def score_uploaded(model: CellScopeModel, data: pd.DataFrame) -> pd.DataFrame:
    scored = data.copy()
    scored["estimated_remaining_cycles"] = np.maximum(
        0, model.rul_model.predict(scored[FEATURES])
    ).round(1)
    scored["anomaly_score"] = -model.anomaly_model.decision_function(scored[ANOMALY_FEATURES])
    scored["anomaly_flag"] = model.anomaly_model.predict(scored[ANOMALY_FEATURES]) == -1
    return scored
