"""Model factory, hyperparameter tuning, and evaluation.

Three candidate regressors are tuned with RandomizedSearchCV on a
TimeSeriesSplit, scored by negative MAE. The best validation-MAE model wins.
Tree-based regressors are scale-invariant, so we don't apply a StandardScaler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment]

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None  # type: ignore[assignment]

log = logging.getLogger("model")


@dataclass
class Candidate:
    name: str
    builder: Callable[[int], Any]
    param_distributions: Dict[str, Any] = field(default_factory=dict)


def _rf_builder(random_state: int):
    return RandomForestRegressor(
        n_jobs=-1,
        random_state=random_state,
    )


def _lgbm_builder(random_state: int):
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not installed")
    return LGBMRegressor(
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
        force_row_wise=True,
        verbose=-1,
    )


def _xgb_builder(random_state: int):
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed")
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def candidates(random_state: int = 42) -> Dict[str, Candidate]:
    cands: Dict[str, Candidate] = {
        "random_forest": Candidate(
            name="random_forest",
            builder=_rf_builder,
            param_distributions={
                "n_estimators": [200, 300, 500, 800],
                "max_depth": [None, 8, 12, 16, 24],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", 0.5, 0.75, 1.0],
            },
        ),
    }
    if LGBMRegressor is not None:
        cands["lightgbm"] = Candidate(
            name="lightgbm",
            builder=_lgbm_builder,
            param_distributions={
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.02, 0.05, 0.08],
                "num_leaves": [15, 31, 63],
                "min_child_samples": [30, 50, 100],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [0.0, 0.1, 1.0],
            },
        )
    if XGBRegressor is not None:
        cands["xgboost"] = Candidate(
            name="xgboost",
            builder=_xgb_builder,
            param_distributions={
                "n_estimators": [200, 400, 600],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.02, 0.05, 0.08],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "min_child_weight": [1, 5, 10],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0],
            },
        )
    return cands


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, horizon: int) -> Dict[str, float]:
    """Regression metrics + the in-domain `accuracy within ±1 day`."""
    y_pred_clipped = np.clip(np.round(y_pred), 0, horizon)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    acc1 = float(np.mean(np.abs(y_pred_clipped - y_true) <= 1))
    return {
        "mae_days": round(mae, 4),
        "rmse_days": round(rmse, 4),
        "r2": round(r2, 4),
        "accuracy_within_1day": round(acc1, 4),
    }


def tune_candidate(
    cand: Candidate,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: int = 0,
) -> Tuple[Any, Dict[str, Any], float]:
    """Randomized search with TimeSeriesSplit, scored on neg-MAE."""
    estimator = cand.builder(random_state)
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=cand.param_distributions,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=verbose,
        refit=True,
    )

    log.info("Tuning %s: %d iter × %d splits", cand.name, n_iter, n_splits)
    search.fit(X, y)
    best_cv_mae = -float(search.best_score_)
    log.info("%s best CV MAE = %.4f, params = %s", cand.name, best_cv_mae, search.best_params_)
    return search.best_estimator_, search.best_params_, best_cv_mae


def select_best(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    horizon: int,
    n_iter: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
    only: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Tune every candidate and return a result dict for the best validation-MAE model."""
    cands = candidates(random_state=random_state)
    if only:
        cands = {k: v for k, v in cands.items() if k in only}
    if not cands:
        raise RuntimeError("No candidates available — check installs (lightgbm/xgboost)")

    results: Dict[str, Dict[str, Any]] = {}
    fitted: Dict[str, Any] = {}

    for name, cand in cands.items():
        try:
            model, best_params, cv_mae = tune_candidate(
                cand,
                X_train,
                y_train,
                n_iter=n_iter,
                n_splits=n_splits,
                random_state=random_state,
            )
        except Exception as exc:
            log.exception("Tuning failed for %s: %s", name, exc)
            continue

        y_pred = model.predict(X_val)
        metrics = evaluate(y_val.to_numpy(), y_pred, horizon=horizon)
        metrics["cv_mae_days"] = round(cv_mae, 4)

        log.info(
            "%s — val MAE %.4f, RMSE %.4f, R² %.4f, acc±1 %.2f%%",
            name,
            metrics["mae_days"],
            metrics["rmse_days"],
            metrics["r2"],
            100 * metrics["accuracy_within_1day"],
        )

        fitted[name] = model
        results[name] = {
            "best_params": best_params,
            "metrics": metrics,
        }

    if not results:
        raise RuntimeError("All candidates failed to fit")

    best_name = min(results, key=lambda k: results[k]["metrics"]["mae_days"])
    log.info(
        "🏆 Best model: %s (val MAE = %.4f)",
        best_name,
        results[best_name]["metrics"]["mae_days"],
    )

    return {
        "best_name": best_name,
        "best_model": fitted[best_name],
        "all_results": results,
    }
