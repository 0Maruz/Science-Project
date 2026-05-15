"""Model factory, hyperparameter tuning, and evaluation.

Three candidate regressors are tuned with RandomizedSearchCV on a
TimeSeriesSplit. CV refits on a hybrid score (MAE minus weighted acc±1); the
winning family is chosen on the validation split with the same trade-off.
Tree-based regressors are scale-invariant, so we don't apply a StandardScaler.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
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


def hybrid_acc_weight() -> float:
    """Weight on acc±1 in CV / model selection: lower MAE is good, higher acc is good.

    Score to **maximise** (sklearn): ``-(MAE - w * acc±1)``.  So +0.10 acc is worth
    ``0.10 * w`` days of MAE.  Default ``w=3.5`` makes accuracy matter in tuning
    without totally ignoring MAE (operator-facing days error).
    """
    return float(os.getenv("TRAIN_HYBRID_ACC_WEIGHT", "3.5"))


def _hybrid_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    horizon: int,
    w: float,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Return value to **maximise** (sklearn scorer, greater is better)."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    ypc = np.clip(np.round(yp), 0, horizon)
    err = np.abs(yp - yt)
    hit = (np.abs(ypc - yt) <= 1).astype(np.float64)
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64).ravel()
        s = float(sw.sum())
        if s <= 0:
            mae = float(np.mean(err))
            acc1 = float(np.mean(hit))
        else:
            sw = sw / s
            mae = float(np.sum(sw * err))
            acc1 = float(np.sum(sw * hit))
    else:
        mae = float(np.mean(err))
        acc1 = float(np.mean(hit))
    return -(mae - w * acc1)


def make_hybrid_scorer(horizon: int, w: Optional[float] = None):
    w = hybrid_acc_weight() if w is None else w

    def _score(y_true, y_pred, sample_weight=None, **kwargs):
        return _hybrid_from_arrays(
            y_true, y_pred, horizon=horizon, w=w, sample_weight=sample_weight
        )

    return make_scorer(_score, greater_is_better=True, needs_sample_weight=True)


def selection_score(metrics: Dict[str, float], w: Optional[float] = None) -> float:
    """Lower is better — used to pick among RF / LGBM / XGB on the val split."""
    w = hybrid_acc_weight() if w is None else w
    return float(metrics["mae_days"]) - w * float(metrics["accuracy_within_1day"])


class EnsembleRegressor:
    """Averaging ensemble of fixed-hyperparameter models with different seeds.

    Wraps a list of pre-fitted regressors and exposes `predict()` (arithmetic
    mean) and `feature_importances_` (averaged). Picklable — safe for joblib.
    """

    def __init__(self, models: List[Any]) -> None:
        self.models = models

    def predict(self, X: Any) -> np.ndarray:
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        imps = [
            m.feature_importances_
            for m in self.models
            if hasattr(m, "feature_importances_")
        ]
        if not imps:
            return None
        return np.stack(imps, axis=0).mean(axis=0)


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
    # MAE objective (regression_l1): optimises directly for the metric we care
    # about (days error). Compared to MSE, L1 is less sensitive to outliers
    # (e.g. cells that burn much later than predicted), which lets the model
    # concentrate capacity on the ±1 day band where accuracy is measured.
    return LGBMRegressor(
        objective="regression_l1",
        random_state=random_state,
        n_jobs=-1,
        force_row_wise=True,
        verbose=-1,
    )


def _xgb_builder(random_state: int):
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not installed")
    # MAE objective: same rationale as LightGBM above — optimise what we measure.
    return XGBRegressor(
        objective="reg:absoluteerror",
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
        # Grid tuned for MAE objective (regression_l1).
        # Key changes vs MSE grid:
        #   • Lower learning rates (0.003–0.03) since L1 gradient is ±1 and
        #     converges more slowly than smooth MSE gradient.
        #   • More estimators (up to 2000) to compensate for slower convergence.
        #   • Wider num_leaves (15–255) — L1 splits are coarse; more leaves help
        #     per-cell variation instead of collapsing toward a global mean.
        #   • min_child_samples as low as 5 on dense grids when reg allows.
        #   • reg_lambda includes 0.0 so search is not always strongly smoothed.
        #   • path_smooth spans fine→heavy; max=1.0 alone often narrows preds.
        cands["lightgbm"] = Candidate(
            name="lightgbm",
            builder=_lgbm_builder,
            param_distributions={
                "n_estimators": [500, 800, 1200, 1500, 2000],
                "learning_rate": [0.003, 0.005, 0.01, 0.02, 0.03],
                "num_leaves": [15, 31, 47, 63, 95, 127, 191, 255],
                "max_depth": [-1, 5, 6, 8, 10, 12, 14],
                "min_child_samples": [5, 10, 15, 20, 50, 80, 120, 200],
                "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
                "subsample_freq": [1, 5, 10],
                "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
                "reg_alpha": [0.0, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0],
                "reg_lambda": [0.0, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0],
                "min_split_gain": [0.0, 0.01, 0.05, 0.1, 0.2],
                "extra_trees": [True, False],
                "path_smooth": [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
            },
        )
    if XGBRegressor is not None:
        # XGBoost grid for MAE objective (reg:absoluteerror).
        # Key additions:
        #   • Lower learning rates for L1 convergence.
        #   • max_delta_step removed (helps MSE, not L1).
        #   • Added grow_policy="lossguide" to allow best-first tree growth
        #     which works better with L1 loss that has flat gradients.
        #   • Light min_child_weight + moderate depth — empirically the old grid
        #     favoured very high min_child_weight / reg_alpha, which collapsed
        #     predictions to a narrow band (bad acc±1 vs baseline).
        cands["xgboost"] = Candidate(
            name="xgboost",
            builder=_xgb_builder,
            param_distributions={
                "n_estimators": [500, 800, 1200, 1500, 2000],
                "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "learning_rate": [0.003, 0.005, 0.01, 0.02, 0.03],
                "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
                "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
                "colsample_bylevel": [0.6, 0.7, 0.8, 1.0],
                "min_child_weight": [1, 3, 5, 10, 20, 30, 50, 80],
                "reg_alpha": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0],
                "reg_lambda": [0.5, 1.0, 5.0, 10.0, 25.0, 50.0],
                "gamma": [0.0, 0.1, 0.3, 1.0, 3.0],
                "grow_policy": ["depthwise", "lossguide"],
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
    sample_weight: Optional[np.ndarray] = None,
    horizon: int = 7,
) -> Tuple[Any, Dict[str, Any], float]:
    """Randomized search with TimeSeriesSplit.

    Primary CV objective is a **hybrid** score (MAE minus weighted acc±1) so
    hyperparameters are not pushed solely toward the mean predictor.  We still
    record ``cv_mae_days`` as the mean CV MAE at the chosen hyperparameters.
    """
    estimator = cand.builder(random_state)
    cv = TimeSeriesSplit(n_splits=n_splits)
    w = hybrid_acc_weight()
    hybrid_sc = make_hybrid_scorer(horizon, w)
    scoring: Union[str, Dict[str, Any]] = {
        "mae": "neg_mean_absolute_error",
        "hybrid": hybrid_sc,
    }
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=cand.param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit="hybrid",
        cv=cv,
        random_state=random_state,
        n_jobs=1,
        verbose=verbose,
    )

    log.info("Tuning %s: %d iter × %d splits (CV refit=hybrid, w_acc=%.2f)", cand.name, n_iter, n_splits, w)
    fit_kwargs: Dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    search.fit(X, y, **fit_kwargs)
    best_idx = int(search.best_index_)
    best_cv_mae = -float(search.cv_results_["mean_test_mae"][best_idx])
    best_hybrid = float(search.cv_results_["mean_test_hybrid"][best_idx])
    log.info(
        "%s best CV MAE = %.4f, CV hybrid = %.4f, params = %s",
        cand.name,
        best_cv_mae,
        best_hybrid,
        search.best_params_,
    )
    return search.best_estimator_, search.best_params_, best_cv_mae


def build_ensemble(
    cand: Candidate,
    best_params: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_ensemble: int = 7,
    base_seed: int = 42,
    sample_weight: Optional[np.ndarray] = None,
) -> EnsembleRegressor:
    """Fit `n_ensemble` copies of `cand` at `best_params` with different random seeds.

    Each model sees the same training data but uses a different seed for
    feature/sample subsampling, so their errors are weakly correlated.
    Averaging reduces prediction variance without touching bias — typically
    yields 3–8% MAE improvement over a single model. Increased to 7 from 5
    for better variance reduction on the seasonal distribution shift.
    """
    models: List[Any] = []
    for i in range(n_ensemble):
        seed = base_seed + i * 37
        m = cand.builder(seed)
        m.set_params(**best_params)
        if sample_weight is not None:
            m.fit(X, y, sample_weight=sample_weight)
        else:
            m.fit(X, y)
        models.append(m)
    log.info(
        "Ensemble: trained %d %s models (seeds %d…%d)",
        n_ensemble, cand.name, base_seed, base_seed + (n_ensemble - 1) * 37,
    )
    return EnsembleRegressor(models)


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
    sample_weight: Optional[np.ndarray] = None,
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
                sample_weight=sample_weight,
                horizon=horizon,
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

    w_sel = hybrid_acc_weight()
    best_name = min(results, key=lambda k: selection_score(results[k]["metrics"], w_sel))
    log.info(
        "🏆 Best model: %s (val selection = MAE − %.2f×acc, score=%.4f; val MAE=%.4f, val acc±1=%.2f%%)",
        best_name,
        w_sel,
        selection_score(results[best_name]["metrics"], w_sel),
        results[best_name]["metrics"]["mae_days"],
        100 * results[best_name]["metrics"]["accuracy_within_1day"],
    )

    return {
        "best_name": best_name,
        "best_model": fitted[best_name],
        "all_results": results,
    }