"""
Model training, validation, and calibration.

Trains regularized logistic regression and gradient boosting models
using strictly time-ordered walk-forward validation.  Selects the
best model by out-of-sample ROC-AUC, calibrates probabilities via
Platt scaling or isotonic regression, and reports all metrics honestly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config import MIN_TRAIN_SIZE, N_SPLITS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Container for trained model and its evaluation metrics."""
    model: Any
    model_name: str
    scaler: StandardScaler
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray
    feature_importance: Dict[str, float]
    oos_predictions: pd.Series
    oos_probabilities: pd.Series
    oos_actuals: pd.Series
    is_useful: bool  # True if model beats majority-class baseline
    baseline_accuracy: float
    prev_day_accuracy: float
    price_only_accuracy: Optional[float] = None
    price_only_auc: Optional[float] = None
    calibration_method: str = "sigmoid"


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _majority_baseline(y: pd.Series) -> float:
    """Accuracy of always predicting the majority class."""
    counts = y.value_counts()
    return counts.max() / len(y) if len(y) > 0 else 0.0


def _prev_day_baseline(y: pd.Series) -> float:
    """Accuracy of predicting same direction as previous day."""
    shifted = y.shift(1).dropna()
    aligned = y.iloc[1:]
    return (shifted.values == aligned.values).mean()


def _walk_forward_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_cls: Any,
    model_params: Dict,
    n_splits: int = N_SPLITS,
    calibration: str = "sigmoid",
) -> Tuple[Any, Dict[str, float], np.ndarray, pd.Series, pd.Series, pd.Series, StandardScaler]:
    """
    Walk-forward time-series cross-validation.

    Returns the model trained on the largest training set (last fold),
    along with out-of-sample metrics aggregated across all folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_preds: List[int] = []
    all_probs: List[float] = []
    all_actuals: List[int] = []
    all_indices: List[int] = []

    final_model = None
    final_scaler = None

    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < MIN_TRAIN_SIZE:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Skip if only one class in train
        if y_train.nunique() < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        base_model = model_cls(**model_params)
        base_model.fit(X_train_s, y_train)

        # Calibrate
        try:
            cal_model = CalibratedClassifierCV(
                base_model, cv="prefit", method=calibration
            )
            cal_model.fit(X_train_s, y_train)
            probs = cal_model.predict_proba(X_test_s)[:, 1]
        except Exception:
            probs = base_model.predict_proba(X_test_s)[:, 1]
            cal_model = base_model

        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_actuals.extend(y_test.values)
        all_indices.extend(test_idx)

        final_model = cal_model
        final_scaler = scaler

    # Aggregate metrics
    if not all_actuals:
        raise ValueError("Not enough data for walk-forward validation")

    all_actuals_arr = np.array(all_actuals)
    all_preds_arr = np.array(all_preds)
    all_probs_arr = np.array(all_probs)

    metrics = {
        "accuracy": accuracy_score(all_actuals_arr, all_preds_arr),
        "precision": precision_score(all_actuals_arr, all_preds_arr, zero_division=0),
        "recall": recall_score(all_actuals_arr, all_preds_arr, zero_division=0),
        "f1": f1_score(all_actuals_arr, all_preds_arr, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(all_actuals_arr, all_probs_arr)
    except ValueError:
        metrics["roc_auc"] = 0.5

    cm = confusion_matrix(all_actuals_arr, all_preds_arr, labels=[0, 1])

    oos_preds = pd.Series(all_preds_arr, index=X.index[all_indices], name="prediction")
    oos_probs = pd.Series(all_probs_arr, index=X.index[all_indices], name="probability")
    oos_actuals = pd.Series(all_actuals_arr, index=X.index[all_indices], name="actual")

    return final_model, metrics, cm, oos_preds, oos_probs, oos_actuals, final_scaler


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    X_price_only: Optional[pd.DataFrame] = None,
    y_price_only: Optional[pd.Series] = None,
) -> ModelResult:
    """
    Train logistic regression and gradient boosting, select the best
    by validation ROC-AUC, and return a ModelResult.

    Parameters
    ----------
    X, y : feature matrix and target (sentiment + price features)
    X_price_only, y_price_only : price-only baseline features/target

    Returns
    -------
    ModelResult with metrics, predictions, feature importance, and
    baseline comparisons.
    """
    candidates = [
        (
            "Logistic Regression (L2)",
            LogisticRegression,
            {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 500, "random_state": 42},
        ),
        (
            "Gradient Boosting",
            LGBMClassifier,
            {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "random_state": 42,
                "verbose": -1,
            },
        ),
    ]

    best_result: Optional[Tuple] = None
    best_auc = -1.0

    for name, cls, params in candidates:
        try:
            model, metrics, cm, preds, probs, actuals, scaler = _walk_forward_evaluate(
                X, y, cls, params
            )
            logger.info("%s — AUC: %.4f, Acc: %.4f", name, metrics["roc_auc"], metrics["accuracy"])
            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_result = (name, model, metrics, cm, preds, probs, actuals, scaler)
        except Exception as exc:
            logger.warning("Training %s failed: %s", name, exc)

    if best_result is None:
        raise ValueError("All models failed to train")

    name, model, metrics, cm, preds, probs, actuals, scaler = best_result

    # Feature importance
    importance: Dict[str, float] = {}
    try:
        # Try gradient boosting feature_importances_
        inner = model.estimator if hasattr(model, "estimator") else model
        if hasattr(inner, "calibrated_classifiers_"):
            inner = inner.calibrated_classifiers_[0].estimator
        if hasattr(inner, "feature_importances_"):
            importance = dict(zip(X.columns, inner.feature_importances_))
        elif hasattr(inner, "coef_"):
            importance = dict(zip(X.columns, np.abs(inner.coef_[0])))
    except Exception:
        importance = {c: 0.0 for c in X.columns}

    # Baselines
    baseline_acc = _majority_baseline(y)
    prev_day_acc = _prev_day_baseline(y)

    # Price-only baseline
    price_only_acc: Optional[float] = None
    price_only_auc: Optional[float] = None
    if X_price_only is not None and y_price_only is not None and len(X_price_only) >= MIN_TRAIN_SIZE:
        try:
            _, po_metrics, _, _, _, _, _ = _walk_forward_evaluate(
                X_price_only, y_price_only,
                LGBMClassifier,
                {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "random_state": 42, "verbose": -1},
            )
            price_only_acc = po_metrics["accuracy"]
            price_only_auc = po_metrics["roc_auc"]
        except Exception:
            pass

    is_useful = metrics["accuracy"] > baseline_acc + 0.01

    return ModelResult(
        model=model,
        model_name=name,
        scaler=scaler,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        roc_auc=metrics["roc_auc"],
        confusion=cm,
        feature_importance=importance,
        oos_predictions=preds,
        oos_probabilities=probs,
        oos_actuals=actuals,
        is_useful=is_useful,
        baseline_accuracy=baseline_acc,
        prev_day_accuracy=prev_day_acc,
        price_only_accuracy=price_only_acc,
        price_only_auc=price_only_auc,
    )


def predict_next(
    model_result: ModelResult,
    latest_features: pd.DataFrame,
) -> Tuple[str, float, str]:
    """
    Predict the next-day direction using the trained model.

    Returns
    -------
    direction : str
        'Bullish', 'Neutral', or 'Bearish'.
    probability : float
        Calibrated probability of the predicted direction.
    confidence : str
        'High', 'Moderate', or 'Low'.
    """
    from config import CONFIDENCE_BINS

    X_scaled = model_result.scaler.transform(latest_features)
    prob_up = model_result.model.predict_proba(X_scaled)[0, 1]
    prob_down = 1.0 - prob_up

    if prob_up > 0.55:
        direction = "Bullish"
        prob = prob_up
    elif prob_down > 0.55:
        direction = "Bearish"
        prob = prob_down
    else:
        direction = "Neutral"
        prob = max(prob_up, prob_down)

    # Confidence
    confidence = "Low"
    for label, threshold in sorted(CONFIDENCE_BINS.items(), key=lambda x: -x[1]):
        if prob >= threshold:
            confidence = label
            break

    return direction, round(prob, 4), confidence
