"""Temporal rolling-window backtesting and cross-validation framework."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import WRMSSEEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TemporalFold:
    """Container for a single backtest fold split."""

    fold_idx: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    cutoff_date: pd.Timestamp
    val_start_date: pd.Timestamp
    val_end_date: pd.Timestamp


class RollingWindowSplitter:
    """Generates temporal cross-validation folds with expanding or sliding windows."""

    def __init__(
        self,
        horizon: int = 28,
        n_splits: int = 3,
        step_size: Optional[int] = None,
        date_col: str = "date",
        expanding: bool = True,
        min_train_days: int = 90,
    ):
        self.horizon = horizon
        self.n_splits = n_splits
        self.step_size = step_size or horizon
        self.date_col = date_col
        self.expanding = expanding
        self.min_train_days = min_train_days

    def split(self, df: pd.DataFrame) -> Generator[TemporalFold, None, None]:
        """Generate (train, validation) temporal splits in chronological order."""
        unique_dates = pd.to_datetime(df[self.date_col].drop_duplicates().sort_values().values)
        total_days = len(unique_dates)

        required_days = self.min_train_days + self.horizon + (self.n_splits - 1) * self.step_size
        if total_days < required_days:
            logger.warning(
                f"Total days ({total_days}) is less than recommended ({required_days}). "
                f"Adjusting min_train_days to fit {self.n_splits} folds."
            )
            available_train = total_days - (self.horizon + (self.n_splits - 1) * self.step_size)
            self.min_train_days = max(14, available_train)

        # Work backwards from the most recent date to create fold cutoffs
        cutoffs = []
        for i in range(self.n_splits):
            end_idx = total_days - (i * self.step_size)
            cutoff_idx = end_idx - self.horizon
            if cutoff_idx >= self.min_train_days:
                cutoffs.append(cutoff_idx)

        cutoffs.reverse()  # Chronological order

        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy[self.date_col]):
            df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col])

        for fold_idx, c_idx in enumerate(cutoffs):
            cutoff_date = unique_dates[c_idx - 1]
            val_start_date = unique_dates[c_idx]
            val_end_date = unique_dates[c_idx + self.horizon - 1]

            if self.expanding:
                train_mask = df_copy[self.date_col] <= cutoff_date
            else:
                train_start_date = unique_dates[max(0, c_idx - self.min_train_days)]
                train_mask = (df_copy[self.date_col] >= train_start_date) & (
                    df_copy[self.date_col] <= cutoff_date
                )

            val_mask = (df_copy[self.date_col] >= val_start_date) & (
                df_copy[self.date_col] <= val_end_date
            )

            train_df = df_copy[train_mask]
            val_df = df_copy[val_mask]

            yield TemporalFold(
                fold_idx=fold_idx + 1,
                train_df=train_df,
                val_df=val_df,
                cutoff_date=cutoff_date,
                val_start_date=val_start_date,
                val_end_date=val_end_date,
            )


def run_backtest(
    model: Any,
    df: pd.DataFrame,
    horizon: int = 28,
    n_splits: int = 3,
    date_col: str = "date",
    target_col: str = "sales",
    price_col: Optional[str] = "sell_price",
) -> Tuple[Dict[str, float], pd.DataFrame, List[Dict[str, Any]]]:
    """Execute temporal rolling-window backtest on a forecaster model.

    Returns:
        Tuple[aggregate_metrics, out_of_fold_predictions, per_fold_summary]
    """
    splitter = RollingWindowSplitter(
        horizon=horizon,
        n_splits=n_splits,
        date_col=date_col,
    )

    oof_predictions_list: List[pd.DataFrame] = []
    fold_summaries: List[Dict[str, Any]] = []

    logger.info(f"Starting temporal backtest with {n_splits} folds (horizon={horizon}d)...")

    for fold in splitter.split(df):
        logger.info(
            f"--- Fold {fold.fold_idx}/{n_splits} | Train cutoff: {fold.cutoff_date.strftime('%Y-%m-%d')} "
            f"| Val: {fold.val_start_date.strftime('%Y-%m-%d')} to {fold.val_end_date.strftime('%Y-%m-%d')} ---"
        )

        t0 = time.time()
        # 1. Fit model on training fold
        model.fit(fold.train_df, target_col=target_col, date_col=date_col)
        fit_time = time.time() - t0

        # 2. Generate predictions on validation horizon
        t0 = time.time()
        preds = model.predict(fold.val_df, horizon=horizon)
        pred_time = time.time() - t0

        # Standardize prediction DataFrame
        if isinstance(preds, (np.ndarray, list)):
            pred_df = fold.val_df[["id", date_col]].copy()
            pred_df["y_pred"] = preds
        elif isinstance(preds, pd.DataFrame):
            pred_df = preds.copy()
            if "y_pred" not in pred_df.columns and target_col in pred_df.columns:
                pred_df["y_pred"] = pred_df[target_col]
        else:
            raise ValueError(f"Unsupported prediction return type: {type(preds)}")

        pred_df["fold"] = fold.fold_idx

        # 3. Evaluate with official M5 WRMSSE
        evaluator = WRMSSEEvaluator(
            train_df=fold.train_df,
            date_col=date_col,
            target_col=target_col,
            price_col=price_col,
        )
        fold_metrics = evaluator.score(fold.val_df, pred_df)

        fold_summary = {
            "fold": fold.fold_idx,
            "cutoff_date": str(fold.cutoff_date.date()),
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time,
            "wrmsse": fold_metrics["wrmsse"],
            "rmsse_mean": fold_metrics["rmsse_mean"],
            "wape": fold_metrics["wape"],
            "mae": fold_metrics["mae"],
            "rmse": fold_metrics["rmse"],
        }
        fold_summaries.append(fold_summary)
        oof_predictions_list.append(pred_df)

        logger.info(
            f"Fold {fold.fold_idx} Score: WRMSSE={fold_metrics['wrmsse']:.4f} | WAPE={fold_metrics['wape']:.4f} "
            f"| RMSE={fold_metrics['rmse']:.4f} (Fit: {fit_time:.2f}s, Pred: {pred_time:.2f}s)"
        )

    oof_df = pd.concat(oof_predictions_list, ignore_index=True)

    # Compute overall average metrics across folds
    agg_metrics = {
        "mean_wrmsse": float(np.mean([f["wrmsse"] for f in fold_summaries])),
        "std_wrmsse": float(np.std([f["wrmsse"] for f in fold_summaries])),
        "mean_wape": float(np.mean([f["wape"] for f in fold_summaries])),
        "mean_rmse": float(np.mean([f["rmse"] for f in fold_summaries])),
        "mean_mae": float(np.mean([f["mae"] for f in fold_summaries])),
        "total_fit_time_sec": float(np.sum([f["fit_time_sec"] for f in fold_summaries])),
        "total_pred_time_sec": float(np.sum([f["pred_time_sec"] for f in fold_summaries])),
    }

    logger.info(
        f"=== Backtest Completed: Mean WRMSSE={agg_metrics['mean_wrmsse']:.4f} "
        f"(+/- {agg_metrics['std_wrmsse']:.4f}) | Mean WAPE={agg_metrics['mean_wape']:.4f} ==="
    )

    return agg_metrics, oof_df, fold_summaries
