"""MLflow experiment tracking, parameter/metric logging, and model artifact management."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Ensure local file store or sqlite store compatibility
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


class ExperimentTracker:
    """Standardized MLflow experiment tracking wrapper for forecasting workflows."""

    def __init__(
        self,
        experiment_name: str = "retail-demand-forecast",
        tracking_uri: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.active_run: Optional[mlflow.ActiveRun] = None

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None
    ) -> mlflow.ActiveRun:
        """Start a new MLflow tracking run."""
        self.active_run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(
            f"Started MLflow run '{run_name or self.active_run.info.run_id}' in experiment '{self.experiment_name}'"
        )
        return self.active_run

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameter dictionary."""
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run found. Call start_run() first.")
        clean_params = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in params.items()}
        mlflow.log_params(clean_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics."""
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run found. Call start_run() first.")
        mlflow.log_metrics(metrics, step=step)

    def log_figure(self, figure: plt.Figure, artifact_file: str) -> None:
        """Log matplotlib figure artifact."""
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run found. Call start_run() first.")
        mlflow.log_figure(figure, artifact_file)

    def log_artifact(
        self, local_path: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """Log generic file artifact."""
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run found. Call start_run() first.")
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """Log model artifact using standard cloudpickle serialization."""
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run found. Call start_run() first.")

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                serialization_format="cloudpickle",
            )
        except Exception:
            # Fallback to direct cloudpickle / pickle
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                serialization_format="pickle",
            )

    def end_run(self) -> None:
        """End currently active MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("Ended active MLflow run.")
            self.active_run = None
