"""MLflow Model Registry manager: Champion vs Challenger evaluation and automated promotion."""

import os
from typing import Dict, Optional, Tuple
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger(__name__)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


class ModelRegistryManager:
    """Manages MLflow model registration, version tagging, and Champion vs Challenger promotion."""

    def __init__(
        self,
        model_name: str = "retail-demand-champion",
        tracking_uri: Optional[str] = None,
    ):
        self.model_name = model_name
        self.tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_model_version(
        self,
        run_id: str,
        artifact_path: str = "model",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a model run into the MLflow Model Registry and return its version."""
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info(
            f"Registering model from {model_uri} into registry under '{self.model_name}'..."
        )
        mv = mlflow.register_model(model_uri=model_uri, name=self.model_name, tags=tags)
        if description:
            self.client.update_model_version(
                name=self.model_name,
                version=mv.version,
                description=description,
            )
        return str(mv.version)

    def set_champion(self, version: str) -> None:
        """Assign the 'champion' alias to a specific registered model version."""
        logger.info(f"Setting model version {version} as 'champion' alias...")
        self.client.set_registered_model_alias(
            name=self.model_name,
            alias="champion",
            version=version,
        )

    def evaluate_and_promote(
        self,
        new_version: str,
        new_wrmsse: float,
        champion_metric_key: str = "mean_wrmsse",
    ) -> Tuple[bool, str]:
        """Compare new challenger model against existing champion and promote if superior.

        Returns:
            (promoted: bool, message: str)
        """
        try:
            champion_mv = self.client.get_model_version_by_alias(self.model_name, "champion")
            champion_run = self.client.get_run(champion_mv.run_id)
            champion_wrmsse = champion_run.data.metrics.get(champion_metric_key, float("inf"))

            if new_wrmsse < champion_wrmsse:
                self.set_champion(new_version)
                msg = (
                    f"Challenger v{new_version} ({champion_metric_key}={new_wrmsse:.4f}) beat "
                    f"Champion v{champion_mv.version} ({champion_metric_key}={champion_wrmsse:.4f}). "
                    f"Promoted to Champion!"
                )
                logger.info(msg)
                return True, msg
            else:
                msg = (
                    f"Challenger v{new_version} ({champion_metric_key}={new_wrmsse:.4f}) did not beat "
                    f"Champion v{champion_mv.version} ({champion_metric_key}={champion_wrmsse:.4f}). "
                    f"Retaining existing Champion."
                )
                logger.info(msg)
                return False, msg

        except Exception as e:
            # If no champion exists yet, promote new version as initial champion
            logger.info(
                f"No existing champion found ({e}). Promoting v{new_version} as initial Champion."
            )
            self.set_champion(new_version)
            return True, f"Promoted v{new_version} as initial Champion model."
