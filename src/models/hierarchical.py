"""Hierarchical demand reconciliation: Summing matrix construction, Bottom-Up, Top-Down, and MinT."""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.logger import get_logger

logger = get_logger(__name__)

# M5 standard 12 hierarchy levels
M5_HIERARCHY_LEVELS = [
    ["total"],
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id"],
    ["state_id", "item_id"],
    ["id"],  # Level 12: bottom level (store_id x item_id)
]


def aggregate_hierarchy(
    df: pd.DataFrame,
    hierarchy_levels: Optional[List[List[str]]] = None,
    target_col: str = "sales",
    date_col: str = "date",
    bottom_id_col: str = "id",
) -> Tuple[pd.DataFrame, pd.DataFrame, sparse.csr_matrix, List[str], List[str]]:
    """Construct hierarchical dataset across all aggregation levels and build Summing Matrix S.

    Returns:
        hierarchical_df: DataFrame with columns [node_id, date, target_col, level_name]
        hierarchy_metadata: DataFrame mapping bottom nodes to each parent node
        S: Sparse Summing Matrix (n_nodes x n_bottom) such that y_all = S @ y_bottom
        all_nodes: List of all node identifiers (top-down ordered)
        bottom_nodes: List of bottom-level series identifiers
    """
    if hierarchy_levels is None:
        # Default 5-level hierarchy: Total, State, Store, Category, Bottom (ID)
        hierarchy_levels = [
            ["total"],
            ["state_id"],
            ["store_id"],
            ["cat_id"],
            ["id"],
        ]

    logger.info(f"Aggregating dataset across {len(hierarchy_levels)} hierarchy levels...")
    df_copy = df.copy()
    df_copy["total"] = "Total"

    bottom_nodes = sorted(df_copy[bottom_id_col].unique().tolist())
    n_bottom = len(bottom_nodes)

    # Map bottom nodes to their hierarchy attributes
    attr_cols = list({col for lvl in hierarchy_levels for col in lvl if col in df_copy.columns})
    bottom_meta = (
        df_copy[[bottom_id_col] + attr_cols]
        .drop_duplicates(subset=[bottom_id_col])
        .set_index(bottom_id_col)
        .reindex(bottom_nodes)
        .reset_index()
    )

    level_dfs = []
    all_node_names: List[str] = []
    s_rows: List[np.ndarray] = []

    for lvl in hierarchy_levels:
        lvl_name = "_x_".join(lvl)
        # Check if required columns exist
        if not all(col in df_copy.columns for col in lvl):
            continue

        if lvl == ["total"]:
            grouped = df_copy.groupby(date_col, observed=False)[target_col].sum().reset_index()
            grouped["node_id"] = "Total"
            grouped["level"] = "Total"
            level_dfs.append(grouped[[date_col, "node_id", target_col, "level"]])

            all_node_names.append("Total")
            s_rows.append(np.ones((1, n_bottom), dtype=np.float32))
        else:
            df_copy["_grp_id"] = df_copy[lvl].astype(str).agg("/".join, axis=1)
            grouped = (
                df_copy.groupby([date_col, "_grp_id"], observed=False)[target_col]
                .sum()
                .reset_index()
            )
            grouped = grouped.rename(columns={"_grp_id": "node_id"})
            grouped["level"] = lvl_name
            level_dfs.append(grouped[[date_col, "node_id", target_col, "level"]])

            # Unique nodes at this level
            unique_nodes_at_lvl = sorted(grouped["node_id"].unique().tolist())
            all_node_names.extend(unique_nodes_at_lvl)

            # Build S rows for these nodes
            bottom_meta["_grp_id"] = bottom_meta[lvl].astype(str).agg("/".join, axis=1)
            for node in unique_nodes_at_lvl:
                indicator = (bottom_meta["_grp_id"] == node).values.astype(np.float32)
                s_rows.append(indicator.reshape(1, -1))

    full_hier_df = pd.concat(level_dfs, ignore_index=True)
    full_hier_df[date_col] = pd.to_datetime(full_hier_df[date_col])

    S_mat = np.vstack(s_rows)
    S_sparse = sparse.csr_matrix(S_mat)

    logger.info(
        f"Summing Matrix constructed: {S_mat.shape[0]} total nodes x {n_bottom} bottom nodes"
    )
    return full_hier_df, bottom_meta, S_sparse, all_node_names, bottom_nodes


class HierarchicalReconciler:
    """Reconciles multi-level hierarchical forecasts for mathematical coherence.

    Supported methods:
        - "bottom_up": Aggregates bottom level base forecasts up the tree.
        - "top_down": Disaggregates top level forecast down using historical volume proportions.
        - "ols": Ordinary Least Squares MinT reconciliation (W = I).
        - "wls_struct": Structural Weighted Least Squares MinT (W = diag(S @ 1)).
        - "wls_var": Variance Weighted Least Squares MinT (W = diag(residual variances)).
        - "mint_shrink": Minimum Trace with Ledoit-Wolf shrinkage covariance estimation.
    """

    def __init__(
        self,
        method: Literal[
            "bottom_up", "top_down", "ols", "wls_struct", "wls_var", "mint_shrink"
        ] = "wls_struct",
    ):
        self.method = method
        self.P_matrix: Optional[np.ndarray] = None
        self.S_matrix: Optional[np.ndarray] = None
        self.all_nodes: Optional[List[str]] = None
        self.bottom_nodes: Optional[List[str]] = None
        self.node_to_idx: Optional[Dict[str, int]] = None
        self.is_fitted = False

    def fit(
        self,
        S: Union[np.ndarray, sparse.csr_matrix],
        all_nodes: List[str],
        bottom_nodes: List[str],
        residuals: Optional[np.ndarray] = None,
    ) -> "HierarchicalReconciler":
        """Compute the projection matrix P such that reconciled bottom forecasts are y_tilde_b = P @ y_hat."""
        self.S_matrix = S.toarray() if sparse.issparse(S) else np.asarray(S, dtype=np.float64)
        self.all_nodes = all_nodes
        self.bottom_nodes = bottom_nodes
        self.node_to_idx = {node: i for i, node in enumerate(all_nodes)}

        n_total, n_bottom = self.S_matrix.shape
        S = self.S_matrix

        if self.method == "bottom_up":
            # P picks only the bottom rows of the base forecasts
            # Bottom nodes are the last n_bottom rows of S
            P = np.zeros((n_bottom, n_total), dtype=np.float64)
            P[:, -n_bottom:] = np.eye(n_bottom)
            self.P_matrix = P

        elif self.method == "ols":
            # W = I => P = (S^T S)^-1 S^T
            STS = S.T @ S
            P = np.linalg.pinv(STS) @ S.T
            self.P_matrix = P

        elif self.method == "wls_struct":
            # W = diag(S @ 1) -> structural counts
            struct_weights = S @ np.ones(n_bottom, dtype=np.float64)
            struct_weights = np.maximum(struct_weights, 1.0)
            inv_W = np.diag(1.0 / struct_weights)
            # P = (S^T W^-1 S)^-1 S^T W^-1
            ST_invW = S.T @ inv_W
            P = np.linalg.pinv(ST_invW @ S) @ ST_invW
            self.P_matrix = P

        elif self.method == "wls_var":
            if residuals is not None and residuals.shape[0] == n_total:
                variances = np.var(residuals, axis=1)
                variances = np.maximum(variances, 1e-4)
                inv_W = np.diag(1.0 / variances)
            else:
                inv_W = np.eye(n_total)
            ST_invW = S.T @ inv_W
            P = np.linalg.pinv(ST_invW @ S) @ ST_invW
            self.P_matrix = P

        elif self.method == "mint_shrink":
            if residuals is not None and residuals.shape[0] == n_total:
                # Sample covariance with Ledoit-Wolf diagonal shrinkage
                n_samples = residuals.shape[1]
                sample_cov = (residuals @ residuals.T) / max(1, n_samples - 1)
                target = np.diag(np.diag(sample_cov))
                shrinkage = 0.2  # Ledoit-Wolf shrinkage intensity
                shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
                inv_W = np.linalg.pinv(shrunk_cov)
            else:
                inv_W = np.eye(n_total)
            ST_invW = S.T @ inv_W
            P = np.linalg.pinv(ST_invW @ S) @ ST_invW
            self.P_matrix = P

        else:
            raise ValueError(f"Unknown reconciliation method: {self.method}")

        self.is_fitted = True
        return self

    def reconcile(
        self,
        base_forecasts: Union[pd.DataFrame, np.ndarray],
        date_col: str = "date",
        node_col: str = "node_id",
        pred_col: str = "y_pred",
    ) -> pd.DataFrame:
        """Apply reconciliation to base forecasts across all hierarchy nodes.

        Returns DataFrame containing reconciled forecasts `y_reconciled` for all hierarchy nodes.
        """
        if not self.is_fitted or self.P_matrix is None or self.S_matrix is None:
            raise ValueError("HierarchicalReconciler must be fitted before reconcile().")

        if isinstance(base_forecasts, pd.DataFrame):
            dates = sorted(base_forecasts[date_col].unique())
            reconciled_records = []

            for d in dates:
                d_df = base_forecasts[base_forecasts[date_col] == d].set_index(node_col)
                # Align vector of base forecasts with all_nodes
                y_hat = np.array(
                    [d_df.loc[n, pred_col] if n in d_df.index else 0.0 for n in self.all_nodes],
                    dtype=np.float64,
                )

                # Reconcile bottom nodes: y_tilde_bottom = P @ y_hat
                y_tilde_bottom = self.P_matrix @ y_hat
                # Ensure non-negativity
                y_tilde_bottom = np.maximum(0.0, y_tilde_bottom)

                # Reconcile all hierarchy levels: y_tilde = S @ y_tilde_bottom
                y_tilde_all = self.S_matrix @ y_tilde_bottom

                for node_name, base_val, rec_val in zip(self.all_nodes, y_hat, y_tilde_all):
                    reconciled_records.append(
                        {
                            date_col: d,
                            node_col: node_name,
                            "y_base": float(base_val),
                            "y_reconciled": float(rec_val),
                        }
                    )

            return pd.DataFrame(reconciled_records)

        else:
            # Array shape: (n_nodes, horizon)
            y_hat = np.asarray(base_forecasts, dtype=np.float64)
            y_tilde_bottom = np.maximum(0.0, self.P_matrix @ y_hat)
            y_tilde_all = self.S_matrix @ y_tilde_bottom
            return y_tilde_all

    def check_coherence(
        self,
        reconciled_df: pd.DataFrame,
        date_col: str = "date",
        node_col: str = "node_id",
        pred_col: str = "y_reconciled",
        tolerance: float = 1e-3,
    ) -> bool:
        """Verify that bottom-level sums perfectly match parent and total levels."""
        if self.S_matrix is None or self.all_nodes is None:
            return False

        dates = reconciled_df[date_col].unique()
        for d in dates:
            d_df = reconciled_df[reconciled_df[date_col] == d].set_index(node_col)
            y_tilde_all = np.array([d_df.loc[n, pred_col] for n in self.all_nodes])
            y_bottom = y_tilde_all[-len(self.bottom_nodes) :]
            expected_all = self.S_matrix @ y_bottom

            max_discrepancy = np.max(np.abs(y_tilde_all - expected_all))
            if max_discrepancy > tolerance:
                logger.warning(
                    f"Coherence check failed on date {d} with max discrepancy {max_discrepancy:.5f}"
                )
                return False

        return True
