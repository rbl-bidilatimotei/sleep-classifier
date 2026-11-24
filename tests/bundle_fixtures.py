from typing import Any

import torch

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.model_factory import ModelFactory
from sleep_classifier.motion_feature_input import MOTION_DEFINITION, MOTION_FEATURE


def make_checkpoint() -> dict[str, Any]:
    """Build random weights and synthetic preprocessing for the exact v1 profile."""
    factory = ModelFactory({
        "model_type": "patch_transformer", "context_epochs": 30, "patch_size": 3,
        "transformer": {
            "d_model": 16, "nhead": 2, "num_layers": 1, "feedforward_dim": 32,
            "dropout": 0.2, "max_context": 30,
        },
        "window_features": {
            "name": "heart_rate_deviation", "history_epochs": 10, "min_history_epochs": 4,
        },
        "motion_features": MOTION_DEFINITION,
    })
    columns = [*FEATURE_COLUMNS, MOTION_FEATURE]
    state = {
        "schema_version": 1, "heart_rate_scale": 3.0, "delta_mean": 0.5, "delta_scale": 2.0,
        "available_training_targets": 9, "scaling_fit": "training_targets_only",
    }
    motion = {
        "name": "fixed_lag_motion", "definition": dict(MOTION_DEFINITION),
        "audit_config_sha256": "1" * 64, "snapshot_manifest_sha256": "2" * 64,
        "scaling_fit": "training_targets_only", "missing_policy": "training_mean",
        "training_observed_targets": 12, "training_missing_targets": 3,
        "validation_observed_targets": 4, "validation_missing_targets": 1,
        "imputation_mean": 0.125,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(58)
        model = factory.build(7, 4, window_feature_state=state)
    return {
        "state_dict": model.state_dict(), "model_type": factory.model_type,
        "model_config": factory.checkpoint_config(7, 4), "features": columns,
        "model_features": factory.model_features(columns), "context_epochs": 30,
        "scaler_mean": [1.0] * 7, "scaler_scale": [3.0] * 7, "window_feature_state": state,
        "motion_features": motion,
    }

