"""Script to store constants, data paths and model configuration."""
import os
from typing import Any

import torch
import yaml
from pydantic import BaseModel


class Paths(BaseModel):
    """Store data paths."""

    shap_values_cache: str = f"{os.getcwd()}/model/shap_values_cache.pkl"
    train_path: str = f"{os.getcwd()}/data/DBPEDIA_train.csv"
    val_path: str = f"{os.getcwd()}/data/DBPEDIA_val.csv"
    test_path: str = f"{os.getcwd()}/data/DBPEDIA_val.csv"


class Constants(BaseModel):
    """Store constants."""

    label_column: str = "l1"
    label_column_encoded: str = "l1_encoded"


class ModelConfig(BaseModel):
    """Specify model configuration."""

    name: str
    parameters: dict[str, Any] = None
    train_config: dict[str, Any] = None

    @classmethod
    def from_yaml(cls, model_config_path: str):
        """Create Model Class config from yaml file."""
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)

        if model_config is None:
            return cls(
                name="WikiTaxonomyClassifier",
                parameters={
                    "scheduler": "cosine_annealing",
                    "bert_model_name": "distilbert-base-cased",
                    "max_token_len": 512,
                },
                train_config={"batch_size": 8, "epochs": 10, "lr": 1e-6, "weight_decay": 1e-3},
            )

        return cls(**model_config)


DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
data_paths = Paths()
constants = Constants()
