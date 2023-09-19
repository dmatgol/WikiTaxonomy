"""Script to store constants, data paths and model configuration."""
import os
from typing import Any

import torch
import yaml
from pydantic import BaseModel


class Paths(BaseModel):
    """Store data paths."""

    bert_shap_values_cache: str = (
        f"{os.getcwd()}/model/cached_results/shap_values_cache_bert_classifier.pkl"
    )
    train_path: str = f"{os.getcwd()}/data/DBPEDIA_train.csv"
    val_path: str = f"{os.getcwd()}/data/DBPEDIA_val.csv"
    test_path: str = f"{os.getcwd()}/data/DBPEDIA_val.csv"
    best_bert_based_model_path: str = f"{os.getcwd()}/model/best_models/best-checkpoint-v2.ckpt"
    best_tfidf_logistic_model_path: str = f"{os.getcwd()}/model/best_models/logistic_model.pkl"
    # cached results bert based model
    bert_based_model_cached_predictions_path: str = (
        f"{os.getcwd()}/model/cached_results/predictions_first_version.pkl"
    )
    bert_based_model_cached_labels_path: str = (
        f"{os.getcwd()}/model/cached_results/labels_first_version.pkl"
    )
    bert_based_model_cached_shap_path: str = (
        f"{os.getcwd()}/model/cached_results/shap_values_cache_bert_classifier.pkl"
    )
    bert_based_model_cached_test_set: str = (
        f"{os.getcwd()}/model/cached_results/bert_based_text.pkl"
    )
    # cached results logistic regressor model
    tfidf_logistic_model_cached_predictions_path: str = (
        f"{os.getcwd()}/model/cached_results/logistic_predictions.pkl"
    )
    tfidf_logistic_model_cached_labels_path: str = (
        f"{os.getcwd()}/model/cached_results/logistic_labels.pkl"
    )
    tfidf_logistic_model_cached_shap_path: str = (
        f"{os.getcwd()}/model/cached_results/logistic_shap_values.pkl"
    )
    tfidf_logistic_model_cached_test_set: str = (
        f"{os.getcwd()}/model/cached_results/logistic_text.pkl"
    )


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
