"""Inference pipeline stage."""
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.wiki_taxonomy_dataset import WikiTaxonomyDataset
from src.model.logistic_classifier import TFIDFLogisticTextClassifier
from src.model.wiki_taxonomy_classifier import WikiTaxonomyClassifier
from src.pipelines.base import Pipeline
from src.settings.general import DEVICE, ModelConfig, constants, data_paths


class Inference(Pipeline):
    """Initialize the Inference pipeline class.

    Args:
        test_df (pd.DataFrame): Test dataset.
        model_config (ModelConfig): Model configuration to run.
        evaluate (bool): True ti evaluate results on test dataset (evaluation).
        If false, live predictions on test set without label.
    """

    def __init__(
        self, class_label_to_index: dict[int, str], test_df: pd.DataFrame, model_config: ModelConfig
    ) -> None:
        """Initialize the Inference pipeline class."""
        self.test_df = test_df
        self.model_config = model_config
        self.class_label_to_index = class_label_to_index

    def run(self):
        """Run the inference pipeline."""
        # For a new test set without labels, the label returns a mock tensor of zeros
        # this is done just to make sure predict_batch_text is able to do inference with and
        # without labels.
        if self.model_config.name == WikiTaxonomyClassifier.__name__:
            predictions_logits, labels = self.generate_predictions_wiki_taxonomy_classifier()
            model_interpretability = self.compute_model_interpretability(self.model_config.name)
        elif self.model_config.name == TFIDFLogisticTextClassifier.__name__:
            predictions_logits, labels = self.generate_predictions_taxonomy_logistic_classifier()
            predictions_logits = torch.tensor(predictions_logits)
            model_interpretability = self.compute_model_interpretability(self.model_config.name)
        else:
            raise ValueError("Model not defined...")

        predicted_label = self.get_label_class(predictions_logits)
        return predictions_logits, labels, predicted_label, model_interpretability

    def generate_predictions_wiki_taxonomy_classifier(self):
        """Generate predictions using the wiki taxonomy classifier."""
        trained_model = WikiTaxonomyClassifier.load_from_checkpoint(
            checkpoint_path=data_paths.best_bert_based_model_path, map_location=torch.device(DEVICE)
        )
        test_dataloader = self._setup_test_dataloader(trained_model)
        predictions, labels = trained_model.predict_batch_text(test_dataloader)
        return predictions, labels

    def _setup_test_dataloader(self, trained_model):
        """Set the test dataloader."""
        test_data_class = WikiTaxonomyDataset(
            data=self.test_df,
            tokenizer=trained_model.tokenizer,
            max_token_len=self.model_config.parameters["max_token_len"],
            class_label_to_index=self.class_label_to_index,
        )
        test_dataloader = DataLoader(
            test_data_class, batch_size=self.model_config.train_config["batch_size"]
        )
        return test_dataloader

    def get_label_class(self, logits):
        """Get predicted labels from predicted indexes."""
        if isinstance(logits, torch.Tensor):
            logits = logits.numpy()
        class_indexes = np.argmax(logits, axis=1)
        classes = [self.class_label_to_index[label] for label in class_indexes]
        return classes

    def compute_model_interpretability(self, model_name: str):
        """Compute model interpretability using shap values."""
        if model_name == WikiTaxonomyClassifier.__name__:
            trained_model = WikiTaxonomyClassifier.load_from_checkpoint(
                checkpoint_path=data_paths.best_bert_based_model_path,
                map_location=torch.device(DEVICE),
            )
            test_dataloader = self._setup_test_dataloader(trained_model)
            token_shap_value_dict = trained_model.compute_shap_values_batch(
                test_batch=test_dataloader, class_names=list(self.class_label_to_index.values())
            )
            return token_shap_value_dict
        elif model_name == TFIDFLogisticTextClassifier.__name__:
            with open(data_paths.best_tfidf_logistic_model_path, "rb") as file:
                trained_model = pickle.load(file)
            shap_values = trained_model.compute_shap_values(self.test_df)
            return shap_values

    def generate_predictions_taxonomy_logistic_classifier(self):
        """Generate predictions using the tf-idf logistic taxonomy classifier."""
        with open(data_paths.best_tfidf_logistic_model_path, "rb") as file:
            trained_model = pickle.load(file)

        predictions = trained_model.predict(test_data=self.test_df)
        return predictions, self.test_df[constants.label_column_encoded]
