"""Evaluate pipeline stage."""
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torchmetrics import AUROC

from src.pipelines.base import Pipeline
from src.utils.utils import load_results


class Evaluate(Pipeline):
    """Initialize the evaluation pipeline class.

    Args:
    class_label_to_index (dict[int, str]): Mapping from class label index to class label.
    cached_predictions_path (str): Path for cached predictions.
    cached_labels_path (str): Path for cached labels.
    cached_shap_values (str): Path for cached shap values.
    classification_threshold (float): Classification threshold for model prediction.
    """

    def __init__(
        self,
        class_label_to_index: dict[int, str],
        cached_predictions_path: str,
        cached_labels_path: str,
        cached_shap_values_path: str,
        classification_threshold: float = 0.5,
    ) -> None:
        """Initialize the evaluation class."""
        self.cached_predictions = load_results(cached_predictions_path)
        self.cached_labels = load_results(cached_labels_path)
        self.cached_shap_values = load_results(cached_shap_values_path)
        self.class_label_to_index = class_label_to_index
        self.classification_threshold = classification_threshold

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the evaluation pipeline.

        Computes AUROC and classification report (including, precision, recall and f1 score)
        given into consideration a set of predictions and a set of labels.
        """
        category_auroc, class_report = self.evaluate(self.cached_predictions, self.cached_labels)
        return category_auroc, class_report

    def evaluate(self, predictions, labels) -> pd.DataFrame:
        """Evaluate the performance of the model predictions on F1 score and AUROC."""
        category_auroc = self.calculate_auroc(predictions, labels, self.class_label_to_index)
        class_report = self.calculate_classification_report(
            predictions, labels, self.class_label_to_index, self.classification_threshold
        )

        return category_auroc, class_report

    @staticmethod
    def calculate_auroc(predictions, labels, class_label_to_index):
        """Calculate AUROC for a set of predictions and corresponding labels."""
        logging.info("AUROC per tag")
        category_auroc = {}
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions)

        for label_idx, label in class_label_to_index.items():
            auroc = AUROC(task="binary")
            tag_auroc = auroc(predictions[:, label_idx], labels[:, label_idx])
            category_auroc[label] = tag_auroc.item()
            logging.info(f"{label}: {tag_auroc}")

        category_auroc_df = pd.DataFrame(
            {"ClassLabel": list(category_auroc.keys()), "AUROC": list(category_auroc.values())}
        )
        return category_auroc_df

    @staticmethod
    def calculate_classification_report(
        predictions,
        labels,
        class_label_to_index,
        classification_threshold: float = 0.5,
    ):
        """Compute the classification report for a set of predictions and corresponding labels."""
        y_pred = predictions
        if isinstance(predictions, torch.Tensor):
            y_pred = predictions.cpu().numpy()
        y_true = labels.cpu().numpy()

        upper, lower = 1, 0

        y_pred = np.where(y_pred > classification_threshold, upper, lower)

        class_report = classification_report(
            y_true,
            y_pred,
            target_names=class_label_to_index.values(),
            zero_division=0,
            output_dict=True,
        )
        logging.info(class_report)
        classification_report_df = pd.DataFrame(class_report).transpose().reset_index()
        return classification_report_df
