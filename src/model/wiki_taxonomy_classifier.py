"""Taxonomy Classifier based on Distilbert with a classification head."""
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import shap
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm
from transformers import DistilBertModel
from transformers import DistilBertTokenizerFast as DistilBertTokenizer
from transformers import get_linear_schedule_with_warmup

from src.settings.general import DEVICE, data_paths
from src.utils.utils import get_mean_shap_value_per_token, save_results


class WikiTaxonomyClassifier(pl.LightningModule):
    """Taxonomy Classifier based on Distilbert with a classification head.

    Args:
        n_classes (int): Number of classes in the classification task.
        class_weights (dict[int, str], optional): Class weights for handling
        class imbalance.
        scheduler (str, optional): Learning rate scheduler type
        ("linear_schedule" or "cosine_annealing").
        lr (float, optional): Initial learning rate.
        weight_decay (float, optional): Weight decay for the optimizer.
        n_training_steps (int, optional): Total number of training steps.
        n_warmup_steps (int, optional): Number of warmup steps for learning
        rate scheduling.
    """

    def __init__(
        self,
        n_classes: int,
        class_weights: dict[int, str] = None,
        scheduler: str = "cosine_annealing",
        lr: float = 1e-7,
        weight_decay: float = 1e-3,
        n_training_steps=None,
        n_warmup_steps=None,
    ) -> None:
        """Classifier based on Distilbert with a classification head."""
        super().__init__()
        # Model specific arguments
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-cased", return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

        # scheduler/optimizer related parameter
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.lr = float(lr)
        self.scheduler = scheduler
        self.weight_decay = float(weight_decay)

        # loss
        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        )
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.n_classes = n_classes
        self.save_hyperparameters()

        self.train_outputs: list[dict[str, Any]] = []
        self.val_outputs: list[dict[str, Any]] = []

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass of the model."""
        output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(output.last_hidden_state, dim=1)
        output = self.classifier(pooled_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        """Training step for a batch of data."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.train_outputs.append({"predictions": outputs, "labels": label})
        return {"loss": loss, "predictions": outputs, "labels": label}

    def validation_step(self, batch, batch_idx):
        """Validation step for a batch of data."""  # noqa: D401
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.val_outputs.append({"predictions": outputs, "labels": label})
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for a batch of data."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        """Evaluate train performance on epoch training end."""
        labels = []
        predictions = []

        # Retrieve the class label to index mapping
        class_label_to_index = self.trainer.datamodule.class_label_to_index

        for output in self.train_outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        all_class_roc_auc = []

        for label_index, label in enumerate(class_label_to_index):
            roc_auc_metric = BinaryAUROC()
            class_roc_auc = roc_auc_metric(predictions[:, label_index], labels[:, label_index])
            all_class_roc_auc.append(class_roc_auc)
            self.logger.experiment.add_scalar(
                f"{label}_roc_auc/Train", class_roc_auc, self.current_epoch
            )

        self.logger.experiment.add_scalar(
            "Average_roc_auc_all_classes/Train", np.mean(all_class_roc_auc), self.current_epoch
        )

    def on_validation_epoch_end(self):
        """Evaluate validation performance on epoch training end."""
        labels = []
        predictions = []

        # Retrieve the class label to index mapping
        class_label_to_index = self.trainer.datamodule.class_label_to_index

        for output in self.val_outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        all_class_roc_auc = []

        for label_index, label in enumerate(class_label_to_index):
            roc_auc_metric = BinaryAUROC()
            class_roc_auc = roc_auc_metric(predictions[:, label_index], labels[:, label_index])
            all_class_roc_auc.append(class_roc_auc)
            self.logger.experiment.add_scalar(
                f"{label}_roc_auc/Valid", class_roc_auc, self.current_epoch
            )

        all_class_roc_auc_tensor = torch.tensor(all_class_roc_auc)
        self.logger.experiment.add_scalar(
            "Average_roc_auc_all_classes/Valid",
            torch.mean(all_class_roc_auc_tensor),
            self.current_epoch,
        )

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler == "linear_schedule":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.n_warmup_steps,
                num_training_steps=self.n_training_steps,
            )
        elif self.scheduler == "cosine_annealing":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.n_training_steps)
        else:
            raise ValueError("Scheduler not defined.")

        return dict(  # noqa: C408
            optimizer=optimizer,
            lr_scheduler=dict(scheduler=scheduler, interval="step"),  # noqa: C408
        )

    def predict_text(self, new_article: str, class_label_to_index: dict[int, str]):
        """Predict the class label for a single article input."""
        self.eval()
        self.freeze()

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
        encoding = tokenizer.encode_plus(
            new_article,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Perform inference with the model
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        with torch.no_grad():
            _, predicted_probabilities = self.to("cpu")(input_ids, attention_mask)

        predicted_class_index = torch.argmax(predicted_probabilities, dim=1).item()
        predicted_class = class_label_to_index[predicted_class_index]
        return predicted_class_index, predicted_class

    def predict_batch_text(self, new_batch=None):
        """Predict the class label for a batch of articles input."""
        self.eval()
        self.freeze()

        predictions = []
        labels = []
        texts = []
        for test_batch in tqdm(new_batch):

            with torch.no_grad():
                _, outputs = self.to(DEVICE)(
                    test_batch["input_ids"].to(DEVICE), test_batch["attention_mask"].to(DEVICE)
                )

            predictions.append(outputs)
            labels.append(test_batch["label"].int())
            texts.append(test_batch["article_text"])

        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        texts = [text for text_batches in texts for text in text_batches]

        for result, result_path in [
            (predictions, data_paths.bert_based_model_cached_predictions_path),
            (labels, data_paths.bert_based_model_cached_labels_path),
            (texts, data_paths.bert_based_model_cached_test_set),
        ]:
            save_results(result, result_path)

        return predictions, labels

    def compute_output_for_shap_values(self, text):
        """Calculate model interpretability with SHAP values."""
        tv = torch.tensor(
            [
                self.tokenizer.encode(word, padding="max_length", max_length=512, truncation=True)
                for word in text
            ]
        )
        attention_mask = (tv != 0).type(torch.int64)
        with torch.no_grad():
            outputs = (
                self.to(DEVICE)(tv.to(DEVICE), attention_mask=attention_mask.to(DEVICE))[1]
                .detach()
                .cpu()
                .numpy()
            )

        return outputs

    def compute_shap_values_batch(self, test_batch, class_names: list):
        """Compute shap values per batch of article inputs."""
        shap_value_batch_list = []
        for _, batch in tqdm(enumerate(test_batch)):
            explainer = shap.Explainer(
                self.compute_output_for_shap_values, self.tokenizer, output_names=class_names
            )
            shap_values = explainer(batch["article_text"])
            shap_value_batch_list.append(shap_values)

        token_shap_value_dict = get_mean_shap_value_per_token(shap_value_batch_list, class_names)

        save_results(token_shap_value_dict, data_paths.bert_shap_values_cache)
        return token_shap_value_dict

    @staticmethod
    def plot_shap_values(token_shap_value_dict: dict, class_label: str):
        """Plot most important features for a given class label."""
        df = pd.DataFrame.from_dict(token_shap_value_dict, orient="index").reset_index()
        df.columns = ["word"] + list(df.columns[1:])
        top_k_values = df[df[class_label] > 0].nlargest(10, class_label).index

        # PLOT
        plt.figure(figsize=(8, 4))
        plt.barh(
            df.iloc[top_k_values]["word"], df.iloc[top_k_values][class_label], color="royalblue"
        )

        # Set the plot title and labels
        plt.title(
            f"Top 10 features that contribute the most to "
            f'"{class_label}" taxonomy classification'
        )
        plt.xlabel("Mean |shap|")
        plt.ylabel("Features")

        return plt
