"""Train pipeline."""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizerFast as DistilBertTokenizer

from src.data.wiki_taxonomy_data_module import WikiTaxonomyDataModule
from src.model.logistic_classifier import TFIDFLogisticTextClassifier
from src.model.wiki_taxonomy_classifier import WikiTaxonomyClassifier
from src.pipelines.base import Pipeline
from src.settings.general import ModelConfig, constants


class Train(Pipeline):
    """Initialize the training pipeline.

    Args:
        train_df (pd.DataFrame): Training dataset.
        val_df (pd.DataFrame): Validation dataset.
        model_config (ModelConfig): Configuration for the model.
        test_df (pd.DataFrame, optional): Test dataset, if available.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_config: ModelConfig,
        test_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Initialize the training pipeline."""
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.model_config = model_config
        folder = self._create_experiment_folder()
        self.tensorboard_results = os.path.join(os.getcwd(), f"model_experiments/{folder}")

    def run(self) -> None:
        """Run the training pipeline based on the model configuration."""
        if self.model_config.name == WikiTaxonomyClassifier.__name__:
            self.train_wikitaxonomy_bert_classifier()
        elif self.model_config.name == TFIDFLogisticTextClassifier.__name__:
            self.train_wikitaxonomy_logistic_classifier()
        else:
            raise ValueError("Model not defined...")

    def _create_experiment_folder(self):
        """Create an experiment folder for storing results and checkpoints."""
        folder = (
            str(datetime.now()).replace(" ", "_").replace(":", "_").replace("-", "_").split(".")[0]
        )
        os.makedirs(os.path.join(os.getcwd(), f"model_experiments/{folder}/"))
        return folder

    def train_wikitaxonomy_bert_classifier(self):
        """Train a BERT-based classifier for WikiTaxonomy dataset."""
        # Tokenizer
        bert_model_name = self.model_config.parameters["bert_model_name"]
        tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)

        # Data setup
        batch_size = self.model_config.train_config["batch_size"]
        wiki_data_module = WikiTaxonomyDataModule(
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_token_len=self.model_config.parameters["max_token_len"],
        )
        wiki_data_module.setup()

        # Model parameters
        classes = np.unique(self.train_df[constants.label_column_encoded])
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=self.train_df[constants.label_column_encoded],
        )
        epochs = self.model_config.train_config["epochs"]
        steps_per_epoch = len(self.train_df) // batch_size
        total_training_steps = steps_per_epoch * epochs
        warmup_steps = total_training_steps // 5  # 1/5 steps for warmup

        # model definition
        model = WikiTaxonomyClassifier(
            n_classes=len(classes),
            class_weights=class_weights,
            scheduler=self.model_config.parameters["scheduler"],
            lr=self.model_config.train_config["lr"],
            weight_decay=self.model_config.train_config["weight_decay"],
            n_training_steps=total_training_steps,
            n_warmup_steps=warmup_steps,
        )

        # Pytorch lightning trainer
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.tensorboard_results,
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3)
        logger = TensorBoardLogger(self.tensorboard_results, name="wiki-taxonomy")
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=self.model_config.train_config["epochs"],
            callbacks=[checkpoint_callback, early_stopping_callback],
            gradient_clip_val=0.9,
        )
        trainer.fit(model, wiki_data_module)

    def train_wikitaxonomy_logistic_classifier(self) -> None:
        """Train a logistic classifier using TF-IDF features."""
        tfidf_logistic_classifier = TFIDFLogisticTextClassifier(
            train_data=self.train_df, tf_idf_min_df=self.model_config.parameters["tf_idf_min_df"]
        )
        tfidf_logistic_classifier.train(self.tensorboard_results)
