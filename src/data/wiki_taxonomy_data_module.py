"""Module with functions and classes related to the WikiTaxonomy dataset."""
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.wiki_taxonomy_dataset import WikiTaxonomyDataset


class WikiTaxonomyDataModule(pl.LightningDataModule):
    """Manage datasets and data loading for a WikiTaxonomy model.

    Args:
        train_df (pd.DataFrame): DataFrame containing the training data.
        val_df (pd.DataFrame): DataFrame containing the validation data.
        test_df (pd.DataFrame): DataFrame containing the test data.
        tokenizer: The tokenizer to use for processing text data.
        batch_size (int, optional): The batch size for data loading.
        Default is 8.
        max_token_len (int, optional): The maximum token length for input
        sequences. Default is 128.
        class_label_to_index (dict, optional): A dictionary mapping class
        labels to their indices.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df,
        test_df,
        tokenizer,
        batch_size=8,
        max_token_len=128,
        class_label_to_index: dict = None,
    ):
        """Manage datasets and data loading for a WikiTaxonomy model."""
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.class_label_to_index = class_label_to_index

    def setup(self, stage=None):
        """Set up the training, validation, and test datasets."""
        self.train_dataset = WikiTaxonomyDataset(
            data=self.train_df,
            tokenizer=self.tokenizer,
            max_token_len=self.max_token_len,
            class_label_to_index=self.class_label_to_index,
        )

        self.val_dataset = WikiTaxonomyDataset(
            data=self.val_df,
            tokenizer=self.tokenizer,
            max_token_len=self.max_token_len,
            class_label_to_index=self.class_label_to_index,
        )

        self.test_dataset = WikiTaxonomyDataset(
            data=self.test_df,
            tokenizer=self.tokenizer,
            max_token_len=self.max_token_len,
            class_label_to_index=self.class_label_to_index,
        )

    def train_dataloader(self):
        """Create and return a DataLoader for the training dataset."""
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        """Create and return a DataLoader for the validation dataset."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        """Create and return a DataLoader for the test/prediction dataset."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)
