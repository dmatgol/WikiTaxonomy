"""WikiTaxonomy dataset."""
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast as DistilBertTokenizer

from src.settings.general import constants


class WikiTaxonomyDataset(Dataset):
    """Dataset class to process Wikipedia articles."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: DistilBertTokenizer,
        max_token_len: int = 128,
        class_label_to_index: dict = None,
    ):
        """Dataset class to process Wikipedia articles.

        Args:
            data (pd.DataFrame): A DataFrame containing text and labels.
            tokenizer (DistilBertTokenizer): A DistilBERT tokenizer for
            tokenizing the text.
            max_token_len (int, optional): The maximum token length for the
            input text. Defaults to 128.
            class_label_to_index (dict, optional): A dictionary mapping class
            labels to their indices.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.n_classes = len(class_label_to_index)
        self.class_label_to_index = class_label_to_index

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Return tokenized text and labels for an index."""
        data_row = self.data.iloc[index]

        article_text = data_row.text
        labels = torch.zeros(self.n_classes)
        #  For inference in a new test set without labels, the true label is unknown
        if constants.label_column_encoded in data_row.index:
            labels[data_row[constants.label_column_encoded]] = 1

        encoding = self.tokenizer.encode_plus(
            article_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            # Makes sure smaller articles, have the same tokens as larger.
            padding="max_length",
            truncation=True,  # Truncate token len at max_length
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(  # noqa: C408
            article_text=article_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            label=torch.FloatTensor(labels),
        )
