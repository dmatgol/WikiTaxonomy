"""Preprocessing pipeline stage."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.pipelines.base import Pipeline
from src.settings.general import constants


class PreProcessing(Pipeline):
    """Initialize the PreProcessing class.

    Args:
        train_path (str): File path to the training dataset.
        val_path (str): File path to the validation dataset.
        test_path (str): File path to the test dataset.
    """

    def __init__(self, train_path: str, val_path: str, test_path: str) -> None:
        """Initialize the PreProcessing class."""
        self.train_df = pd.read_csv(train_path)
        self.train_df = pd.read_csv(val_path)
        self.train_df = pd.read_csv(test_path)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
        """Run the preprocessing pipeline.

        Includes label encoding for train, validation, and test datasets.
        """
        encoder = self.encode_label(self.train_df)
        train_df = self.apply_label_encoder(encoder, self.train_df)
        val_df = self.apply_label_encoder(encoder, self.train_df)
        test_df = self.apply_label_encoder(encoder, self.train_df)

        label_column = constants.label_column
        label_column_encoded = constants.label_column_encoded
        class_label_to_index = train_df.set_index(label_column_encoded)[label_column].to_dict()

        return train_df, val_df, test_df, class_label_to_index

    @staticmethod
    def encode_label(df: pd.DataFrame) -> LabelEncoder:
        """Encode the labels in a DataFrame using a LabelEncoder."""
        encoder = LabelEncoder()
        encoder.fit(df[constants.label_column])
        return encoder

    @staticmethod
    def apply_label_encoder(encoder: LabelEncoder, df: pd.DataFrame) -> pd.DataFrame:
        """Apply a LabelEncoder to a DataFrame to encode the labels."""
        encoded_y = encoder.transform(df[constants.label_column])
        df[constants.label_column_encoded] = encoded_y
        return df
