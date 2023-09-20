"""Preprocess helper step for the api."""
from src.pipelines.preprocessing import PreProcessing
from src.settings.general import data_paths

train_df, val_df, test_df, class_label_to_index = PreProcessing(
    train_path=data_paths.train_path,
    val_path=data_paths.val_path,
    test_path=data_paths.test_path,
).run()
