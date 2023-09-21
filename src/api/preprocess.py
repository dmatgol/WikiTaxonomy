"""Preprocess helper step for the api."""
import argparse

from pipelines.preprocessing import PreProcessing


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        help="Please provide the train path.",
    )
    parser.add_argument(
        "--valid_path",
        help="Please provide the valid path.",
    )
    parser.add_argument(
        "--test_path",
        help="Please provide the test path.",
    )
    args = parser.parse_args()
    return args


args = parse_cli_args()


train_df, val_df, test_df, class_label_to_index = PreProcessing(
    train_path=args.train_path,
    val_path=args.valid_path,
    test_path=args.test_path,
).run()
