"""Wiki Taxonomy classifier train entrypoint."""
import argparse
import logging

from pipelines.evaluate import Evaluate
from pipelines.inference import Inference
from pipelines.preprocessing import PreProcessing
from pipelines.train import Train
from settings.general import ModelConfig, data_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["bert_classifier", "tfidf_logistic_classifier"],
        help="Select the model to use. Choices: bert_classifier or tfidf_logistic_classifier",
        default="bert_classifier",
    )
    parser.add_argument(
        "--run_mode",
        choices=["train", "inference"],
        help="Select the mode to use the model. Choices: train or inference",
        default="inference",
    )
    args = parser.parse_args()

    return args


def read_model_config(model_config_path: str) -> ModelConfig:
    """Read model configuration from a YAML file."""
    model_config = ModelConfig.from_yaml(model_config_path)
    return model_config


def main() -> None:
    """Run main script for preprocessing and training steps."""
    args = parse_cli_args()
    if args.model == "bert_classifier":
        model_config = read_model_config("configs/bert_classifier.yaml")
    else:
        model_config = read_model_config("configs/tfidf_logistic_classifier.yaml")
    logging.info("Started PreProcessing Step...")
    train_df, val_df, test_df, class_label_to_index = PreProcessing(
        train_path=data_paths.train_path,
        val_path=data_paths.val_path,
        test_path=data_paths.test_path,
    ).run()
    if args.run_mode == "train":
        logging.info("Started training the model ...")
        Train(train_df=train_df, val_df=val_df, model_config=model_config, test_df=test_df).run()
    else:
        logging.info("Started inference mode ...")
        Inference(
            class_label_to_index=class_label_to_index, test_df=test_df, model_config=model_config
        ).run()
        if args.model == "bert_classifier":
            Evaluate(
                cached_predictions_path=data_paths.bert_based_model_cached_predictions_path,
                cached_labels_path=data_paths.bert_based_model_cached_labels_path,
                cached_shap_values_path=data_paths.bert_based_model_cached_shap_path,
                class_label_to_index=class_label_to_index,
            ).run()
        else:
            Evaluate(
                cached_predictions_path=data_paths.tfidf_logistic_model_cached_predictions_path,
                cached_labels_path=data_paths.bert_based_model_cached_labels_path,
                cached_shap_values_path=data_paths.tfidf_logistic_model_cached_shap_path,
                class_label_to_index=class_label_to_index,
            ).run()


if __name__ == "__main__":
    main()
