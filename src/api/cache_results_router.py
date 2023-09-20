"""Api router for cached results."""
import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter

from src.api.api_models import ClassifierModel
from src.api.preprocess import class_label_to_index, test_df
from src.pipelines.evaluate import Evaluate
from src.settings.general import data_paths
from src.utils.utils import load_results

router = APIRouter()


def get_label_class(logits, class_label_to_index: dict[int, str]):
    """Get predicted labels from predicted indexes."""
    if isinstance(logits, torch.Tensor):
        logits = logits.numpy()
    class_indexes = np.argmax(logits, axis=1)
    classes = [class_label_to_index[label] for label in class_indexes]
    return classes


@router.get("/test_df/")
async def get_test_df():
    """Get test dataframe."""
    test_df_json = test_df.to_json(orient="split")
    return {"DataFrame": test_df_json}


@router.get("/test_classes/")
async def get_test_classes():
    """Get test classes."""
    test_classes = list(class_label_to_index.values())
    return {"Classes": test_classes, "ClassLabelDict": class_label_to_index}


@router.post("/predictions/")
async def load_model_cache_predictions(model_name: str):
    """Load model cached predictions on the test set."""
    # Predictions
    if model_name == ClassifierModel.BERT_CLASSIFIER.value:
        predictions_cache_path = data_paths.bert_based_model_cached_predictions_path
    elif model_name == ClassifierModel.LOGISTIC_CLASSIFIER.value:
        predictions_cache_path = data_paths.tfidf_logistic_model_cached_predictions_path
    predictions = load_results(predictions_cache_path)
    predicted_labels = get_label_class(predictions, class_label_to_index)
    # Labels
    labels_cache_path = data_paths.cached_labels_path
    labels_cache = load_results(labels_cache_path)
    labels_class = get_label_class(labels_cache, class_label_to_index)
    # Text
    text_cache_path = data_paths.cached_test_set
    text_cache = load_results(text_cache_path)
    df = pd.DataFrame(
        {"text": text_cache, "PredictedClass": predicted_labels, "ActualClass": labels_class}
    )
    df = df.to_json(orient="split")
    return {"DataFrame": df}


@router.get("/model_evaluation/auroc/")
async def compute_model_auroc(model_name: str):
    """Compute model auroc evaluation metric."""
    if model_name == ClassifierModel.BERT_CLASSIFIER.value:
        predictions_cache_path = data_paths.bert_based_model_cached_predictions_path
        cached_predictions = load_results(predictions_cache_path)

    elif model_name == ClassifierModel.LOGISTIC_CLASSIFIER.value:
        predictions_cache_path = data_paths.tfidf_logistic_model_cached_predictions_path
        cached_predictions = load_results(predictions_cache_path)

    labels_path = data_paths.cached_labels_path
    cached_labels = load_results(labels_path)

    auroc = Evaluate.calculate_auroc(cached_predictions, cached_labels, class_label_to_index)
    auroc_df_json = auroc.to_json(orient="split")
    return {"DataFrame": auroc_df_json}


@router.get("/model_evaluation/classification_report/")
async def compute_model_classification_report(model_name: str, confidence: float):
    """Compute model classification report metric."""
    if model_name == ClassifierModel.BERT_CLASSIFIER.value:
        predictions_cache_path = data_paths.bert_based_model_cached_predictions_path
        cached_predictions = load_results(predictions_cache_path)

    elif model_name == ClassifierModel.LOGISTIC_CLASSIFIER.value:
        predictions_cache_path = data_paths.tfidf_logistic_model_cached_predictions_path
        cached_predictions = load_results(predictions_cache_path)

    labels_path = data_paths.cached_labels_path
    cached_labels = load_results(labels_path)

    classification_report_df = Evaluate.calculate_classification_report(
        cached_predictions, cached_labels, class_label_to_index, confidence
    )
    classification_report_df = classification_report_df.to_json(orient="split")
    return {"DataFrame": classification_report_df}
