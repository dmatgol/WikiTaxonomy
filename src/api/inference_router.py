"""Api inference router."""
import shap
import torch
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from src.api.api_models import BertShapValuesResponse, BertTaxonomyResponse
from src.api.preprocess import class_label_to_index
from src.model.wiki_taxonomy_classifier import WikiTaxonomyClassifier
from src.settings.general import DEVICE, data_paths

router = APIRouter()


bert_classifier = WikiTaxonomyClassifier.load_from_checkpoint(
    checkpoint_path=data_paths.best_bert_based_model_path, map_location=torch.device(DEVICE)
)


@router.post("/bert_classifier/", response_model=BertTaxonomyResponse)
async def predict_sentence_using_bert_classifier(request: str):
    """Predict sentence using bert classifier."""
    predicted_idx, predicted_label = bert_classifier.predict_text(request, class_label_to_index)
    response = BertTaxonomyResponse(
        predicted_class_index=predicted_idx, predicted_class=predicted_label
    )
    return response


@router.post("/bert_classifier/shap_values/", response_model=BertShapValuesResponse)
async def compute_shap_values_plot(request: str, prediction_idx: int):
    """Compute shap values for bert classifier and return shap plot."""
    explainer = shap.Explainer(
        bert_classifier.compute_output_for_shap_values,
        bert_classifier.tokenizer,
        output_names=list(class_label_to_index.values()),
    )
    shap_values = explainer([request])
    shap_plot = shap.plots.text(shap_values[:, :, prediction_idx], display=False)
    return HTMLResponse(content=shap_plot, status_code=200)
