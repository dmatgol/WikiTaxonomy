"""Function to get shap values calculated over batches."""
import numpy as np


def get_mean_shap_value_per_token(
    shap_value_batch_list: list, classes: list[str]
) -> dict[str, dict[str, float]]:
    """Save the shap values per class over test batches.

    Ugly function! Saves the shap values per class per batch, and is needed
    since Shap.summary plot is not working for multiclass problems.
    """
    token_shap_value_dict = {}

    for shap_value_batch in shap_value_batch_list:
        for outer_idx, sentence_shap in enumerate(shap_value_batch.values):
            for inner_idx, token_shap in enumerate(sentence_shap):
                token = shap_value_batch.data[outer_idx][inner_idx]
                for label_idx, label in enumerate(classes):
                    if (token in token_shap_value_dict) and (label in token_shap_value_dict[token]):
                        shap_value_before = token_shap_value_dict[token][label]
                        mean_shap_value = np.mean([shap_value_before, token_shap[label_idx]])
                        token_shap_value_dict[token] = {label: mean_shap_value}
                    else:
                        if token in token_shap_value_dict:
                            token_shap_value_dict[token][label] = token_shap[label_idx]
                        else:
                            token_shap_value_dict[token] = {}
                            token_shap_value_dict[token][label] = token_shap[label_idx]
    return token_shap_value_dict
