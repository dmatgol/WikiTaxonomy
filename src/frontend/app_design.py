"""Streamlit frontend design application."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap.plots
import streamlit as st
import torch

from src.main import read_model_config
from src.model.wiki_taxonomy_classifier import WikiTaxonomyClassifier
from src.pipelines.evaluate import Evaluate
from src.pipelines.inference import Inference
from src.pipelines.preprocessing import PreProcessing
from src.settings.general import DEVICE, data_paths
from src.utils.utils import load_results


def app_design():
    """Define the design of the streamlit application."""
    st.set_page_config(layout="wide")
    st.title("Wiki Taxonomy Category Dashboard")
    st.sidebar.title("Settings")
    confidence = st.sidebar.slider("Confidence", min_value=0.5, max_value=1.0, value=0.85)
    sentence_test_set = st.sidebar.selectbox(
        "Sentence or cached test dataset", ["TestSet", "Sentence"]
    )
    instructions = """
        1. Please choose on the side bar between predicting an input sentence or predict on the
        cached testset from https://www.kaggle.com/danofer/dbpedia-classes.
            Default will be an cached dataset.\n
        """
    st.write(instructions)

    train_df, val_df, test_df, class_label_to_index = PreProcessing(
        train_path=data_paths.train_path,
        val_path=data_paths.val_path,
        test_path=data_paths.test_path,
    ).run()

    if sentence_test_set == "Sentence":
        user_input = st.text_input("Enter some text:")
        sentence_prediction_design(class_label_to_index, user_input)
    else:
        st.write("**First 5 rows of the DataFrame**:")
        st.write(test_df.head(5))

        st.title("**Model Predictions**")
        st.write(
            "Establishing a baseline is a good practice when developing a new model as it allows "
            "you to compare its performance."
            " In this case, a Logistic Regression model using TF-IDF features has been employed."
        )
        model_predictions_design(test_df, class_label_to_index)

        st.title("**Model Interpretability**")
        # Create two columns in the Streamlit app
        model_interpretability_design(class_label_to_index)

        # MODEL EVALUATION
        st.title("**Model evaluation**")
        offline_evaluation_design(class_label_to_index, confidence)


def sentence_prediction_design(class_label_to_index, user_input: str):
    """Define the design of the sentence prediction section."""
    trained_model = WikiTaxonomyClassifier.load_from_checkpoint(
        checkpoint_path=data_paths.best_bert_based_model_path, map_location=torch.device(DEVICE)
    )
    prediction_idx, prediction_label = trained_model.predict_text(user_input, class_label_to_index)
    st.write(f"**You typed: {user_input}**")
    st.write("**Please wait some seconds to see the model prediction and interpretability.**")

    st.title("**Model Interpretability**")
    explainer = shap.Explainer(
        trained_model.compute_output_for_shap_values,
        trained_model.tokenizer,
        output_names=list(class_label_to_index.values()),
    )
    shap_values = explainer([user_input])
    # Display the input text
    st.write("Model prediction is:", prediction_label)
    html_plot = shap.plots.text(shap_values[:, :, prediction_idx], display=False)
    st.components.v1.html(html_plot, height=600, scrolling=True)


def model_predictions_design(test_df: pd.DataFrame, class_label_to_index: dict[int, str]):
    """Define the design of the model prediction section."""
    col1, col2 = st.columns(2, gap="large")
    with col1:
        col1.subheader("**Bert pre-trained model with a classification head**")
        predictions_cache = data_paths.bert_based_model_cached_predictions_path
        text_cache = data_paths.cached_test_set
        labels_cache = data_paths.cached_labels_path
        model_config = read_model_config("configs/bert_classifier.yaml")
        inference_stage = Inference(
            class_label_to_index=class_label_to_index,
            test_df=test_df,
            model_config=model_config,
        )
        df = load_model_predictions(inference_stage, predictions_cache, labels_cache, text_cache)
        col1.write(df, unsafe_allow_html=True)

    with col2:
        col2.subheader("**TD-IDF Logistic regression classifier**")
        predictions_cache = data_paths.tfidf_logistic_model_cached_predictions_path
        text_cache = data_paths.cached_test_set
        labels_cache = data_paths.cached_labels_path
        model_config = read_model_config("configs/tfidf_logistic_classifier.yaml")
        inference_stage = Inference(
            class_label_to_index=class_label_to_index,
            test_df=test_df,
            model_config=model_config,
        )
        df = load_model_predictions(inference_stage, predictions_cache, labels_cache, text_cache)
        col2.write(df, unsafe_allow_html=True)


def model_interpretability_design(class_label_to_index: dict[int, str]):
    """Define the design of the model interpretability section."""
    test_classes = list(class_label_to_index.values())
    class_label = st.selectbox("Class to Explain predictions", test_classes)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        col1.subheader("**Bert pre-trained model with a classification head**")
        shap_values_cache = data_paths.bert_based_model_cached_shap_path
        shap_values = load_results(shap_values_cache)
        shap_plt = WikiTaxonomyClassifier.plot_shap_values(shap_values, class_label)
        st.pyplot(shap_plt)

    with col2:
        col2.subheader("**TD-IDF Logistic regression classifier**")
        shap_values_cache = data_paths.tfidf_logistic_model_cached_shap_path
        shap_values = load_results(shap_values_cache)
        fig, ax = plt.subplots()
        reversed_dict = {v: k for k, v in class_label_to_index.items()}
        shap.plots.bar(shap_values[:, :, reversed_dict[class_label]], show=False)
        st.pyplot(fig)


def load_model_predictions(inference_stage, predictions_cache, labels_cache, text_cache):
    """Load model predictions, labels and test set to create a dataframe with model predictions."""
    # predictions
    predictions = load_results(predictions_cache)
    predicted_labels = inference_stage.get_label_class(predictions)
    # labels
    labels_cache = load_results(labels_cache)
    labels_class = inference_stage.get_label_class(labels_cache)
    # text
    text_cache = load_results(text_cache)
    df = pd.DataFrame(
        {"text": text_cache, "PredictedClass": predicted_labels, "ActualClass": labels_class}
    )
    return df


def offline_evaluation_design(class_label_to_index: dict[int, str], confidence: float):
    """Define the design of the offline evaluation section in the streamlit application."""
    col1, col2 = st.columns(2, gap="large")
    with col1:
        col1.subheader("**Bert pre-trained model with a classification head**")
        evaluate_col_1 = Evaluate(
            cached_predictions_path=data_paths.bert_based_model_cached_predictions_path,
            cached_labels_path=data_paths.cached_labels_path,
            cached_shap_values_path=data_paths.bert_based_model_cached_shap_path,
            class_label_to_index=class_label_to_index,
        )
        offline_eval_table_design(evaluate_col_1, col1, confidence)

    with col2:
        col2.subheader("**TD-IDF Logistic regression classifier**")
        evaluate_col_2 = Evaluate(
            cached_predictions_path=data_paths.tfidf_logistic_model_cached_predictions_path,
            cached_labels_path=data_paths.cached_labels_path,
            cached_shap_values_path=data_paths.tfidf_logistic_model_cached_shap_path,
            class_label_to_index=class_label_to_index,
        )
        offline_eval_table_design(evaluate_col_2, col2, confidence)


def offline_eval_table_design(evaluate: Evaluate, col, confidence):
    """Calculate the metrics to show in the offline evaluation section in streamlit application."""
    col.write("**AUROC metric**")
    auroc = evaluate.calculate_auroc(evaluate.cached_predictions, evaluate.cached_labels)
    col.write(auroc)
    col.write(f"Macro AUROC for all categories {np.around(auroc.AUROC.mean(), 4)}")
    col.write("**Classification report**")
    class_report_df = evaluate.calculate_classification_report(
        evaluate.cached_predictions, evaluate.cached_labels, confidence
    )
    # Render the DataFrame as an HTML table in Streamlit
    col.write(class_report_df, unsafe_allow_html=True)
