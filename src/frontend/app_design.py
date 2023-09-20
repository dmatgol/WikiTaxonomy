"""Streamlit frontend design application."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap.plots
import streamlit as st

from src.model.wiki_taxonomy_classifier import WikiTaxonomyClassifier
from src.settings.general import data_paths
from src.utils.utils import load_results

backend = "http://localhost:8001/"


def app_design():
    """Define the design of the streamlit application."""
    st.set_page_config(layout="wide")
    st.title("Wiki Taxonomy Category Dashboard")
    st.sidebar.title("Settings")
    confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.85)
    sentence_test_set = st.sidebar.selectbox(
        "Sentence or cached test dataset", ["TestSet", "Sentence"]
    )
    instructions = """
        1. Please choose on the side bar between predicting an input sentence or predict on the
        cached testset from https://www.kaggle.com/danofer/dbpedia-classes.
            Default will be an cached dataset.\n
        """
    st.write(instructions)

    # train_df, val_df, test_df, class_label_to_index = PreProcessing(
    #     train_path=data_paths.train_path,
    #     val_path=data_paths.val_path,
    #     test_path=data_paths.test_path,
    # ).run()

    if sentence_test_set == "Sentence":
        user_input = st.text_input("Enter some text:")
        sentence_prediction_design(user_input)
    else:
        st.write("**First 5 rows of the DataFrame**:")
        test_df = get_test_df()
        st.write(test_df.head(5))

        st.title("**Model Predictions**")
        st.write(
            "Establishing a baseline is a good practice when developing a new model as it allows "
            "you to compare its performance."
            " In this case, a Logistic Regression model using TF-IDF features has been employed."
        )
        model_predictions_design()

        st.title("**Model Interpretability**")
        model_interpretability_design()

        # MODEL EVALUATION
        st.title("**Model evaluation**")
        offline_evaluation_design(confidence)


def predict_input_sentence(user_input: str):
    """Predict input sentence using bert classifier."""
    url = backend + "predict/bert_classifier/"
    data = {"request": user_input}
    sentence_prediction = requests.post(url=url, params=data, timeout=8000)
    return sentence_prediction.json()


def compute_shap_values(user_input, predicted_class_idx: int):
    """Compute shapley values for bert classifier."""
    url = backend + "predict/bert_classifier/shap_values"
    data = {"request": user_input, "prediction_idx": predicted_class_idx}
    shap_plot = requests.post(url=url, params=data, timeout=8000)
    return shap_plot.text


def sentence_prediction_design(user_input: str):
    """Define the design of the sentence prediction section."""
    if user_input != "":
        st.write(f"**You typed**: **{user_input}**")
        st.write("**Please wait some seconds to see the model prediction and interpretability.**")
        # PREDICTION
        sentence_prediction = predict_input_sentence(user_input)
        predicted_class = sentence_prediction["predicted_class"]
        predicted_class_idx = sentence_prediction["predicted_class_index"]
        st.title(f"Model prediction is: {predicted_class}")
        # SHAP PLOT
        st.title("Model Interpretability")
        shap_plot = compute_shap_values(user_input, predicted_class_idx)
        st.components.v1.html(shap_plot, height=600, scrolling=True)


@st.cache_data
def get_test_df():
    """Get test dataset."""
    url = backend + "cache/test_df/"
    response = requests.get(url=url, timeout=8000)
    test_df_json = response.json()
    return pd.read_json(test_df_json["DataFrame"], orient="split")


@st.cache_data
def get_cached_predictions(model_name: str):
    """Get chached predictions for a given model name."""
    url = backend + "cache/cached_predictions/"
    data = {"model_name": model_name}
    response = requests.post(url=url, params=data, timeout=8000)
    test_df_json = response.json()
    return pd.read_json(test_df_json["DataFrame"], orient="split")


def model_predictions_design():
    """Define the design of the model prediction section."""
    col1, col2 = st.columns(2, gap="large")
    with col1:
        col1.subheader("**Bert pre-trained model with a classification head**")
        bert_cached_predictions_df = get_cached_predictions("bert-classifier")
        col1.write(bert_cached_predictions_df, unsafe_allow_html=True)

    with col2:
        col2.subheader("**TD-IDF Logistic regression classifier**")
        logistic_cached_predictions_df = get_cached_predictions("logistic-classifier")
        col2.write(logistic_cached_predictions_df, unsafe_allow_html=True)


@st.cache_data
def get_test_classes():
    """Get test classes."""
    url = backend + "cache/test_classes/"
    response = requests.get(url=url, timeout=8000)
    test_df_json = response.json()
    return test_df_json["Classes"]


@st.cache_data
def get_class_label_to_index():
    """Get class label to index dictionary."""
    url = backend + "cache/test_classes/"
    response = requests.get(url=url, timeout=8000)
    test_df_json = response.json()
    return test_df_json["ClassLabelDict"]


def model_interpretability_design():
    """Define the design of the model interpretability section."""
    test_classes = get_test_classes()
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
        class_label_to_index = get_class_label_to_index()
        reversed_dict = {v: k for k, v in class_label_to_index.items()}
        shap.plots.bar(shap_values[:, :, int(reversed_dict[class_label])], show=False)
        st.pyplot(fig)


@st.cache_data
def get_model_auroc(model_name: str):
    """Get model AUROC."""
    url = backend + "cache/model_evaluation/auroc/"
    data = {"model_name": model_name}
    response = requests.get(url=url, params=data, timeout=8000)
    test_df_json = response.json()
    return pd.read_json(test_df_json["DataFrame"], orient="split")


@st.cache_data
def get_model_classification_report(model_name: str, confidence: float):
    """Get model classification report."""
    url = backend + "cache/model_evaluation/classification_report/"
    data = {"model_name": model_name, "confidence": confidence}
    response = requests.get(url=url, params=data, timeout=8000)
    test_df_json = response.json()
    return pd.read_json(test_df_json["DataFrame"], orient="split")


def offline_evaluation_design(confidence: float):
    """Define the design of the offline evaluation section in the streamlit application."""
    col1, col2 = st.columns(2, gap="large")
    with col1:
        col1.subheader("**Bert pre-trained model with a classification head**")
        offline_eval_table_design("bert-classifier", col1, confidence)

    with col2:
        col2.subheader("**TD-IDF Logistic regression classifier**")
        offline_eval_table_design("logistic-classifier", col2, confidence)


def offline_eval_table_design(model_name: str, col, confidence):
    """Calculate the metrics to show in the offline evaluation section in streamlit application."""
    col.write("**AUROC metric**")
    auroc = get_model_auroc(model_name)
    col.write(auroc)
    col.write(f"Macro AUROC for all categories {np.around(auroc.AUROC.mean(), 4)}")
    col.write("**Classification report**")
    class_report_df = get_model_classification_report(model_name, confidence)
    col.write(class_report_df, unsafe_allow_html=True)
