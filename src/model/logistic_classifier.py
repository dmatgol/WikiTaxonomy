"""A text classifier using TF-IDF vectorization and Logistic Regression."""
import logging
import pickle
import re

import nltk
import pandas as pd
import shap
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.settings.general import constants, data_paths


class TFIDFLogisticTextClassifier:
    """A text classifier using TF-IDF vectorization and Logistic Regression.

    Args:
        data (list): A list of text data for training.
        labels (list): A list of corresponding labels for the text data.
        include in the test split. Default is 0.2.
        random_state (int, optional): Seed for random state. Default is 0.
    """

    def __init__(self, train_data: pd.DataFrame, tf_idf_min_df: int = 5, random_state=0):
        """Classifier using TF-IDF vectorization and Logistic Regression."""
        self.download_nltk_packages()
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True, min_df=tf_idf_min_df, ngram_range=(1, 2), stop_words="english"
        )
        self.train_data = train_data
        self.logistic_regression_classifier = LogisticRegression(random_state=random_state)

    @staticmethod
    def download_nltk_packages():
        """Download necessary nltk packages."""
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")

    def train(self, save_path: str):
        """Train classifier."""
        # Create TF-IDF vectors
        features = [self.preprocess(article) for article in tqdm(self.train_data.text)]
        tfidf_features = self.vectorizer.fit_transform(features).toarray()
        labels = self.train_data[constants.label_column_encoded]

        # Train the Logistic Regression classifier using the tdidf features
        self.logistic_regression_classifier.fit(tfidf_features, labels)

        model_path = f"{save_path}/logistic_model.pkl"
        with open(model_path, "wb") as model_file:
            pickle.dump(self, model_file)

    def predict(self, test_data: pd.DataFrame):
        """Generate predictions for the test set."""
        features = [self.preprocess(article) for article in tqdm(test_data.text)]
        tfidf_test = self.vectorizer.transform(features).toarray()
        y_pred = self.logistic_regression_classifier.predict_proba(tfidf_test)
        return y_pred

    def compute_shap_values(self, test_data: pd.DataFrame):
        """Compute shap values for the model."""
        test_features = [self.preprocess(article) for article in tqdm(test_data.text)]
        train_features = [self.preprocess(article) for article in tqdm(self.train_data.text)]
        tfidf_train = self.vectorizer.fit_transform(train_features).toarray()
        tfidf_test = self.vectorizer.transform(test_features).toarray()
        explainer = shap.Explainer(
            self.logistic_regression_classifier,
            tfidf_train,
            feature_names=self.vectorizer.get_feature_names_out(),
        )
        shap_values = explainer(tfidf_test)
        with open(f"{data_paths.shap_values_cache}_logistic_classifier.pkl", "wb") as fp:
            pickle.dump(shap_values, fp)
            logging.info("dictionary saved successfully to file")
        return shap_values

    def preprocess(self, document):
        """Preprocess text.

        Includes lower case, removal stopwords and lemmatization.
        """
        wordnet_lemmatizer = WordNetLemmatizer()
        # change sentence to lower case
        document = document.lower()

        # tokenize into words
        words = word_tokenize(document)

        # remove stop words
        words = [word for word in words if word not in stopwords.words("english")]

        # Filter out words not containing only English alphabet characters
        words = [word for word in words if re.match(r"\b[a-zA-Z]+\b", word)]

        words = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in words]

        # join words to make sentence
        document = " ".join(words)

        return document
