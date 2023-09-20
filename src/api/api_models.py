"""API models defintion."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ClassifierModel(Enum):
    """Define the two classification model Enum."""

    BERT_CLASSIFIER = "bert-classifier"
    LOGISTIC_CLASSIFIER = "logistic-classifier"


class BaseModelExternal(BaseModel):
    """Extend pydantic Base Model to print the dictionary items."""

    def __str__(self):
        """Define __str__ built-in method."""
        return str({x: y for x, y in self.__dict__.items() if y is not None})


class APIState(BaseModelExternal):
    """Define the API health state."""

    machine_name: str
    version: str


class BertTaxonomyResponse(BaseModelExternal):
    """Define Bert Taxonomy Response."""

    predicted_class_index: int
    predicted_class: str


class BertShapValuesResponse(BaseModelExternal):
    """Define BertShapValues response."""

    shap_values: Any


class TaxonomyRequest(BaseModelExternal):
    """Define TaxonomyRequest."""

    text: str
