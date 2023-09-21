FROM python:3.10-slim

RUN mkdir /wiki-taxonomy-classifier

WORKDIR wiki-taxonomy-classifier

RUN  mkdir ./src/

COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y --no-install-recommends git gcc build-essential libpq-dev

RUN pip install -r requirements.txt

COPY ./src/model/best_models/bert_best_model.ckpt ./src/model/best_models/bert_best_model.ckpt
COPY ./src/model/cached_results/ ./src/model/cached_results
COPY ./src/api/ ./src/api
COPY ./src/model/wiki_taxonomy_classifier.py ./src/model/wiki_taxonomy_classifier.py
COPY ./src/pipelines/ ./src/pipelines/
COPY ./src/settings/ ./src/settings
COPY ./src/utils/ ./src/utils
COPY ./src/api_entrypoint_main.py ./src/
COPY ./src/app_dashboard_entrypoint.py ./src/
COPY ./src/frontend/ ./src/frontend

RUN  mkdir ./src/data

EXPOSE 8001
EXPOSE 8501
