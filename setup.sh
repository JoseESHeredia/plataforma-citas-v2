#!/bin/bash
# Descarga el modelo base de spaCy para la extracción de entidades (PER, DATE, etc.).
# Esto es crucial para que procesador_nlp.py cargue "es_core_news_sm".
python -m spacy download es_core_news_sm