# 🐦 Sub-event Detection in Twitter Streams

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-LSTM%20%2F%20Embeddings-orange)]()
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

> **Team Monad** | Kaggle Data Challenge Solution

## 📖 Project Overview
This project focuses on **Sub-event Detection** within high-volume Twitter streams. The goal is to classify tweets into specific sub-events or identify them as non-relevant in a temporal context. 

To ensure **scientific reproducibility** and robust benchmarking, this system simulates a real-time streaming environment using large-scale historical datasets (CSV), treating the problem as a sequence modeling task using **LSTMs** and **Custom Embeddings**.

## 🏗️ Project Architecture
The codebase is modularized to support experimentation and hyperparameter tuning.

```text
├── main.py              # Entry point: handles CLI args and pipeline orchestration
├── preprocessing.py     # ETL: Tokenization, cleaning, and handling missing data
├── embedding.py         # Vectorization: Implementation of learnable embeddings/Word2Vec
├── clf.py               # Model Definitions: LSTM & Classifier architecture
├── result_log.py        # Logging: Tracks experiments and metrics
├── data_challenge...pdf # Full scientific report and presentation
└── requirements.txt     # Dependencies
