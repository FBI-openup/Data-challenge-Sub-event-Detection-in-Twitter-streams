# 🐦 Sub-event Detection in Twitter Streams

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-LSTM%20%2F%20Embeddings-orange)]()
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

> **Team Monad** | Kaggle Data Challenge Solution
> -Work of team {Monad} in Data-challenge-Sub-event-Detection-in-Twitter-streams in kaggle

-link [Data-challenge-Sub-event-Detection-in-Twitter-streams](https://www.kaggle.com/competitions/sub-event-detection-in-twitter-streams)  

##  Project Overview
This project applies Deep Learning techniques to predict the presence of specific sub-events (e.g., 'Goal', 'Red Card', 'Penalty') in tweets posted during the 2010 and 2014 Football World Cups. Analyzing social media interactions during high-stakes tournaments reveals insights into crowd sentiment and collective audience reactions.

**The Task:**
The goal is to classify 1-minute time periods of tweet streams to determine if a relevant sub-event occurred (Label 1) or not (Label 0). The project involves extending a basic baseline to a robust model evaluated by **Accuracy**.

##  Approach & Architecture
We moved beyond the standard Logistic Regression baseline to implement a sequence modeling approach suitable for stream-like text data.

*   **Input:** JSON streams split by MatchID and PeriodID.
*   **Preprocessing:** Tokenization, cleaning, and temporal aggregation.
*   **Model:** Long Short-Term Memory (LSTM) network with custom learnable embeddings to capture temporal dependencies in crowd sentiment.
*   **Optimization:** Implemented Grid Search for hyperparameter tuning.

##  Data Structure
The dataset consists of tweets annotated with binary labels (`0` or `1`).
*   **Format:** `*.json` files.
*   **Columns:** `MatchID`, `PeriodID` (1-minute window), `EventType`, `Timestamp`, `Tweet`.

The code expects the following directory structure at the root or specified via CLI:
```text
/Data/
├── train_tweets/       # JSON files for training
└── eval_tweets/        # JSON files for evaluation

Usage
1. Setup Environment
Ensure all dependencies are installed in a virtual environment.
code
Bash
python3 -m venv .venv 
source .venv/bin/activate 
pip install -r requirements.txt
2. Training & Evaluation (End-to-End)
To train the LSTM model on the full dataset and evaluate the accuracy against the evaluation set:
code
Bash
python main.py --kaggle --verbose
3. Hyperparameter Search
To perform a grid search for optimal embedding dimensions and learning rates, and visualize the results:
code
Bash
python main.py --hyper-search --viz
4. Custom Data Directory
If your dataset is located in a different path:
code
Bash
python main.py --data-dir /path/to/your/json/files/
Note: The baseline.py script contains the initial Logistic Regression implementation acting as a benchmark, while the current branch contains the advanced deep learning (LSTM) solution.
