# Base
from pathlib import Path

# logging and loading bars
import logging

# Our own code
from preprocessing import TweetProcessor

# Argument parsing setup
import argparse

# Training
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process and train on tweet data")
    parser.add_argument('--ignore-saved', action='store_true',
                        help='Ignore saved data and models.')
    parser.add_argument('--train-files', type=int, default=None,
                        help='Number of files to use for training (None for all files)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--kaggle', action='store_true',
                        help='Will train on full dataset and evaluate the model')
    parser.add_argument('--hyper-search', action='store_true',
                        help='Perform an hyper parameter grid search and log results')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize hyper search results')
    parser.add_argument('--load', default=None, type=str,
                        help='Give a folder to load the model from')
    return parser.parse_args()


# Set up logging and argument parsing
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(message)s')
args = parse_args()

# Set logging level based on verbose flag
if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)

# Data dir
dir = Path("/Data/comev_data_challenge/")
if not dir.exists():
    dir = Path("./")
train_dir = dir / "train_tweets"
if not train_dir.exists():
    logging.error("Could not find train tweets repository in ", train_dir)
    exit(1)
eval_dir = dir / "eval_tweets"
if not eval_dir.exists():
    logging.error("Could not find eval tweets repository in ", eval_dir)
    exit(1)

""" --- PREPROCESSING --- """
train_out_dir = dir / "preprocessed_train"
train_out_dir.mkdir(exist_ok=True)
tweet_processor = TweetProcessor(train_dir=train_dir, output_dir=train_out_dir)
df = tweet_processor.run(args.train_files)

""" --- TRAINING --- """


# --- Dataset Preparation ---
class TweetDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tweet = self.data.iloc[index]["Tweet"]
        label = self.data.iloc[index]["EventType"]
        encoding = self.tokenizer(
            tweet,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(p):
    predictions, labels = p
    # Convert logits to class predictions
    preds = np.argmax(predictions, axis=1)

    # Compute precision, recall, f1-score, and accuracy
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# Train-Test Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# --- Model Preparation ---
model_path = Path(args.load) if args.load else None
if model_path and model_path.exists():
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    train_dataset = TweetDataset(train_df, tokenizer, max_length=128)
    val_dataset = TweetDataset(val_df, tokenizer, max_length=128)
    # Evaluate on test set
    logging.info("Evaluating...")
    test_results = trainer.evaluate(eval_dataset=val_dataset)
    print(test_results)
    logging.info("Done")
else:
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = TweetDataset(train_df, tokenizer, max_length=128)
    val_dataset = TweetDataset(val_df, tokenizer, max_length=128)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)


# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=200,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --- Model Training ---
trainer.train()

logging.info("Training done.")
eval_results = trainer.evaluate()
logging.info(eval_results)


# For Kaggle submission
if not args.kaggle:
    logging.info("Flag --kaggle not used, not generating predictions.")
    exit(0)

# We read each file separately, we preprocess the tweets and then use the
# classifier to predict the labels. Finally, we concatenate all predictions
# into a list that will eventually be concatenated and exported to be submitted
# on Kaggle.
# Iterate over files in the evaluation directory
logging.info("Now evaluating full model...")

eval_out_dir = dir / "preprocessed_eval"

tweet_processor = TweetProcessor(train_dir=eval_dir, output_dir=eval_out_dir)
eval_df = tweet_processor.run(args.train_files)

encoding_out_dir = dir / "eval_embeddings"
encoding_out_dir.mkdir(exist_ok=True)
embedding_processor = TweetEmbeddingProcessor(
    model_name='vinai/bertweet-base',
    batch_size=128,
    output_dir=encoding_out_dir
)
eval_df_with_embeddings = embedding_processor.run(eval_df)

preds = event_model.predict(eval_df_with_embeddings)

logging.info("Saving predictions to CSV files...")
preds .to_csv('predictions.csv', index=False)
logging.info("Predictions saved successfully.")
