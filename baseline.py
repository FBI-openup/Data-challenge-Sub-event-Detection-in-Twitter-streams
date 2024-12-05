# Base
import os
import re
import nltk
import pandas as pd
import pickle
import numpy as np
from pathlib import Path

# Multithreading
from multiprocessing import Pool

# Preprocessing
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

# logging and loading bars
import logging
from tqdm import tqdm

# Training and models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch

# Argument parsing setup
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process and train on tweet data")
    parser.add_argument('--ignore_saved', action='store_true',
                        help='Ignore saved data and models.')
    parser.add_argument('--train_files', type=int, default=None,
                        help='Number of files to use for training (None for all files)')
    parser.add_argument('--eval_files', type=int, default=None,
                        help='Number of files to use for evaluation (None for all files)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for encoding tweets (default: 128)')
    parser.add_argument('--model', type=str, default='RandomForest',
                        choices=['RandomForest', 'LogisticRegression'],
                        help='Choose model for training (RandomForest or LogisticRegression)')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Use CPU instead of GPU for model training')
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

# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')

# Load resources once
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

"""--- PREPROCESSING ---"""


def preprocess_text(text):
    # Lowercasing is not required, as BERTweet handles case sensitivity well
    # Replace URLs and mentions with BERTweet-compatible tokens
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', 'USER', text)  # Replace mentions with USER
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # Remove hashtags
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Encode emojis
    text = emoji.demojize(text)
    # Tokenization and lemmatization
    words = text.split()
    words = [lemmatizer.lemmatize(word)
             for word in words if word not in stop_words]
    return ' '.join(words)


def process_file(file_path):
    """Read, preprocess, and save file as pickle if not already processed."""
    output_path = output_dir / file_path.name.replace(".csv", ".pkl")

    if output_path.exists():  # Check if pickle file exists
        return pd.read_pickle(output_path)

    # Process and save new pickle if it doesn't exist
    df = pd.read_csv(file_path)

    df['hash'] = df['Tweet'].apply(lambda x: hash(x))
    df = df.drop_duplicates(subset='hash').drop(columns='hash')

    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    df.to_pickle(output_path)
    return df


# Load and preprocess files in parallel
output_dir = dir / "preprocessed"
logging.debug("Will save pickles to : ", output_dir)
output_dir.mkdir(exist_ok=True)

csv_files = [train_dir / file for file in os.listdir(train_dir)
             if file.endswith(".csv")]
if args.train_files:
    csv_files = csv_files[:args.train_files]

with Pool() as pool:
    all_dfs = list(
        tqdm(pool.imap(process_file, csv_files),
             total=len(csv_files), desc="Preprocessing tweets"))

# Concatenate all DataFrames into a single DataFrame
df = pd.concat(all_dfs, ignore_index=True)

"""--- EMBEDDING ---"""

tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
model = AutoModel.from_pretrained('vinai/bertweet-base')
device = torch.device('cuda' if torch.cuda.is_available()
                      and not args.use_cpu else 'cpu')
model.to(device)


def encode_tweets_in_batches(texts, batch_size=args.batch_size):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing and embedding tweets"):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts, padding=True,
                           truncation=True, return_tensors="pt",
                           max_length=128)

        # Move tensors to appropriate device (GPU if available, otherwise CPU)
        tokens = {key: val.to(torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'))
            for key, val in tokens.items()}
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Encode batch and get embeddings
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        # Move embeddings back to CPU to save memory
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


embeddings_output_dir = dir / "embeddings"
embeddings_output_dir.mkdir(exist_ok=True)
embeddings_output_file = embeddings_output_dir / "embeddings.pkl"

if not args.ignore_saved and embeddings_output_file.exists():
    logging.info("Reading embeddings from file...")
    with open(embeddings_output_file, 'rb') as file:
        df = pickle.load(file)
else:
    # Convert tweets to embeddings
    embeddings = encode_tweets_in_batches(
        df['Tweet'].tolist(), batch_size=args.batch_size)

    # Add embeddings to DataFrame
    embedding_df = pd.DataFrame(embeddings, columns=[
                                f'Embedding_{i}' for i in range(embeddings.shape[1])])
    df = pd.concat([df, embedding_df], axis=1)
    df.to_pickle(embeddings_output_file)


def window_mean(df):
    # 1. Convert 'Timestamp' to datetime and create 'TimeWindow' (rounded down to the nearest minute)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df['TimeWindow'] = df['Timestamp'].dt.floor(
        'min')  # group for minute frequency

    # 2. Identify embedding columns
    embedding_cols = [
        col for col in df.columns if col.startswith('Embedding_')]

    # 3. Group by 'TimeWindow' and compute mean embeddings and tweet count
    window_features = df.groupby('ID', 'TimeWindow').agg(
        {
            **{col: 'mean' for col in embedding_cols},  # Mean of embeddings
            'Tweet': 'count'  # Count of tweets
        }
    ).reset_index()

    # 4. Rename 'Tweet' column to 'TweetCount'
    window_features.rename(columns={'Tweet': 'TweetCount'}, inplace=True)

    # 5. Determine if an event occurred in each TimeWindow
    # Assuming 'EventType' is 1 when an event occurred, 0 otherwise
    event_in_window = df.groupby('TimeWindow')['EventType'].max().reset_index()

    # Merge 'EventType' into 'window_features'
    window_features = window_features.merge(
        event_in_window, on='TimeWindow', how='left')

    # Fill NaN values in 'EventType' with 0 (no event occurred)
    window_features['EventType'] = window_features['EventType'].fillna(0)

    return window_features


# Prepare features and labels for training
# Features: embeddings and 'TweetCount'
period_features = df.drop(columns=['Timestamp', 'Tweet'])
period_features = period_features.groupby(
    ['MatchID', 'PeriodID', 'ID']).mean().reset_index()
X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID', 'EventType']).values

# Labels: 'EventType' (whether an event occurred in that minute)
y = period_features['EventType'].values.astype(int)

# Evaluating on a test set:
# We split our data into a training and test set that we can use to train our
# classifier without fine-tuning into the validation set and without submitting
# too many times into Kaggle
if args.model == 'RandomForest':
    clf = RandomForestClassifier(random_state=42, n_estimators=100)
elif args.model == 'LogisticRegression':
    clf = LogisticRegression(random_state=42, max_iter=1000)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)
logging.info(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")
logging.info(
    f"Classification Report:\n{classification_report(y_test, y_pred)}")


# For Kaggle submission
# This time we train our classifier on the full dataset that is available to us
clf_dir = Path("models/")
clf_path = clf_dir / args.model
if not args.ignore_saved and clf_path.exists():
    logging.debug("Loading the Random Forest classifier from pickle file...")
    with open(clf_path, 'rb') as file:
        clf = pickle.load(file)
else:
    logging.info(f"Training {args.model} classifier on full dataset...")
    if args.model == 'RandomForest':
        clf = RandomForestClassifier(random_state=42, n_estimators=100)
    elif args.model == 'LogisticRegression':
        clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X, y)
    # Save the trained classifier to a pickle file
    clf_dir.mkdir(exist_ok=True)
    with open(clf_path, 'wb') as file:
        pickle.dump(clf, file)
    logging.debug("Classifier saved to pickle file.")


# We read each file separately, we preprocess the tweets and then use the
# classifier to predict the labels. Finally, we concatenate all predictions
# into a list that will eventually be concatenated and exported to be submitted
# on Kaggle.
# Iterate over files in the evaluation directory
logging.info("Now evaluating full model...")
predictions = []
used_eval_files = 0
for fname in tqdm(os.listdir(eval_dir), desc="Processing evaluation files"):
    csv_path = Path(eval_dir) / fname
    eval_df = pd.read_csv(csv_path)

    # Preprocess tweets
    eval_df['Tweet'] = eval_df['Tweet'].apply(preprocess_text)

    # Convert tweets to embeddings
    embeddings = encode_tweets_in_batches(
        eval_df['Tweet'].tolist(), batch_size=args.batch_size)

    # Add embeddings to DataFrame
    embedding_df = pd.DataFrame(embeddings, columns=[
                                f'Embedding_{i}' for i in range(embeddings.shape[1])])
    eval_df = pd.concat([eval_df, embedding_df], axis=1)

    period_features = eval_df.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(
        ['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    # Predict event types using the classifier
    preds = clf.predict(X)
    period_features['EventType'] = preds

    # Save predictions
    predictions.append(period_features[['ID', 'EventType']])

    used_eval_files += 1
    if args.eval_files and used_eval_files == args.eval_files:
        break

# Combine all predictions into a single DataFrame
final_predictions = pd.concat(predictions, ignore_index=True)

logging.info("Saving predictions to CSV files...")
pred_df = pd.concat(predictions)
pred_df.to_csv('predictions.csv', index=False)
logging.info("Predictions saved successfully.")
