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

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(message)s')

# Data dir
dir = Path("/Data/comev_data_challenge/")
if not dir.exists():
    dir = Path("./")
train_dir = dir / "train_tweets"
eval_dir = dir / "eval_tweets"

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

with Pool() as pool:
    all_dfs = list(
        tqdm(pool.imap(process_file, csv_files),
             total=len(csv_files), desc="Preprocessing tweets"))

# Concatenate all DataFrames into a single DataFrame
df = pd.concat(all_dfs, ignore_index=True)

"""--- EMBEDDING ---"""

tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
model = AutoModel.from_pretrained('vinai/bertweet-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def encode_tweets_in_batches(texts, batch_size=128):
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

if embeddings_output_file.exists():
    logging.info("Reading embeddings from file...")
    with open(embeddings_output_file, 'rb') as file:
        df = pickle.load(file)
else:
    # Convert tweets to embeddings
    embeddings = encode_tweets_in_batches(
        df['Tweet'].tolist(), batch_size=128)

    # Add embeddings to DataFrame
    embedding_df = pd.DataFrame(embeddings, columns=[
                                f'Embedding_{i}' for i in range(embeddings.shape[1])])
    df = pd.concat([df, embedding_df], axis=1)
    df.to_pickle(embeddings_output_file)

# Normalize the timestamps to range [0, 1] to use as weights
# Ensure proper datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Timestamp_Normalized'] = (df['Timestamp'] - df['Timestamp'].min()) / (
    df['Timestamp'].max() - df['Timestamp'].min())


# Weighted mean function
def weighted_mean(group):
    weights = group['Timestamp_Normalized'].values
    embeddings = group[[
        col for col in group.columns if 'Embedding_' in col]].values
    weighted_embeddings = np.average(embeddings, axis=0, weights=weights)
    return pd.Series(weighted_embeddings, index=[f'Embedding_{i}' for i in range(embeddings.shape[1])])


# Group by MatchID, PeriodID, and ID, and compute weighted mean embeddings
period_features = df.groupby(['MatchID', 'PeriodID', 'ID']).apply(
    weighted_mean).reset_index()

# Extract features and labels for training
X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values
y = df['EventType'].values

# Evaluating on a test set:

# We split our data into a training and test set that we can use to train our
# classifier without fine-tuning into the validation set and without submitting
# too many times into Kaggle
clf_path = Path("models/RandomForestClassifier")
if clf_path.exists():
    logging.debug("Loading the Random Forest classifier from pickle file...")
    with open(clf_path, 'rb') as file:
        clf = pickle.load(file)
else:
    logging.info("Training a new Random Forest classifier...")
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Train a new Random Forest classifier
    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X_train, y_train)

    # Save the trained classifier to a pickle file
    with open(clf_path, 'wb') as file:
        pickle.dump(clf, file)
    logging.debug("Classifier saved to pickle file.")

    # Evaluate the classifier on the test set
    y_pred = clf.predict(X_test)
    logging.info(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# For Kaggle submission

# This time we train our classifier on the full dataset that is available to us
logging.info("Training Logistic Regression classifier on full dataset...")
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)

# We read each file separately, we preprocess the tweets and then use the
# classifier to predict the labels. Finally, we concatenate all predictions
# into a list that will eventually be concatenated and exported to be submitted
# on Kaggle.
# Iterate over files in the evaluation directory
predictions = []
for fname in tqdm(os.listdir(eval_dir), desc="Processing evaluation files"):
    csv_path = Path(eval_dir) / fname
    val_df = pd.read_csv(csv_path)

    # Preprocess tweets
    val_df['Preprocessed_Tweet'] = val_df['Tweet'].apply(preprocess_text)

    # Encode tweets in batches using BERTweet
    tweet_texts = val_df['Preprocessed_Tweet'].tolist()
    tweet_embeddings = encode_tweets_in_batches(tweet_texts, batch_size=32)

    # Add embeddings to DataFrame
    tweet_embeddings_df = pd.DataFrame(tweet_embeddings, columns=[
        f'Embedding_{i}' for i in range(tweet_embeddings.shape[1])
    ])
    val_df = pd.concat([val_df, tweet_embeddings_df], axis=1)

    # Drop unnecessary columns
    val_df = val_df.drop(columns=['Timestamp', 'Tweet', 'Preprocessed_Tweet'])

    # Group by MatchID, PeriodID, and ID, and compute mean embeddings
    period_features = val_df.groupby(
        ['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    # Extract features for prediction
    X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    # Predict event types using the classifier
    preds = clf.predict(X)
    period_features['EventType'] = preds

    # Save predictions
    predictions.append(period_features[['ID', 'EventType']])

# Combine all predictions into a single DataFrame
final_predictions = pd.concat(predictions, ignore_index=True)

logging.info("Saving predictions to CSV files...")
pred_df = pd.concat(predictions)
pred_df.to_csv('predictions.csv', index=False)
logging.info("Predictions saved successfully.")
