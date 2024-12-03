# Base
import os
import re
import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# logging and loading bars
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s - %(message)s')

# Data dir
dir = "/Data/comev_data_challenge/"
train_dir = os.path.join(dir, "train_tweets")
eval_dir = os.path.join(dir, "eval_tweets")

# Download some NLP models for processing, optional
nltk.download('stopwords')
nltk.download('wordnet')
# Load GloVe model with Gensim's API
# 200-dimensional GloVe embeddings
logging.info("Loading GloVe model...")
embeddings_model = api.load("glove-twitter-200")
logging.info("GloVe model loaded successfully.")


# Function to compute the average word vector for a tweet
def get_avg_embedding(tweet, model, vector_size=200):
    words = tweet.split()  # Tokenize by whitespace
    word_vectors = [model[word] for word in words if word in model]
    # If no words in the tweet are in the vocabulary, return a zero vector
    if not word_vectors:
        return np.zeros(vector_size)
    return np.mean(word_vectors, axis=0)


# Basic preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# Read all training files and concatenate them into one dataframe
li = []
output_dir = "preprocessed/"
os.makedirs(output_dir, exist_ok=True)
for filename in tqdm(os.listdir(train_dir), desc="Preprocessing tweets..."):
    csv_path = os.path.join(train_dir, filename)
    pkl_path = os.path.join(dir, output_dir, filename.replace(".csv", ".pkl"))

    # Check if pickle file exists
    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
    else:
        df = pd.read_csv(csv_path)
        df['Tweet'] = df['Tweet'].apply(preprocess_text)
        df.to_pickle(pkl_path)

    li.append(df)
df = pd.concat(li, ignore_index=True)

# Apply preprocessing to each tweet and obtain vectors
vector_size = 200  # Adjust based on the chosen GloVe model
tweet_vectors = np.vstack([get_avg_embedding(tweet, embeddings_model, vector_size)
                          for tweet in tqdm(df['Tweet'], desc="Generating vectors")])
tweet_df = pd.DataFrame(tweet_vectors)

# Attach the vectors into the original dataframe
period_features = pd.concat([df, tweet_df], axis=1)
# Drop the columns that are not useful anymore
period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
# Group the tweets into their corresponding periods.
# This way we generate an average embedding vector for each period
period_features = period_features.groupby(
    ['MatchID', 'PeriodID', 'ID']).mean().reset_index()

# We drop the non-numerical features and keep the embeddings values
# for each period
X = period_features.drop(
    columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
# We extract the labels of our training samples
y = period_features['EventType'].values

# Evaluating on a test set:

# We split our data into a training and test set that we can use to train our
# classifier without fine-tuning into the validation set and without submitting
# too many times into Kaggle
logging.info("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# We set up a basic classifier that we train and then calculate
# the accuracy on our test set
logging.info("Training Logistic Regression classifier...")
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X_train, y_train)
y_pred = clf.predict(X_test)
logging.info(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")

# For Kaggle submission

# This time we train our classifier on the full dataset that is available to us
logging.info("Training Logistic Regression classifier on full dataset...")
clf = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)

predictions = []
# We read each file separately, we preprocess the tweets and then use the
# classifier to predict the labels. Finally, we concatenate all predictions
# into a list that will eventually be concatenated and exported to be submitted
# on Kaggle.
for fname in tqdm(os.listdir(eval_dir), desc="Processing evaluation files"):
    csv_path = os.path.join(eval_dir, fname)
    val_df = pd.read_csv(csv_path)
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)

    tweet_vectors = np.vstack([get_avg_embedding(
        tweet, embeddings_model, vector_size) for tweet in val_df['Tweet']])
    tweet_df = pd.DataFrame(tweet_vectors)

    period_features = pd.concat([val_df, tweet_df], axis=1)
    period_features = period_features.drop(columns=['Timestamp', 'Tweet'])
    period_features = period_features.groupby(
        ['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    X = period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values

    preds = clf.predict(X)

    period_features['EventType'] = preds

    predictions.append(period_features[['ID', 'EventType']])

logging.info("Saving predictions to CSV files...")
pred_df = pd.concat(predictions)
pred_df.to_csv('logistic_predictions.csv', index=False)
logging.info("Predictions saved successfully.")
