import re
import logging
import pandas as pd
import nltk
import emoji
from os import listdir
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm


class TweetProcessor:
    def __init__(self, train_dir, output_dir, lemmatizer=None, stop_words=None):
        # Initialize paths and resources
        self.train_dir = Path(train_dir)
        self.output_dir = Path(output_dir)

        if stop_words is None:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = stop_words

        self.lemmatizer = lemmatizer if lemmatizer else WordNetLemmatizer()

        # Make sure output directory exists
        self.output_dir.mkdir(exist_ok=True)

    def preprocess_text(self, text):
        """Preprocess the text data by cleaning and lemmatizing."""
        text = text.lower()
        # Replace URLs and mentions with BERTweet-compatible tokens
        text = re.sub(r'http\S+|www\S+|https\S+',
                      'URL', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', 'USER', text)  # Replace mentions with USER
        text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # Remove hashtags
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Encode emojis
        text = emoji.demojize(text)
        # Tokenization and lemmatization
        words = text.split()
        words = [self.lemmatizer.lemmatize(word)
                 for word in words if word not in self.stop_words]
        return ' '.join(words)

    def process_file(self, file_path):
        """Read, preprocess, and save the file as pickle if not already processed."""
        output_path = self.output_dir / file_path.name.replace(".csv", ".pkl")

        if output_path.exists():  # Check if pickle file exists
            return pd.read_pickle(output_path)

        # Process and save new pickle if it doesn't exist
        df = pd.read_csv(file_path)

        df['hash'] = df['Tweet'].apply(lambda x: hash(x))
        # df = df.drop_duplicates(subset='hash').drop(columns='hash')

        df['Tweet'] = df['Tweet'].apply(self.preprocess_text)
        df.to_pickle(output_path)
        return df

    def process_files_in_parallel(self, files):
        """Process a list of files in parallel and return the concatenated result."""
        with Pool() as pool:
            all_dfs = list(
                tqdm(pool.imap(self.process_file, files),
                     total=len(files), desc="Preprocessing tweets"))

        # Concatenate all DataFrames into a single DataFrame
        df = pd.concat(all_dfs, ignore_index=True)
        return df

    def run(self, train_files_max=None):
        """Main method to run the entire processing pipeline."""
        csv_files = [self.train_dir / file for file in listdir(self.train_dir)
                     if file.endswith(".csv")]
        if train_files_max:
            csv_files = csv_files[:train_files_max]

        logging.debug(f"Will save pickles to : {self.output_dir}")

        # Process files in parallel and return the concatenated DataFrame
        return self.process_files_in_parallel(csv_files)
