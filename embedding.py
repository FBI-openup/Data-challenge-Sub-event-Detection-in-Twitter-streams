import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm


class TweetEmbeddingProcessor:
    def __init__(self, model_name='vinai/bertweet-base', device=None, batch_size=32, output_dir=Path('./embeddings'), ignore_saved=False):
        # Initialize paths and model
        self.model_name = model_name
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.ignore_saved = ignore_saved

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert_model = AutoModel.from_pretrained(self.model_name)
        self.bert_model.to(self.device)

        # Make sure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        self.embeddings_output_file = self.output_dir / "embeddings.pkl"

    def encode_tweets_in_batches(self, texts):
        """Encode the input tweets in batches and return the embeddings."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Tokenizing and embedding tweets"):
            batch_texts = texts[i:i + self.batch_size]
            tokens = self.tokenizer(batch_texts, padding=True,
                                    truncation=True, return_tensors="pt",
                                    max_length=128)

            # Move tensors to the appropriate device (GPU if available, otherwise CPU)
            tokens = {key: val.to(self.device) for key, val in tokens.items()}

            # Encode batch and get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            # Move embeddings back to CPU to save memory
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def process_embeddings(self, df):
        """Process and save embeddings to DataFrame."""
        if not self.ignore_saved and self.embeddings_output_file.exists():
            logging.info("Reading embeddings from file...")
            with open(self.embeddings_output_file, 'rb') as file:
                df = pickle.load(file)
        else:
            # Convert tweets to embeddings
            embeddings = self.encode_tweets_in_batches(df['Tweet'].tolist())

            # Add embeddings to DataFrame
            embedding_df = pd.DataFrame(embeddings, columns=[
                f'Embedding_{i}' for i in range(embeddings.shape[1])])
            df = pd.concat([df, embedding_df], axis=1)

            # Save embeddings to pickle
            with open(self.embeddings_output_file, 'wb') as file:
                pickle.dump(df, file)

        return df

    def run(self, df):
        """Main method to run the entire embedding process."""
        return self.process_embeddings(df)
