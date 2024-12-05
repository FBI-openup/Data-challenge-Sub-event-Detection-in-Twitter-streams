# Base
from pathlib import Path

# logging and loading bars
import logging

# Our own code
from preprocessing import TweetProcessor
from embedding import TweetEmbeddingProcessor
from clf import EventModel, ModelConfig, LSTMClassifier, EventDataset, DataLoader

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
    parser.add_argument('--kaggle', action='store_true',
                        help='Will train on full dataset and evaluate the model.')
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

""" --- ENCODING --- """
encoding_out_dir = dir / "train_embeddings"
encoding_out_dir.mkdir(exist_ok=True)
embedding_processor = TweetEmbeddingProcessor(
    model_name='vinai/bertweet-base',
    batch_size=128,
    output_dir=encoding_out_dir
)

df_with_embeddings = embedding_processor.run(df)

""" --- TRAINING --- """
conf = ModelConfig(batch_size=64, layers=1,
                   hidden_dim=128, learning_rate=0.0001,
                   epochs=1000, ignore_saved=args.ignore_saved)
event_model = EventModel(df_with_embeddings, conf)
accuracy = event_model.evaluate()
logging.info(f"Model accuracy on test set : {accuracy}")

# For Kaggle submission
# This time we train our classifier on the full dataset that is available to us
conf.test = False
event_model = EventModel(df_with_embeddings, conf)


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
