# Base
from pathlib import Path

# logging and loading bars
import logging

# Our own code
from preprocessing import TweetProcessor
from embedding import TweetEmbeddingProcessor
from clf import EventModel, ModelConfig
from result_log import AccuracyLogger, HyperparameterResultsPlotter

# Argument parsing setup
import argparse

# Training
from scipy.stats import uniform, randint
from sklearn.model_selection import ParameterSampler



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
dir = Path("/Data/zhangboyuan/")
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
log_file = Path("model_accuracy_log.csv")
res_logger = AccuracyLogger(log_file, ModelConfig)

if args.hyper_search:
    param_dist = {
        'batch_size': randint(16, 65),
        'layers': randint(1, 4),
        'hidden_dim': randint(64, 513),
        'learning_rate': uniform(1e-5, 1e-2),
        'epochs': randint(500, 3001)
    }

    sampler = ParameterSampler(param_dist, n_iter=100, random_state=42)

    best_accuracy = 0

    try:
        for params in sampler:
            conf = ModelConfig(**params, ignore_saved=args.ignore_saved)
            event_model = EventModel(df_with_embeddings, conf)
            accuracy, precision, recall, f1 = event_model.evaluate()
            logging.info(f"{accuracy.item()}, {precision.item()}, {recall.item()}, {f1.item()}")
            if event_model.time > 0:
                res_logger.log(conf, accuracy, event_model.time)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = params
    except KeyboardInterrupt:
        logging.info("Stopping search.")

    logging.info(f"Best config: {best_config}, Best accuracy: {best_accuracy}")

else:
    # conf = ModelConfig(batch_size=32, layers=1,
    #                    hidden_dim=128, learning_rate=0.0001,
    #                    epochs=2000, ignore_saved=args.ignore_saved)
    conf = ModelConfig(batch_size=64, hidden_dim=128,
                       output_dim=2, learning_rate=0.001,
                       epochs=1000, ignore_saved=args.ignore_saved)
    event_model = EventModel(df_with_embeddings, conf)
    accuracy, precision, recall, f1 = event_model.evaluate()
    logging.info(
        f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    if event_model.time > 0:
        res_logger.log(conf, accuracy, event_model.time)

# Visualize results
if args.viz:
    plotter = HyperparameterResultsPlotter(log_file)
    # plotter.heatmap(index_param='hidden_dim', column_param='batch_size',
    #                 value_param='accuracy', title='Accuracy Heatmap')
    plotter.plot_all_features_vs_accuracy(
        ['epochs', 'learning_rate', 'hidden_dim', 'layers'],
        hue_param='time')
    plotter.three_param_plot(x_param='batch_size', y_param='learning_rate',
                             z_param='hidden_dim', color_param='accuracy', title='3D Visualization')


# For Kaggle submission
if not args.kaggle:
    logging.info("Flag --kaggle not used, not generating predictions.")
    exit(0)
# This time we train our classifier on the full dataset that is available to us
conf.test = False
conf.epochs = 1500
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
