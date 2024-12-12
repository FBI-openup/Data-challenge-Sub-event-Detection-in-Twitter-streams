# Data-challenge-Sub-event-Detection-in-Twitter-streams

-Work of team {Monad} in Data-challenge-Sub-event-Detection-in-Twitter-streams in kaggle

-link [Data-challenge-Sub-event-Detection-in-Twitter-streams](https://www.kaggle.com/competitions/sub-event-detection-in-twitter-streams)  

## Usage

Please first ensure all the required libraries are installed on your system or run :
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Optional arguments:
  -h, --help            show this help message and exit
  --ignore-saved        Ignore saved data and models.
  --train-files TRAIN_FILES
                        Number of files to use for training (None for all files)
  --verbose             Enable verbose logging
  --kaggle              Will train on full dataset and evaluate the model
  --hyper-search        Perform an hyper parameter grid search and log results
  --viz                 Visualize hyper search results
  --data-dir DATA_DIR   Change data directory, default to /Data/username/

## Data

Our code expects data to be in two folders at  DATA_DIR or at the root of the project :
- `DATA_DIR/train_tweets/`
- `DATA_DIR/eval_tweets/`
