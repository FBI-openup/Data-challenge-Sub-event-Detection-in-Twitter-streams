import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
from time import perf_counter


class ModelConfig:
    def __init__(self, clf_dir='models/', batch_size=32, hidden_dim=128, output_dim=2, learning_rate=0.0001, epochs=1000, layers=1, ignore_saved=False, test=True):
        """
        Configuration class to hold the model's hyperparameters and settings.

        Args:
        clf_dir (str): Directory where the model will be saved.
        batch_size (int): Batch size for training.
        hidden_dim (int): Hidden dimension of the LSTM.
        output_dim (int): Number of output classes.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs for training.
        ignore_saved (bool): Flag to ignore loading a saved model.
        test (bool): Flag to split data into test and train to enable evaluation.
        """
        self.clf_dir = Path(clf_dir)
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.ignore_saved = ignore_saved
        self.test = test


class EventModel:
    def __init__(self, df, config):
        """
        Initializes the EventModel with the given DataFrame and configuration.

        Parameters:
        - df: The input DataFrame containing features and labels.
        - config: An instance of ModelConfig with hyperparameters and settings.
        """
        self.df = df
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Prepare the data
        self.X, self.y = self._prepare_data(df)
        self.train_loader, self.test_loader = self._create_dataloaders(
            self.X, self.y)
        self.period_features = None

        self.time = 0

        torch.backends.cudnn.enabled = False

        # Initialize model, loss function, and optimizer
        self.model = LSTMClassifier(
            input_dim=self.X.shape[1],
            layers=self.config.layers,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(),
                              lr=config.learning_rate, weight_decay=1e-5)

        self.clf_path = self.config.clf_dir / \
            ("test" if self.config.test else "full")
        self.clf_path.mkdir(exist_ok=True)
        self.clf_path = self.clf_path / \
            f"clf_{self.config.layers}_{self.config.hidden_dim}_{self.config.epochs}_{self.config.batch_size}_{self.config.learning_rate}.pth"

        if not self.config.ignore_saved and self.clf_path.exists():
            self.model.load_state_dict(
                torch.load(self.clf_path, weights_only=True))
        else:
            self.train()
            torch.save(self.model.state_dict(), self.clf_path)

    def _prepare_data(self, df):
        """
        Prepares the features and labels from the DataFrame.

        Parameters:
        - df: The input DataFrame.

        Returns:
        - X: Feature matrix.
        - y: Label vector.
        """
        # Drop unnecessary columns and compute mean for grouping
        df['NumTweet'] = df.groupby(['MatchID', 'PeriodID', 'ID'])['Tweet'].transform('count')
        period_features = df.drop(columns=['Timestamp', 'Tweet']).groupby(
            ['MatchID', 'PeriodID', 'ID']).mean().reset_index()


        # Extract features and labels
        X = period_features.drop(
            columns=['MatchID', 'PeriodID', 'ID', 'EventType']).values
        y = period_features['EventType'].values.astype(int)

        return X, y

    def _create_dataloaders(self, X, y):
        """
        Splits the data into training and testing sets and creates DataLoaders.

        Parameters:
        - X: Feature matrix.
        - y: Label vector.

        Returns:
        - train_loader: DataLoader for training data.
        - test_loader: DataLoader for testing data.
        """
        if not self.config.test:
            train_loader = DataLoader(EventDataset(
                X, y), batch_size=self.config.batch_size, shuffle=True)
            return train_loader, None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        train_dataset = EventDataset(X_train, y_train)
        test_dataset = EventDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, test_loader

    def train(self):
        """
        Trains the model using the training data.
        """
        time = perf_counter()
        pbar = tqdm(range(self.config.epochs), desc="Running epochs")
        for epoch in pbar:
            self.model.train()
            running_loss = 0

            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(
                    self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            pbar.set_postfix_str(
                f"Loss: {running_loss / len(self.train_loader):.4f}")
        self.time = perf_counter() - time

    def evaluate(self):
        """
        Evaluates the model on the test data.

        Returns:
        - accuracy: The accuracy of the model on the test data.
        - precision: The precision of the model on the test data.
        - recall: The recall of the model on the test data.
        - f1: The F1 score of the model on the test data.
        """
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(
                    self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)

                _, predicted = torch.max(outputs, 1)
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # Calculate the metrics
        accuracy = (np.array(y_pred) == np.array(y_true)).mean()
        # Use 'binary' for binary classification
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')

        return accuracy, precision, recall, f1

    def predict(self, new_data):
        """
        Predict the event type for new input data.

        Args:
        new_data (DataFrame or array-like): The new data to run predictions on.

        Returns:
        predictions (array): The predicted labels (event types).
        """
        # Preprocess the new data (same as the training data preprocessing)
        # Prepare the new data in the same format as the training data
        new_data_features = self._prepare_eval_data(new_data)

        # Convert to DataLoader
        new_data_loader = DataLoader(EventDataset(new_data_features,
                                                  [0]*len(new_data_features)),
                                     batch_size=self.config.batch_size, shuffle=False)

        # Set the model to evaluation mode
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in new_data_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())

        self.period_features['EventType'] = all_preds
        return self.period_features[['ID', 'EventType']]

    def _prepare_eval_data(self, new_data):
        """
        Preprocess new data so that it matches the training data format.

        Args:
        new_data (DataFrame or array-like): New data to be preprocessed.

        Returns:
        processed_data (array-like): Preprocessed features ready for prediction.
        """
        # Here, preprocess the new data the same way you did for training data.
        # For instance, dropping columns, handling missing values, etc.
        #new_data['NumTweet'] = new_data.groupby(['MatchID', 'PeriodID', 'ID'])['Tweet'].transform('count')
        self.period_features = new_data.drop(columns=['Timestamp', 'Tweet'])
        self.period_features = self.period_features.groupby(
            ['MatchID', 'PeriodID', 'ID']).mean().reset_index()

        return self.period_features.drop(columns=['MatchID', 'PeriodID', 'ID']).values


class EventDataset(Dataset):
    def __init__(self, X, y):
        """
        Dataset for loading features and labels for training.

        Args:
        X (array-like): The feature matrix.
        y (array-like): The label vector.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """Return a sample and its label."""
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, layers, hidden_dim, output_dim, dropout=0.5):
        """
        Initialize the LSTM-based classifier.

        Args:
        input_dim (int): The number of features in the input.
        hidden_dim (int): The number of hidden units in the LSTM.
        output_dim (int): The number of output classes.
        """
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass through the LSTM network."""
        x, _ = self.lstm(x.unsqueeze(1))  # Add sequence length dimension
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)  # Use the last time-step output
        return x
