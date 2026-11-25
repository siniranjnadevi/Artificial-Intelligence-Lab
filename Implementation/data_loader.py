"""
Data loader module for Phase 3 Model Training.
Ensures strict separation between training set (used for CV and model fitting)
and the independent final test set (used solely for final benchmarking).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .config import TRAIN_DATA_PATH, TEST_DATA_PATH, RANDOM_SEED, N_CV_FOLDS


def load_detection_data():
    """
    Loads binary disaster detection train and test datasets.
    Returns:
        X_train (pd.Series): Training text
        y_train (np.ndarray): Training binary labels (0/1)
        X_test (pd.Series): Final independent test text
        y_test (np.ndarray): Final independent test binary labels (0/1)
    """
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found at: {TRAIN_DATA_PATH}")
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found at: {TEST_DATA_PATH}")

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Impute any missing cleaned_text with raw text or empty string
    train_df['cleaned_text'] = train_df['cleaned_text'].fillna(train_df['text']).fillna('')
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna(test_df['text']).fillna('')

    X_train = train_df['cleaned_text'].astype(str)
    y_train = train_df['target'].values.astype(int)

    X_test = test_df['cleaned_text'].astype(str)
    y_test = test_df['target'].values.astype(int)

    print(f"[DataLoader] Loaded Training Data: {len(X_train)} samples (Disaster: {sum(y_train)}, Non-Disaster: {len(y_train)-sum(y_train)})")
    print(f"[DataLoader] Loaded Independent Test Data: {len(X_test)} samples (Disaster: {sum(y_test)}, Non-Disaster: {len(y_test)-sum(y_test)})")

    return X_train, y_train, X_test, y_test


def get_stratified_kfold_splits(X_train, y_train, n_splits=N_CV_FOLDS):
    """
    Returns StratifiedKFold generator strictly over the training data.
    The final test set is never seen by this splitter.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    return skf.split(X_train, y_train)


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = load_detection_data()
    print("DataLoader test passed successfully.")
