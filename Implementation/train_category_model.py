"""
Multi-Class Disaster Categorization (Track 2: Weakly Supervised Baseline):
Classifies disaster tweets into 6 categories:
['Other/Unknown', 'Cyclone', 'Fire', 'Flood', 'Earthquake', 'Landslide']

Academic Notice:
These labels are derived from retrieval keywords and represent a weakly supervised exploratory baseline,
NOT human ground truth. Evaluated on the held-out validation split with balanced class weights.
"""

import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from .config import (
    CATEGORY_TRAIN_PATH, CATEGORY_TEST_PATH,
    MODELS_DIR, METRICS_DIR, RANDOM_SEED
)


def train_category_model():
    print(f"\n{'='*70}")
    print("TRACK 2: 6-CLASS DISASTER CATEGORIZATION (WEAKLY SUPERVISED BASELINE)")
    print(f"{'='*70}")

    train_df = pd.read_csv(CATEGORY_TRAIN_PATH)
    test_df = pd.read_csv(CATEGORY_TEST_PATH)

    train_df['cleaned_text'] = train_df['cleaned_text'].fillna(train_df['text']).fillna('')
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna(test_df['text']).fillna('')

    X_train = train_df['cleaned_text'].astype(str)
    target_col = 'disaster_category' if 'disaster_category' in train_df.columns else 'disaster_type'
    y_train = train_df[target_col].values

    X_test = test_df['cleaned_text'].astype(str)
    test_target_col = 'disaster_category' if 'disaster_category' in test_df.columns else 'disaster_type'
    y_test = test_df[test_target_col].values

    classes = np.unique(y_train)
    print(f"[CategoryModel] Number of Classes: {len(classes)} -> {list(classes)}")
    print(f"[CategoryModel] Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    # Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Balanced Logistic Regression to handle 73:1 long-tail imbalance
    model = LogisticRegression(
        C=2.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_SEED
    )

    start_time = time.time()
    model.fit(X_train_vec, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"[CategoryModel] Test Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("\nDetailed Per-Class Performance:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save artifacts
    joblib.dump(model, MODELS_DIR / "category_classifier.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "category_vectorizer.joblib")
    
    with open(METRICS_DIR / "category_evaluation.json", "w") as f:
        json.dump({
            "track": "Track 2: Weakly Supervised Multi-Class Disaster Categorization",
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "training_time_seconds": training_time,
            "classification_report": report
        }, f, indent=2)

    print(f"[CategoryModel] Saved model and evaluation metrics successfully.")
    return model, report


if __name__ == "__main__":
    train_category_model()
