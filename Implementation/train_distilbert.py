"""
DistilBERT Contextual Deep Learning Model Training Script:
Extracts contextual embeddings from pretrained 'distilbert-base-uncased'
and trains a deep sequence classification head with dropout and weight decay.
Evaluated via 5-Fold Stratified Cross-Validation on the training set,
then evaluated once on the independent test set.
"""

import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from .config import MODELS_DIR, RANDOM_SEED, N_CV_FOLDS
from .data_loader import load_detection_data
from .feature_extractors import MultiRepresentationExtractor


class DistilBertNeuralHead(nn.Module):
    """
    Contextual Deep Neural Classification Head on top of DistilBERT [CLS] embeddings.
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class DistilBertClassifierWrapper:
    """
    Scikit-learn compatible wrapper for the DistilBERT model.
    """
    def __init__(self, extractor, epochs=25, batch_size=64, lr=1e-3, weight_decay=1e-2):
        self.extractor = extractor
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def fit(self, X_emb, y_train):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        self.model = DistilBertNeuralHead(input_dim=X_emb.shape[1], hidden_dim=256).to(self.device)
        
        # Compute class weights for loss
        class_counts = np.bincount(y_train)
        weights = torch.tensor([len(y_train) / (2.0 * c) for c in class_counts], dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = TensorDataset(torch.tensor(X_emb, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X_emb):
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_emb, dtype=torch.float32).to(self.device)
            logits = self.model(tensor_x)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities

    def predict(self, X_emb):
        probs = self.predict_proba(X_emb)
        return np.argmax(probs, axis=1)


def train_and_evaluate_distilbert():
    print(f"\n{'='*60}")
    print("Training and Evaluating Deep Learning Model: DistilBERT")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = load_detection_data()

    # Extract or load cached embeddings
    extractor = MultiRepresentationExtractor()
    cache_train = MODELS_DIR / "distilbert_embeddings_train.npy"
    cache_test = MODELS_DIR / "distilbert_embeddings_test.npy"

    X_train_emb = extractor.extract_distilbert_embeddings(X_train, cache_path=cache_train)
    X_test_emb = extractor.extract_distilbert_embeddings(X_test, cache_path=cache_test)

    # 5-Fold Stratified Cross Validation on Training Data
    print(f"[DistilBERT] Running {N_CV_FOLDS}-fold Stratified Cross-Validation on training set...")
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_emb, y_train), 1):
        fold_model = DistilBertClassifierWrapper(extractor, epochs=20, batch_size=64)
        fold_model.fit(X_train_emb[train_idx], y_train[train_idx])
        preds = fold_model.predict(X_train_emb[val_idx])
        fold_f1 = f1_score(y_train[val_idx], preds, average='macro')
        cv_f1s.append(fold_f1)
        print(f"  Fold {fold}/{N_CV_FOLDS} F1-Macro: {fold_f1:.4f}")

    mean_cv_f1 = float(np.mean(cv_f1s))
    std_cv_f1 = float(np.std(cv_f1s))
    print(f"[DistilBERT] CV F1-Macro: {mean_cv_f1:.4f} (+/- {std_cv_f1:.4f})")

    # Fit final model on full training set
    print("[DistilBERT] Training final model on entire training set...")
    final_model = DistilBertClassifierWrapper(extractor, epochs=25, batch_size=64)
    start_time = time.time()
    final_model.fit(X_train_emb, y_train)
    training_time = time.time() - start_time
    print(f"[DistilBERT] Training completed in {training_time:.3f} seconds.")

    # Measure test latency and metrics
    latency_start = time.time()
    y_pred = final_model.predict(X_test_emb)
    total_latency = time.time() - latency_start
    per_sample_latency = (total_latency / len(y_test)) * 1000.0
    y_proba = final_model.predict_proba(X_test_emb)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"[DistilBERT] Test Accuracy: {acc:.4f} | F1-Macro: {f1_macro:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(f"[DistilBERT] Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"[DistilBERT] FPR: {fpr:.4f} | FNR: {fnr:.4f} | Latency: {per_sample_latency:.3f} ms/sample")

    # Save model artifact
    save_path = MODELS_DIR / "distilbert_classifier.pt"
    torch.save(final_model.model.state_dict(), save_path)
    joblib.dump(final_model, MODELS_DIR / "distilbert_wrapper.joblib")
    print(f"[DistilBERT] Saved model artifact to: {save_path}")

    return {
        "model_name": "DistilBERT",
        "cv_f1_macro_mean": mean_cv_f1,
        "cv_f1_macro_std": std_cv_f1,
        "test_accuracy": acc,
        "test_precision_macro": prec_macro,
        "test_recall_macro": rec_macro,
        "test_f1_macro": f1_macro,
        "test_precision_weighted": prec_weighted,
        "test_recall_weighted": rec_weighted,
        "test_f1_weighted": f1_weighted,
        "test_roc_auc": roc_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "training_time_seconds": training_time,
        "prediction_latency_ms": per_sample_latency,
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist()
    }


if __name__ == "__main__":
    train_and_evaluate_distilbert()
