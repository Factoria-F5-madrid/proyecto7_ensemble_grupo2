# train_xgb.py
import json
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import xgboost as xgb
from data import load_preprocessed

def run_cv(X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Flatten images
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)
    # optional PCA to reduce dims
    pca = PCA(n_components=200, random_state=random_state)
    X_red = pca.fit_transform(X_flat)

    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'verbosity': 1,
        'use_label_encoder': False,
    }
    model = xgb.XGBClassifier(**params)

    # simple CV baseline
    scores = cross_val_score(model, X_red, y, cv=skf, scoring='accuracy', n_jobs=-1)
    print("CV accuracy mean:", scores.mean(), "std:", scores.std())
    # save pca for inference
    joblib.dump(pca, "models/pca_for_xgb.joblib")
    return X_red, y, pca

if __name__ == "__main__":
    X, y, classes = load_preprocessed()
    run_cv(X, y)
