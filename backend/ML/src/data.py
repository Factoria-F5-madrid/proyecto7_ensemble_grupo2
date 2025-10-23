# ml/src/data.py
import os
import pickle
import numpy as np

def load_preprocessed(dataset_dir="data/desayuno_preprocessed"):
    """
    Load the preprocessed dataset that your teammate prepared.
    Expects a PKL that contains 'npz_files' and 'class_names', and the corresponding npz files.
    Returns:
      X: numpy array (N, H, W, C) or (N, H, W)
      y: numpy array (N,)
      class_names: list of class strings
    """
    PKL_PATH = os.path.join(dataset_dir, 'food101_desayuno_preprocessed.pkl')
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"{PKL_PATH} not found. Run download script or upload data.")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
    npz_files = data['npz_files']
    class_names = data.get('class_names', None)

    X_parts, y_parts = [], []
    for npz_file in npz_files:
        path = os.path.join(dataset_dir, npz_file)
        with np.load(path) as npz:
            X_parts.append(npz['X'])
            y_parts.append(npz['y'])
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y, class_names
