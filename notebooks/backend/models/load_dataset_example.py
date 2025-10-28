
# ============================================================================
# CÓDIGO PARA CARGAR EL DATASET PREPROCESADO (v2.0)
# ============================================================================

import pickle
import numpy as np
from pathlib import Path

# Ruta al archivo pickle (relativa desde el backend)
PKL_PATH = 'backend/models/food101_desayuno_preprocessed.pkl'

# Cargar metadata
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

npz_files = data['npz_files']
class_names = data['class_names']
stats = data['stats']

print(f"Clases: {len(class_names)}")
print(f"Imágenes totales: {stats['total_imagenes']:,}")
print(f"Archivos NPZ: {len(npz_files)}")
print(f"Tamaño de imágenes: {stats['target_size']}x{stats['target_size']}")
print(f"Tipo de dato: {stats['dtype']}")
print(f"Rango: {stats['range']}")

# Cargar todas las imágenes (⚠️ requiere RAM ~4GB)
X_list, y_list = [], []
base_path = Path('notebooks/data/desayuno_preprocessed')
for npz_file in npz_files:
    npz_filename = Path(npz_file).name
    npz_path = base_path / 'npz_files' / npz_filename
    with np.load(npz_path) as npz:
        X_list.append(npz['X'])
        y_list.append(npz['y'])

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X dtype: {X.dtype}")
print(f"Rango X: [{X.min()}, {X.max()}]")

# Normalizar (opcional para entrenamiento)
X_normalized = X.astype('float32') / 255.0
print(f"\nX normalizado: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
