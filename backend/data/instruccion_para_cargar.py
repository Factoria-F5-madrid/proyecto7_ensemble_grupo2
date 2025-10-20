import os
import zipfile
import pickle
import numpy as np
from tqdm import tqdm

# Configuración: ajustar estas rutas
ZIP_PATH = r'C:\Users\TuUsuario\Downloads\desayuno_preprocessed_archive.zip'  # Ruta del ZIP
DATASET_DIR = r'C:\Users\TuUsuario\DesayunoDataset\desayuno_preprocessed'  # Carpeta para descomprimir
PKL_PATH = os.path.join(DATASET_DIR, 'food101_desayuno_preprocessed.pkl')

# Descomprimir ZIP
os.makedirs(DATASET_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATASET_DIR)
print(f"✅ ZIP descomprimido en: {DATASET_DIR}")

# Cargar pickle
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)
npz_files = data['npz_files']
class_names = data['class_names']
print(f"✅ Clases: {len(class_names)}, Archivos NPZ: {len(npz_files)}")

# Cargar imágenes y etiquetas
X_list, y_list = [], []
for npz_file in tqdm(npz_files, desc="Cargando NPZ"):
    with np.load(os.path.join(DATASET_DIR, npz_file)) as npz_data:
        X_list.append(npz_data['X'])
        y_list.append(npz_data['y'])
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print(f"✅ Dataset cargado: Imágenes {X.shape}, Etiquetas {y.shape}")
# X y y están listos para el modelo (ejemplo: LightGBM)