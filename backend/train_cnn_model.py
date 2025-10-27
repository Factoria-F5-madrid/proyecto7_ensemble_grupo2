"""
Script para entrenar un modelo CNN real usando Transfer Learning con MobileNetV2
Dataset: Food101 Desayuno (21 clases)
"""
import os
import sys

# Configurar TensorFlow ANTES de importarlo (evita bloqueos de mutex)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Importar TensorFlow
print("🔄 Inicializando TensorFlow (esto puede tardar 30-60 segundos)...")
try:
    import tensorflow as tf
    # Configuración adicional para evitar bloqueos
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    print("❌ Error: TensorFlow no está instalado.")
    print("   Ejecuta: pip install tensorflow==2.20.0")
    sys.exit(1)

print("✅ TensorFlow cargado correctamente!")
print()

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
PKL_PATH = '../notebooks/data/desayuno_preprocessed/food101_desayuno_preprocessed.pkl'
NPZ_DIR = '../notebooks/data/desayuno_preprocessed/npz_files/'
MODEL_SAVE_PATH = 'models/breakfast_cnn_model.h5'
CLASS_NAMES_PATH = 'models/class_names.pkl'
HISTORY_PATH = 'models/training_history.pkl'

# Hiperparámetros
IMG_SIZE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 50  # Aumentado de 30 a 50 para mejor aprendizaje
LEARNING_RATE = 0.001  # Se usará 0.0001 en fase 2 de fine-tuning
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

print("="*70)
print("🚀 ENTRENAMIENTO DE MODELO CNN PARA CLASIFICACIÓN DE DESAYUNOS")
print("="*70)

# ============================================================================
# PASO 1: CARGAR DATASET
# ============================================================================
print("\n[1/7] 🔄 Cargando dataset preprocesado...")

if not os.path.exists(PKL_PATH):
    print(f"❌ Error: No se encuentra el archivo {PKL_PATH}")
    sys.exit(1)

# Cargar el pickle con la información del dataset
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

npz_files = data['npz_files']
class_names = data['class_names']

print(f"   ✅ Clases encontradas: {len(class_names)}")
print(f"   ✅ Clases: {', '.join(class_names[:5])}... (+{len(class_names)-5} más)")
print(f"   ✅ Archivos NPZ: {len(npz_files)}")

# ============================================================================
# PASO 2: CARGAR IMÁGENES Y ETIQUETAS
# ============================================================================
print("\n[2/7] 🔄 Cargando imágenes desde archivos NPZ...")

X_list = []
y_list = []

# Verificar que el directorio NPZ existe
if not os.path.exists(NPZ_DIR):
    print(f"❌ Error: No se encuentra el directorio {NPZ_DIR}")
    sys.exit(1)

for npz_file in tqdm(npz_files, desc="   Cargando NPZ"):
    # Extraer solo el nombre del archivo (sin el subdirectorio 'npz_files/')
    npz_filename = os.path.basename(npz_file)
    npz_path = os.path.join(NPZ_DIR, npz_filename)

    if not os.path.exists(npz_path):
        print(f"   ⚠️  Advertencia: {npz_filename} no encontrado en {NPZ_DIR}, saltando...")
        continue

    try:
        with np.load(npz_path) as npz_data:
            X_list.append(npz_data['X'])
            y_list.append(npz_data['y'])
    except Exception as e:
        print(f"   ⚠️  Error cargando {npz_file}: {e}")
        continue

if len(X_list) == 0:
    print("❌ Error: No se pudieron cargar imágenes del dataset")
    sys.exit(1)

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

print(f"\n   ✅ Dataset cargado exitosamente:")
print(f"      - Imágenes: {X.shape}")
print(f"      - Etiquetas: {y.shape}")
print(f"      - Total de muestras: {len(X):,}")

# Verificar distribución de clases
unique, counts = np.unique(y, return_counts=True)
print(f"      - Distribución por clase:")
for idx, count in zip(unique[:5], counts[:5]):
    print(f"        • {class_names[idx]}: {count} imágenes")
print(f"        ... (+{len(unique)-5} clases más)")

# ============================================================================
# PASO 3: PREPROCESAMIENTO
# ============================================================================
print("\n[3/7] 🔄 Preprocesando datos...")

# Normalizar imágenes (0-255 -> 0-1)
X = X.astype('float32') / 255.0
print("   ✅ Imágenes normalizadas (0-1)")

# Convertir etiquetas a one-hot encoding
num_classes = len(class_names)
y_categorical = keras.utils.to_categorical(y, num_classes)
print(f"   ✅ Etiquetas convertidas a one-hot encoding ({num_classes} clases)")

# Data Augmentation para mejorar generalización
print("   🔄 Configurando Data Augmentation...")
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
print("   ✅ Data Augmentation configurado (flip, rotation, zoom, contrast)")

# ============================================================================
# PASO 4: DIVIDIR EN TRAIN/VAL/TEST
# ============================================================================
print("\n[4/7] 🔄 Dividiendo dataset en train/val/test...")

# Primero separar test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_categorical,
    test_size=TEST_SPLIT,
    random_state=42,
    stratify=y
)

# Luego separar train y validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=VALIDATION_SPLIT/(1-TEST_SPLIT),
    random_state=42,
    stratify=y_temp.argmax(axis=1)
)

print(f"   ✅ Train set: {X_train.shape[0]:,} imágenes ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   ✅ Validation set: {X_val.shape[0]:,} imágenes ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"   ✅ Test set: {X_test.shape[0]:,} imágenes ({X_test.shape[0]/len(X)*100:.1f}%)")

# ============================================================================
# PASO 5: CREAR MODELO CNN CON TRANSFER LEARNING
# ============================================================================
print("\n[5/7] 🔄 Creando modelo CNN con Transfer Learning (MobileNetV2)...")

# Cargar modelo base pre-entrenado (sin la capa de clasificación)
base_model = MobileNetV2(
    input_shape=IMG_SIZE,
    include_top=False,
    weights='imagenet'
)

# ESTRATEGIA: Fine-tuning en 2 fases
# Fase 1: Entrenar solo las capas superiores (congelado)
# Fase 2: Descongelar y entrenar todo con learning rate bajo

# Inicialmente congelar el modelo base
base_model.trainable = False
print(f"   ✅ Modelo base MobileNetV2 cargado (capas congeladas para fase 1)")

# Construir modelo completo
# IMPORTANTE: Agregar capa de redimensionamiento porque las imágenes son 48x48 pero MobileNetV2 espera 224x224
model = keras.Sequential([
    layers.Input(shape=(48, 48, 3)),  # Input de 48x48
    data_augmentation,  # ← Data Augmentation (solo se aplica en training)
    layers.Resizing(224, 224),  # Redimensionar a 224x224 para MobileNetV2
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compilar modelo para FASE 1 (capas superiores)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print("\n   ✅ Arquitectura del modelo:")
print("   " + "="*66)
model.summary(print_fn=lambda x: print("   " + x))
print("   " + "="*66)

# Calcular parámetros
total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = total_params - trainable_params

print(f"\n   📊 Parámetros del modelo:")
print(f"      - Total: {total_params:,}")
print(f"      - Entrenables: {trainable_params:,}")
print(f"      - No entrenables: {non_trainable_params:,}")

# ============================================================================
# PASO 6: ENTRENAR MODELO
# ============================================================================
print("\n[6/7] 🔄 Entrenando modelo...")
print(f"   ⏱️  Esto puede tardar entre 10-30 minutos dependiendo de tu hardware...")

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Entrenar FASE 1: Solo capas superiores
print("\n   🔄 FASE 1: Entrenando capas superiores (base congelado)...")
history_phase1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # Solo 10 épocas para fase 1
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# FASE 2: Descongelar y Fine-tuning completo
print("\n   🔄 FASE 2: Fine-tuning completo (descongelando modelo base)...")

# Descongelar el modelo base
base_model.trainable = True
print(f"   ✅ Modelo base descongelado")

# Recompilar con learning rate MUY bajo para fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),  # 10x más bajo
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

# Entrenar FASE 2 con modelo completo
history_phase2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS - 10,  # Resto de épocas
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Combinar historiales
history = type('History', (), {})()
history.history = {
    key: history_phase1.history[key] + history_phase2.history[key]
    for key in history_phase1.history.keys()
}

# ============================================================================
# PASO 7: EVALUAR MODELO
# ============================================================================
print("\n[7/7] 🔄 Evaluando modelo en test set...")

test_results = model.evaluate(X_test, y_test, verbose=0)
test_loss = test_results[0]
test_accuracy = test_results[1]
test_top3_accuracy = test_results[2]

print(f"\n   📊 Resultados en Test Set:")
print(f"      - Loss: {test_loss:.4f}")
print(f"      - Accuracy: {test_accuracy*100:.2f}%")
print(f"      - Top-3 Accuracy: {test_top3_accuracy*100:.2f}%")

# Calcular métricas de overfitting
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
overfitting = abs(train_accuracy - val_accuracy) * 100

print(f"\n   📊 Análisis de Overfitting:")
print(f"      - Train Accuracy: {train_accuracy*100:.2f}%")
print(f"      - Val Accuracy: {val_accuracy*100:.2f}%")
print(f"      - Diferencia: {overfitting:.2f}%")

if overfitting < 5:
    print(f"      ✅ Overfitting controlado (<5%)")
else:
    print(f"      ⚠️  Overfitting alto (>5%) - Considera más regularización")

# ============================================================================
# GUARDAR MODELO Y ARTEFACTOS
# ============================================================================
print("\n💾 Guardando modelo y artefactos...")

# Crear directorio si no existe
os.makedirs('models', exist_ok=True)

# Guardar modelo
model.save(MODEL_SAVE_PATH)
print(f"   ✅ Modelo guardado en: {MODEL_SAVE_PATH}")

# Guardar nombres de clases
with open(CLASS_NAMES_PATH, 'wb') as f:
    pickle.dump(class_names, f)
print(f"   ✅ Nombres de clases guardados en: {CLASS_NAMES_PATH}")

# Guardar historial de entrenamiento
with open(HISTORY_PATH, 'wb') as f:
    pickle.dump(history.history, f)
print(f"   ✅ Historial de entrenamiento guardado en: {HISTORY_PATH}")

# ============================================================================
# GENERAR GRÁFICAS
# ============================================================================
print("\n📊 Generando gráficas de entrenamiento...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/training_curves.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Gráficas guardadas en: models/training_curves.png")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*70)
print("🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
print("="*70)
print(f"\n📁 Archivos generados:")
print(f"   • {MODEL_SAVE_PATH} - Modelo entrenado")
print(f"   • {CLASS_NAMES_PATH} - Nombres de clases")
print(f"   • {HISTORY_PATH} - Historial de entrenamiento")
print(f"   • models/training_curves.png - Gráficas de entrenamiento")

print(f"\n📊 Métricas Finales:")
print(f"   • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   • Test Top-3 Accuracy: {test_top3_accuracy*100:.2f}%")
print(f"   • Overfitting: {overfitting:.2f}%")

print(f"\n🚀 Próximos pasos:")
print(f"   1. Ejecuta el backend: uvicorn main:app --reload")
print(f"   2. Prueba la API con imágenes reales")
print(f"   3. Inicia el frontend para ver la aplicación completa")

print("\n" + "="*70)
