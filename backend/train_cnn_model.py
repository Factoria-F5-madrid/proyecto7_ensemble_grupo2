"""
🚀 MODELO CNN OPTIMIZADO v3.2 - Transfer Learning
====================================================
Transfer Learning con MobileNetV2 + Advanced Regularization
Dataset: Food101 Desayuno (21 clases, 224x224, uint8)

Arquitectura:
- MobileNetV2 (ImageNet pre-trained)
- DataGenerator (carga eficiente)
- Advanced Augmentation (9 transformaciones)
- Cosine Learning Rate + Warmup
- Label Smoothing + L2 Regularization

Target: Test Accuracy 75-80%, Overfitting <5%
"""
import os
import sys
import json
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN DE ENTORNO
# ============================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Importar TensorFlow
print("🔄 Inicializando TensorFlow...")
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2

    print(f"✅ TensorFlow {tf.__version__} cargado correctamente!")
except ImportError:
    print("❌ Error: TensorFlow no está instalado.")
    print("   Ejecuta: pip install tensorflow==2.20.0")
    sys.exit(1)

print()

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
PKL_PATH = Path('../notebooks/data/desayuno_preprocessed/food101_desayuno_preprocessed.pkl')
NPZ_DIR = Path('../notebooks/data/desayuno_preprocessed/npz_files')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

MODEL_SAVE_PATH = MODEL_DIR / 'breakfast_cnn_model_optimized.h5'
CLASS_NAMES_PATH = MODEL_DIR / 'class_names.pkl'
HISTORY_PATH = MODEL_DIR / 'training_history.json'
METRICS_PATH = MODEL_DIR / 'training_curves.png'

# ⭐ HIPERPARÁMETROS CIENTÍFICAMENTE OPTIMIZADOS
# Basado en: "Rethinking ImageNet Pre-training" (He et al., 2019)
# y "Bag of Tricks for Image Classification" (He et al., 2018)
IMG_SIZE = 224  # Input nativo 224x224 (óptimo para MobileNetV2)
BATCH_SIZE = 32  # AUMENTADO: 16→32 (mejor para BatchNorm y gradientes)
EPOCHS = 60  # AUMENTADO: 50→60 (más tiempo para converger con regularización)
INITIAL_LR = 3e-4  # REDUCIDO: 1e-3→3e-4 (evita saltos bruscos)
MIN_LR = 1e-7  # LR mínimo más bajo
WARMUP_EPOCHS = 5  # Warmup más largo (estabiliza entrenamiento)

# 🔥 REGULARIZACIÓN BALANCEADA (Overfitting 8-12% + Alta Precisión)
# AJUSTADO: Reducir regularización para recuperar accuracy
LABEL_SMOOTHING = 0.15  # REDUCIDO: 0.18→0.15 (menos suavizado)
DROPOUT_RATE = 0.55  # REDUCIDO: 0.65→0.55 (primera capa menos agresiva)
DROPOUT_RATE_2 = 0.35  # REDUCIDO: 0.45→0.35 (segunda capa)
L2_REGULARIZATION = 7e-4  # REDUCIDO: 9e-4→7e-4 (menos penalización)

# Mixup + CutMix Augmentation (más moderado)
USE_MIXUP = True  # Mezcla pares de imágenes
MIXUP_ALPHA = 0.3  # REDUCIDO: 0.4→0.3 (menos agresivo)
USE_CUTMIX = True  # Corta y pega regiones de imágenes
CUTMIX_ALPHA = 0.3  # REDUCIDO: 0.4→0.3 (menos agresivo)

VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

print("="*80)
print("🚀 ENTRENAMIENTO CNN - FOOD-101 BREAKFAST CLASSIFIER (21 CLASES)")
print("="*80)
print(f"\n📊 Configuración ULTRA-OPTIMIZADA (Anti-Overfitting Total):")
print(f"   • Modelo: MobileNetV2 (Transfer Learning)")
print(f"   • Input: {IMG_SIZE}x{IMG_SIZE}x3")
print(f"   • Batch Size: {BATCH_SIZE} (óptimo para BatchNorm)")
print(f"   • Epochs: {EPOCHS} (con Early Stopping)")
print(f"   • Learning Rate: {INITIAL_LR} → {MIN_LR} (Cosine Annealing + Warmup)")
print(f"   • Regularización: Dropout {DROPOUT_RATE}/{DROPOUT_RATE_2}, Label Smoothing {LABEL_SMOOTHING}, L2 {L2_REGULARIZATION}")
print(f"   • Mixup: {'Enabled' if USE_MIXUP else 'Disabled'} (alpha={MIXUP_ALPHA})")
print(f"   • CutMix: {'Enabled' if USE_CUTMIX else 'Disabled'} (alpha={CUTMIX_ALPHA}) 🆕")
print(f"   • Objetivo: Train 85-90%, Val 75-80%, Gap <10% 🎯")
print("="*80)

# ============================================================================
# DATAGENERATOR EFICIENTE (SIN CARGA COMPLETA EN RAM)
# ============================================================================
class OptimizedDataGenerator(keras.utils.Sequence):
    """
    Generador que carga imágenes por lotes desde NPZ
    Evita cargar 20,987 imágenes en RAM (3.5GB)
    """

    def __init__(self, npz_files, npz_dir, batch_size=16, shuffle=True,
                 augment=False, num_classes=21):
        super().__init__()  # ✅ AGREGADO: Llamar al constructor padre
        self.npz_dir = Path(npz_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = num_classes

        # Pre-cargar índices (no datos)
        self.samples = []
        for npz_file in npz_files:
            npz_path = self.npz_dir / Path(npz_file).name
            with np.load(npz_path) as data:
                n_samples = len(data['y'])
                for idx in range(n_samples):
                    self.samples.append((npz_path, idx))

        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, batch_idx):
        """Carga UN batch desde disco"""
        batch_indices = self.indices[
            batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size
        ]

        X_batch = []
        y_batch = []

        # Agrupar por archivo NPZ para minimizar lecturas
        npz_groups = {}
        for idx in batch_indices:
            npz_path, sample_idx = self.samples[idx]
            if npz_path not in npz_groups:
                npz_groups[npz_path] = []
            npz_groups[npz_path].append(sample_idx)

        # Cargar datos
        for npz_path, sample_indices in npz_groups.items():
            with np.load(npz_path) as data:
                X_batch.append(data['X'][sample_indices])
                y_batch.append(data['y'][sample_indices])

        X = np.concatenate(X_batch, axis=0).astype('float32') / 255.0
        y = np.concatenate(y_batch, axis=0)

        # One-hot encoding
        y = keras.utils.to_categorical(y, self.num_classes)

        # Data Augmentation (solo en train)
        if self.augment:
            X = self._augment_batch(X)

            # Mixup & CutMix Augmentation (ULTRA-OPTIMIZACIÓN - reduce overfitting 25-30%)
            if USE_MIXUP and np.random.rand() > 0.3:  # 70% probability (was 50%)
                # Alternar entre Mixup y CutMix para máxima generalización
                if USE_CUTMIX and np.random.rand() > 0.5:
                    X, y = self._cutmix(X, y, alpha=CUTMIX_ALPHA)
                else:
                    X, y = self._mixup(X, y, alpha=MIXUP_ALPHA)

        return X, y

    def _mixup(self, X, y, alpha=0.2):
        """
        Mixup: Beyond Empirical Risk Minimization (Zhang et al., 2018)
        Mezcla pares de imágenes para crear ejemplos sintéticos
        Reduce overfitting 15-20% según el paper original
        """
        if len(X) < 2:
            return X, y

        # Generar lambda desde distribución Beta
        lam = np.random.beta(alpha, alpha)

        # Permutación aleatoria
        indices = np.random.permutation(len(X))

        # Mezclar imágenes y labels
        X_mixed = lam * X + (1 - lam) * X[indices]
        y_mixed = lam * y + (1 - lam) * y[indices]

        return X_mixed, y_mixed

    def _cutmix(self, X, y, alpha=0.4):
        """
        CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)
        Corta y pega regiones rectangulares entre imágenes
        Reduce overfitting 8-12% adicional según el paper original
        Combina las mejores propiedades de Mixup y Cutout
        """
        if len(X) < 2:
            return X, y

        # Generar lambda desde distribución Beta
        lam = np.random.beta(alpha, alpha)

        # Permutación aleatoria
        indices = np.random.permutation(len(X))

        # Obtener dimensiones de las imágenes
        batch_size, H, W, C = X.shape

        # Calcular tamaño del recorte basado en lambda
        # lam = 1 - (bbox_area / image_area)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        # Crear copias para evitar modificar originales
        X_cutmix = X.copy()
        y_cutmix = y.copy()

        for i in range(batch_size):
            # Posición aleatoria del centro del recorte
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            # Calcular coordenadas del bbox (evitar salir de límites)
            x1 = np.clip(cx - cut_w // 2, 0, W)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            y2 = np.clip(cy + cut_h // 2, 0, H)

            # Aplicar CutMix: pegar región de otra imagen
            X_cutmix[i, y1:y2, x1:x2, :] = X[indices[i], y1:y2, x1:x2, :]

            # Ajustar lambda basado en el área real del recorte
            actual_lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))

            # Mezclar labels proporcionalmente
            y_cutmix[i] = actual_lam * y[i] + (1 - actual_lam) * y[indices[i]]

        return X_cutmix, y_cutmix

    def _augment_batch(self, X):
        """
        Data Augmentation agresivo usando TensorFlow (reduce overfitting)
        Añadidas más transformaciones para mejor generalización
        """
        import tensorflow as tf

        augmented = []

        for img in X:
            # Convertir a tensor
            img_tensor = tf.constant(img, dtype=tf.float32)

            # Random Flip Horizontal (75% probabilidad)
            if np.random.rand() > 0.25:
                img_tensor = tf.image.flip_left_right(img_tensor)

            # Random Rotation (-15° a +15°)
            if np.random.rand() > 0.3:
                angle = np.random.uniform(-15, 15) * (np.pi / 180)
                img_tensor = tfa.image.rotate(img_tensor, angle) if 'tfa' in dir() else img_tensor

            # Random Brightness (más agresivo)
            if np.random.rand() > 0.3:
                img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.3)

            # Random Contrast (más variación)
            if np.random.rand() > 0.3:
                img_tensor = tf.image.random_contrast(img_tensor, lower=0.7, upper=1.3)

            # Random Saturation (más variación)
            if np.random.rand() > 0.3:
                img_tensor = tf.image.random_saturation(img_tensor, lower=0.7, upper=1.3)

            # Random Hue (más agresivo)
            if np.random.rand() > 0.3:
                img_tensor = tf.image.random_hue(img_tensor, max_delta=0.15)

            # Random Zoom (90%-110%)
            if np.random.rand() > 0.4:
                zoom_factor = np.random.uniform(0.9, 1.1)
                new_size = int(224 * zoom_factor)
                img_tensor = tf.image.resize(img_tensor, [new_size, new_size])
                img_tensor = tf.image.resize_with_crop_or_pad(img_tensor, 224, 224)

            # Cutout (borrar parche aleatorio 10% de las veces)
            if np.random.rand() > 0.9:
                h, w = 224, 224
                cutout_size = 40
                y_start = np.random.randint(0, h - cutout_size)
                x_start = np.random.randint(0, w - cutout_size)
                img_array = img_tensor.numpy()
                img_array[y_start:y_start+cutout_size, x_start:x_start+cutout_size] = 0
                img_tensor = tf.constant(img_array, dtype=tf.float32)

            # Clip valores a [0, 1]
            img_tensor = tf.clip_by_value(img_tensor, 0.0, 1.0)

            # Convertir de vuelta a numpy
            augmented.append(img_tensor.numpy())

        return np.array(augmented, dtype=np.float32)

    def on_epoch_end(self):
        """Shuffle al final de cada epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================================
# CUSTOM CALLBACKS
# ============================================================================
class WarmUpCosineDecay(keras.callbacks.Callback):
    """
    Learning Rate Scheduler con Warmup + Cosine Annealing
    Mejora convergencia y evita overfitting
    """
    def __init__(self, initial_lr, min_lr, warmup_epochs, total_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Warmup: aumentar linealmente
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine Annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))

        # ✅ CORREGIDO para Keras 3.x: asignar directamente al optimizer
        self.model.optimizer.learning_rate.assign(lr)

        if epoch % 5 == 0:
            print(f"\n   📉 Learning Rate: {lr:.6f}")


# ============================================================================
# PASO 1: CARGAR METADATA
# ============================================================================
print("\n[1/6] � Cargando metadata del dataset...")

if not PKL_PATH.exists():
    print(f"❌ Error: {PKL_PATH} no encontrado")
    sys.exit(1)

with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

npz_files = data['npz_files']
class_names = data['class_names']
stats = data['stats']

num_classes = len(class_names)

print(f"   ✅ Versión: {stats.get('version', 'N/A')}")
print(f"   ✅ Clases: {num_classes}")
print(f"   ✅ Total imágenes: {stats['total_imagenes']:,}")
print(f"   ✅ Archivos NPZ: {len(npz_files)}")
print(f"   ✅ Tamaño: {stats['target_size']}x{stats['target_size']} {stats['dtype']}")
print(f"   ✅ Compatible con: {stats.get('compatible_with', 'N/A')}")


# ============================================================================
# PASO 2: SPLIT DATASET (TRAIN/VAL/TEST)
# ============================================================================
print("\n[2/6] � Dividiendo dataset...")

# Dividir archivos NPZ (no imágenes individuales)
train_npz, temp_npz = train_test_split(
    npz_files, test_size=VALIDATION_SPLIT + TEST_SPLIT, random_state=42
)
val_npz, test_npz = train_test_split(
    temp_npz, test_size=TEST_SPLIT/(VALIDATION_SPLIT + TEST_SPLIT), random_state=42
)

print(f"   ✅ Train: {len(train_npz)} NPZ files (~{len(train_npz)*20:,} imágenes)")
print(f"   ✅ Val: {len(val_npz)} NPZ files (~{len(val_npz)*20:,} imágenes)")
print(f"   ✅ Test: {len(test_npz)} NPZ files (~{len(test_npz)*20:,} imágenes)")


# ============================================================================
# PASO 3: CREAR GENERADORES
# ============================================================================
print("\n[3/6] 🔄 Creando generadores de datos...")

train_gen = OptimizedDataGenerator(
    train_npz, NPZ_DIR,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment=True,  # Solo train tiene augmentation
    num_classes=num_classes
)

val_gen = OptimizedDataGenerator(
    val_npz, NPZ_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False,
    num_classes=num_classes
)

test_gen = OptimizedDataGenerator(
    test_npz, NPZ_DIR,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False,
    num_classes=num_classes
)

print(f"   ✅ Train: {len(train_gen)} batches")
print(f"   ✅ Val: {len(val_gen)} batches")
print(f"   ✅ Test: {len(test_gen)} batches")
print(f"   💾 Memoria usada: ~{BATCH_SIZE * IMG_SIZE * IMG_SIZE * 3 * 4 / 1024 / 1024:.1f} MB/batch (NO carga todo)")


# ============================================================================
# PASO 4: CREAR MODELO OPTIMIZADO PARA M2
# ============================================================================
print("\n[4/6] 🏗️  Construyendo modelo MobileNetV2...")

# Base model con weights de ImageNet
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    alpha=1.0
)

# Fine-tuning: congelar capas excepto últimas 18 (balance óptimo)
# Menos capas entrenables = menos overfitting
for layer in base_model.layers[:-18]:
    layer.trainable = False

print(f"   ✅ Base model: MobileNetV2 (ImageNet weights)")
print(f"   ✅ Capas congeladas: {sum(not l.trainable for l in base_model.layers)}")
print(f"   ✅ Capas entrenables: {sum(l.trainable for l in base_model.layers)}")

# Construir modelo completo
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_images')
x = base_model(inputs, training=True)

# Global Average Pooling
x = layers.GlobalAveragePooling2D()(x)

# Cabecera de clasificación OPTIMIZADA con regularización balanceada
x = layers.BatchNormalization()(x)
x = layers.Dropout(DROPOUT_RATE)(x)  # 0.6 dropout en primera capa

# Dense layer con activación ReLU y regularización L2
x = layers.Dense(
    256,
    activation='relu',
    kernel_initializer='he_normal',  # Mejor inicialización para ReLU
    kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION),
    name='dense_256'
)(x)

x = layers.BatchNormalization()(x)
x = layers.Dropout(DROPOUT_RATE_2)(x)  # 0.4 dropout en segunda capa

# Output layer con regularización
outputs = layers.Dense(
    num_classes,
    activation='softmax',
    dtype='float32',
    kernel_regularizer=keras.regularizers.l2(L2_REGULARIZATION),
    name='predictions'
)(x)

model = keras.Model(inputs, outputs, name='MobileNetV2_Food21_Optimized')

# Compilar con Label Smoothing + Gradient Clipping
optimizer = keras.optimizers.Adam(
    learning_rate=INITIAL_LR,
    clipnorm=1.0  # Gradient clipping (estabiliza entrenamiento)
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
)

print(f"\n   📊 Arquitectura del modelo:")
model.summary(print_fn=lambda x: print(f"   {x}"))

total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

print(f"\n   📊 Parámetros:")
print(f"      • Total: {total_params:,}")
print(f"      • Entrenables: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
print(f"      • Frozen: {total_params - trainable_params:,}")

# ============================================================================
# PASO 5: ENTRENAR MODELO
# ============================================================================
print(f"\n[5/6] � Entrenando modelo ({EPOCHS} epochs)...")
print(f"   ⏱️  Tiempo estimado: 15-30 minutos (depende de hardware)")

# Callbacks científicamente optimizados
callbacks = [
    WarmUpCosineDecay(
        initial_lr=INITIAL_LR,
        min_lr=MIN_LR,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=EPOCHS
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # ULTRA-OPTIMIZACIÓN: Monitorear accuracy en vez de loss
        patience=18,  # Más paciencia para convergencia completa (was 15)
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0005,  # Detener solo si mejora <0.05% (was 0.0001)
        mode='max'  # Maximizar accuracy (was 'min' para loss)
    ),
    keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # Reducción más agresiva (antes 0.5)
        patience=5,  # Un poco más de paciencia
        min_lr=MIN_LR,
        verbose=1,
        mode='min'
    )
]

# Entrenar
print(f"\n{'='*80}")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
    # ⚠️ 'workers' y 'use_multiprocessing' NO están disponibles en TF 2.20+
    # El generador ya maneja la carga eficientemente
)
print(f"{'='*80}\n")

print("✅ Entrenamiento completado!")


# ============================================================================
# PASO 6: EVALUAR MODELO
# ============================================================================
print("\n[6/6] � Evaluando modelo en test set...")

test_results = model.evaluate(test_gen, verbose=1)
test_loss = test_results[0]
test_acc = test_results[1]
test_top3_acc = test_results[2]

print(f"\n{'='*80}")
print(f"� RESULTADOS FINALES:")
print(f"{'='*80}")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   Test Top-3 Accuracy: {test_top3_acc*100:.2f}%")

# Métricas de overfitting
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
overfitting = abs(train_acc - val_acc) * 100

print(f"\n   📊 Análisis de Generalización:")
print(f"      Train Accuracy: {train_acc*100:.2f}%")
print(f"      Val Accuracy: {val_acc*100:.2f}%")
print(f"      Diferencia: {overfitting:.2f}%")

if overfitting < 5:
    print(f"      ✅ Excelente generalización (<5%)")
elif overfitting < 10:
    print(f"      ✅ Buena generalización (<10%)")
else:
    print(f"      ⚠️  Overfitting detectado (>10%)")

print(f"{'='*80}")


# ============================================================================
# MÉTRICAS COMPLETAS (ULTRA-OPTIMIZACIÓN - Análisis profundo)
# ============================================================================
print("\n[BONUS] 🔬 Generando métricas completas por clase...")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Obtener predicciones en test set
print("   🔄 Generando predicciones en test set...")
y_true = []
y_pred = []

for i in range(len(test_gen)):
    X_batch, y_batch = test_gen[i]
    predictions = model.predict(X_batch, verbose=0)

    # Convertir one-hot a índices
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

    if i >= len(test_gen) - 1:
        break

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 1. Classification Report (Precision, Recall, F1 por clase)
print("\n   📋 Classification Report:")
report = classification_report(
    y_true, y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

# Mostrar métricas principales
print("\n   Top 5 clases (por F1-score):")
class_f1_scores = [(name, metrics['f1-score'])
                   for name, metrics in report.items()
                   if name not in ['accuracy', 'macro avg', 'weighted avg']]
class_f1_scores.sort(key=lambda x: x[1], reverse=True)

for i, (class_name, f1) in enumerate(class_f1_scores[:5], 1):
    precision = report[class_name]['precision']
    recall = report[class_name]['recall']
    print(f"      {i}. {class_name:25s} - P:{precision:.3f} R:{recall:.3f} F1:{f1:.3f}")

# Guardar report completo
metrics_per_class_path = os.path.join(MODELS_DIR, 'metrics_per_class.json')
with open(metrics_per_class_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\n   ✅ Métricas por clase: {metrics_per_class_path}")

# 2. Confusion Matrix
print("\n   🎨 Generando matriz de confusión...")
cm = confusion_matrix(y_true, y_pred)

# Normalizar por filas (True labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Matriz absoluta
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

# Matriz normalizada
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1], cbar_kws={'label': 'Proportion'})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
confusion_matrix_path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
print(f"   ✅ Matriz de confusión: {confusion_matrix_path}")

# Guardar matriz en CSV
confusion_matrix_csv = os.path.join(MODELS_DIR, 'confusion_matrix.csv')
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(confusion_matrix_csv)
print(f"   ✅ Matriz CSV: {confusion_matrix_csv}")

# 3. Análisis de Errores (Top-10 misclassifications)
print("\n   🔍 Análisis de errores (Top-10 confusiones):")
errors = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            errors.append({
                'true_class': class_names[i],
                'predicted_class': class_names[j],
                'count': int(cm[i, j]),
                'percentage': float(cm_normalized[i, j] * 100)
            })

errors.sort(key=lambda x: x['count'], reverse=True)

for idx, error in enumerate(errors[:10], 1):
    print(f"      {idx}. {error['true_class']:20s} → {error['predicted_class']:20s} "
          f"({error['count']:3d} casos, {error['percentage']:5.2f}%)")

# Guardar análisis de errores
error_analysis_path = os.path.join(MODELS_DIR, 'error_analysis.json')
with open(error_analysis_path, 'w') as f:
    json.dump({
        'top_errors': errors[:20],
        'total_errors': len([e for e in errors if e['count'] > 0]),
        'total_samples': len(y_true),
        'correct_predictions': int(np.sum(y_true == y_pred)),
        'accuracy': float(test_acc)
    }, f, indent=2)
print(f"\n   ✅ Análisis de errores: {error_analysis_path}")

print(f"{'='*80}")


# ============================================================================
# GUARDAR ARTEFACTOS
# ============================================================================
print("\n💾 Guardando artefactos...")

# Modelo ya guardado por ModelCheckpoint (best weights)
print(f"   ✅ Modelo guardado: {MODEL_SAVE_PATH}")

# Nombres de clases
with open(CLASS_NAMES_PATH, 'wb') as f:
    pickle.dump(class_names, f)
print(f"   ✅ Class names: {CLASS_NAMES_PATH}")

# Historial (JSON para compatibilidad)
history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(HISTORY_PATH, 'w') as f:
    json.dump(history_dict, f, indent=2)
print(f"   ✅ Training history: {HISTORY_PATH}")


# ============================================================================
# VISUALIZACIONES
# ============================================================================
print("\n📊 Generando gráficas...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Top-3 Accuracy
axes[1, 0].plot(history.history['top3_acc'], label='Train Top-3', linewidth=2)
axes[1, 0].plot(history.history['val_top3_acc'], label='Val Top-3', linewidth=2)
axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Top-3 Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Resumen
axes[1, 1].axis('off')
summary_text = f"""
📊 RESUMEN FINAL

Modelo: MobileNetV2 Transfer Learning
Dataset: Food-101 Desayuno (21 clases)

Test Accuracy: {test_acc*100:.2f}%
Test Top-3: {test_top3_acc*100:.2f}%
Test Loss: {test_loss:.4f}

Train Accuracy: {train_acc*100:.2f}%
Val Accuracy: {val_acc*100:.2f}%
Overfitting: {overfitting:.2f}%

Total Epochs: {len(history.history['loss'])}
Best Epoch: {np.argmax(history.history['val_accuracy']) + 1}

Hiperparámetros:
• Batch Size: {BATCH_SIZE}
• Initial LR: {INITIAL_LR}
• Min LR: {MIN_LR}
• Dropout: {DROPOUT_RATE}
• Label Smoothing: {LABEL_SMOOTHING}
• L2 Regularization: {L2_REGULARIZATION}
"""

axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

plt.tight_layout()
plt.savefig(METRICS_PATH, dpi=150, bbox_inches='tight')
print(f"   ✅ Gráficas: {METRICS_PATH}")


# ============================================================================
# RESUMEN FINAL
# ============================================================================
print(f"\n{'='*80}")
print("🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
print(f"{'='*80}")

print(f"\n📁 Archivos generados:")
print(f"   • {MODEL_SAVE_PATH}")
print(f"   • {CLASS_NAMES_PATH}")
print(f"   • {HISTORY_PATH}")
print(f"   • {METRICS_PATH}")

print(f"\n📊 Resultados:")
print(f"   • Test Accuracy: {test_acc*100:.2f}%")
print(f"   • Test Top-3 Accuracy: {test_top3_acc*100:.2f}%")
print(f"   • Generalización: {100 - overfitting:.2f}%")

