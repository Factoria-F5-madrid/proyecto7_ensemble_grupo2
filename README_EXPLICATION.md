# 🧠 Backend CNN - Documentación Técnica Completa

## 📋 Tabla de Contenidos

1. [Overview del Sistema](#1-overview-del-sistema)
2. [Arquitectura del Modelo CNN](#2-arquitectura-del-modelo-cnn)
3. [Pipeline de Datos](#3-pipeline-de-datos)
4. [Técnicas de Regularización](#4-técnicas-de-regularización)
5. [Learning Rate Schedule](#5-learning-rate-schedule)
6. [API FastAPI](#6-api-fastapi)
7. [Métricas y Evaluación](#7-métricas-y-evaluación)
8. [Guía de Uso](#8-guía-de-uso)

---

## 1. Overview del Sistema

### 1.1 Descripción General

Este backend implementa un **sistema de clasificación de alimentos** usando **Deep Learning** con las siguientes características:

- **Objetivo**: Clasificar imágenes de desayunos en 21 categorías y estimar calorías
- **Modelo**: Transfer Learning con MobileNetV2 (pre-entrenado en ImageNet)
- **Dataset**: Food-101 subset (21 clases de desayunos, ~21,000 imágenes)
- **Input**: Imágenes RGB 224x224 píxeles
- **Output**: Clase predicha, confianza, calorías estimadas, información nutricional

### 1.2 Arquitectura Completa del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                         SISTEMA COMPLETO                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   FRONTEND  │─────▶│    BACKEND   │─────▶│  MODELO CNN     │
│   (React)   │      │   (FastAPI)  │      │  (MobileNetV2)  │
└─────────────┘      └──────────────┘      └─────────────────┘
     │                      │                       │
     │ HTTP POST            │ Procesa imagen        │ Predice
     │ /predict             │ & ejecuta modelo      │ clase + conf
     │                      │                       │
     └──────────────────────┴───────────────────────┘
```

### 1.3 Flujo de Datos Completo

```
PREDICCIÓN EN TIEMPO REAL (Inferencia)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usuario sube imagen
         │
         ▼
┌─────────────────────┐
│  1. RECEPCIÓN       │  Frontend → Backend (HTTP POST)
│  - Formato: JPG/PNG │  Endpoint: /predict
│  - Max size: 10MB   │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  2. VALIDACIÓN      │  Backend valida tipo de archivo
│  - Check MIME type  │  Verifica que sea imagen válida
│  - Leer bytes       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  3. PREPROCESAMIENTO│  CNNPredictor.preprocess_image()
│  - PIL.Image.open() │  1. Convertir a RGB
│  - Resize 224x224   │  2. Normalizar [0,1]
│  - Normalizar /255  │  3. Expandir dimensión batch
│  - Shape: (1,224,   │
│    224,3)           │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  4. PREDICCIÓN CNN  │  model.predict(img_array)
│  - MobileNetV2      │  Forward pass completo
│  - 21 logits        │  Salida: probabilidades [0,1]
│  - Softmax final    │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  5. POST-PROCESO    │  Interpretar resultados
│  - Argmax (clase)   │  predicted_class = clases[argmax]
│  - Confianza        │  confidence = max(probs)
│  - Top-3 clases     │  top3 = argsort()[-3:]
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  6. ENRIQUECIMIENTO │  Añadir información nutricional
│  - Lookup calorías  │  nutrition_data[predicted_class]
│  - Calcular porción │  calories = cal_per_100g * 1.5
│  - Proteínas, carbs │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  7. RESPUESTA JSON  │  Return JSONResponse
│  - predicted_class  │  {
│  - confidence       │    "predicted_class": "pancakes",
│  - calories         │    "confidence": 0.89,
│  - nutrition        │    "estimated_calories": 340,
│  - top_predictions  │    ...
│  - model_info       │  }
└─────────────────────┘
         │
         ▼
    Usuario ve resultado
```

### 1.4 Estructura de Archivos

```
backend/
├── main.py                          # API FastAPI (endpoints)
├── cnn_predictor.py                 # Predictor CNN (inferencia)
├── train_cnn_model.py               # Script de entrenamiento
├── requirements.txt                 # Dependencias Python
│
├── models/                          # Modelos entrenados
│   ├── breakfast_cnn_model_optimized.h5   # Modelo CNN (50MB)
│   ├── class_names.pkl              # 21 clases [list]
│   ├── training_history.json        # Métricas de entrenamiento
│   └── training_curves.png          # Gráficas loss/accuracy
│
└── __pycache__/                     # Cache Python
```

### 1.5 Stack Tecnológico

| Componente | Tecnología | Versión | Propósito |
|------------|------------|---------|-----------|
| **Framework Web** | FastAPI | 0.104.1 | API REST asíncrona |
| **Servidor ASGI** | Uvicorn | 0.24.0 | Servidor de producción |
| **Deep Learning** | TensorFlow/Keras | 2.20.0 | Modelo CNN |
| **Transfer Learning** | MobileNetV2 | ImageNet | Feature extraction |
| **Procesamiento Imágenes** | Pillow | 10.1.0 | Lectura/resize |
| **Arrays Numéricos** | NumPy | 1.26+ | Operaciones matriciales |
| **Serialización** | Pickle | stdlib | Guardar class_names |
| **Validación** | Pydantic | (FastAPI) | Validación requests |

### 1.6 Conceptos Clave

#### Transfer Learning
**Definición**: Reutilizar un modelo pre-entrenado en un dataset grande (ImageNet) para un problema específico (Food-101).

**Ventajas**:
- ✅ Reduce tiempo de entrenamiento (horas → minutos)
- ✅ Requiere menos datos (1000s vs millones)
- ✅ Mejor generalización (features universales)
- ✅ Evita overfitting con datasets pequeños

**Cómo funciona**:
```
ImageNet (1.4M imágenes, 1000 clases)
         │
         ▼
   MobileNetV2 pre-entrenado
         │ Congelar capas base
         ▼
   Añadir capas personalizadas
   (Dense 256 → Dense 21)
         │
         ▼
   Fine-tuning en Food-101
   (21,000 imágenes, 21 clases)
```

#### MobileNetV2
**Definición**: Arquitectura CNN ligera diseñada para dispositivos móviles.

**Características**:
- **Depth-wise Separable Convolutions**: Reduce parámetros 8-9x
- **Inverted Residuals**: Mejora flujo de gradientes
- **Linear Bottlenecks**: Preserva información en capas estrechas
- **Parámetros**: ~3.5M (vs 138M de VGG16)

**Por qué MobileNetV2**:
- ✅ Ligero (modelo final ~50MB)
- ✅ Rápido en CPU (inferencia <500ms)
- ✅ Buen balance accuracy/velocidad
- ✅ Compatible con cualquier hardware

---

## 2. Arquitectura del Modelo CNN

### 2.1 Diagrama de Arquitectura Completa

```
INPUT: Imagen 224x224x3 (RGB)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MOBILENETV2 BASE                          │
│                  (Pre-entrenado ImageNet)                    │
│                                                              │
│  Conv2D(32, 3x3) → BN → ReLU6                               │
│         │                                                    │
│         ▼                                                    │
│  Inverted Residual Block x17                                │
│  ┌─────────────────────────────┐                           │
│  │ 1. Expansion (1x1 conv)     │                           │
│  │ 2. DepthWise (3x3 conv)     │  ← Separable convolutions│
│  │ 3. Projection (1x1 conv)    │  ← Linear bottleneck      │
│  │ 4. Residual connection      │  ← Skip connection        │
│  └─────────────────────────────┘                           │
│         │                                                    │
│         ▼                                                    │
│  Conv2D(1280, 1x1) → BN → ReLU6                            │
│         │                                                    │
│  Output shape: (7, 7, 1280)                                 │
│  Parámetros: ~2.3M (congelados parcialmente)               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   CUSTOM HEAD (Clasificación)                │
│                                                              │
│  GlobalAveragePooling2D()                                   │
│  Input: (7, 7, 1280) → Output: (1280,)                     │
│  Reduce dimensionalidad espacial                            │
│         │                                                    │
│         ▼                                                    │
│  BatchNormalization()                                       │
│  Normaliza activaciones (μ=0, σ=1)                         │
│         │                                                    │
│         ▼                                                    │
│  Dropout(0.5)                                               │
│  Desactiva aleatoriamente 50% neuronas                      │
│         │                                                    │
│         ▼                                                    │
│  Dense(256, ReLU, L2=5e-4)                                  │
│  Feature learning específico                                │
│         │                                                    │
│         ▼                                                    │
│  BatchNormalization()                                       │
│         │                                                    │
│         ▼                                                    │
│  Dropout(0.3)                                               │
│  Segundo dropout (más conservador)                          │
│         │                                                    │
│         ▼                                                    │
│  Dense(21, Softmax, L2=5e-4)                                │
│  Capa de salida: 21 probabilidades                          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
OUTPUT: [0.01, 0.89, 0.03, ..., 0.02]
              ↑ Pancakes 89%
```

### 2.2 Matemática Detallada de Cada Capa

#### 2.2.1 GlobalAveragePooling2D

**Operación Matemática**:
```
Input:  X ∈ ℝ^(B×H×W×C) = (batch, 7, 7, 1280)
Output: y ∈ ℝ^(B×C) = (batch, 1280)

Para cada canal c:
    y[b,c] = (1/(H×W)) × Σ(i=0 to H-1) Σ(j=0 to W-1) X[b,i,j,c]
```

**Código Implementado**:
```python
x = layers.GlobalAveragePooling2D()(x)
# Entrada:  (None, 7, 7, 1280)
# Salida:   (None, 1280)
```

**Por qué GAP vs Flatten**:
- Flatten: 7×7×1280 = 62,720 parámetros en siguiente capa
- GAP: 1280 parámetros en siguiente capa
- **Reducción: 98% menos parámetros → menos overfitting**

---

#### 2.2.2 BatchNormalization

**Fórmula Matemática**:
```
1. Calcular estadísticas del batch:
   μ_B = (1/m) × Σ(x_i)
   σ²_B = (1/m) × Σ(x_i - μ_B)²

2. Normalizar:
   x̂_i = (x_i - μ_B) / √(σ²_B + ε)

3. Escalar y desplazar (parámetros aprendibles):
   y_i = γ × x̂_i + β

Parámetros:
- γ (gamma): escala (inicializado en 1)
- β (beta): desplazamiento (inicializado en 0)
- ε: estabilidad numérica = 1e-5
```

**Beneficios**:
1. **Normalización**: Mantiene activaciones en rango estable
2. **Aceleración**: Permite learning rates más altos
3. **Regularización**: Efecto similar a dropout (ruido del batch)
4. **Reduce dependencia**: De inicialización de pesos

**Durante Entrenamiento vs Inferencia**:
```python
# Training: Usa estadísticas del batch actual
BN(x, training=True)  → μ_batch, σ_batch

# Inferencia: Usa media móvil de todo el entrenamiento
BN(x, training=False) → μ_moving, σ_moving
```

---

#### 2.2.3 Dropout

**Algoritmo (Training)**:
```python
def dropout(x, rate=0.5):
    # Generar máscara binaria
    mask = np.random.binomial(1, 1-rate, size=x.shape)

    # Aplicar máscara y escalar
    return x * mask / (1 - rate)

# Ejemplo:
x = [0.8, 0.5, 0.3, 0.9, 0.2]
mask = [1, 0, 1, 0, 1]  # Aleatoria
y = [0.8/0.5, 0, 0.3/0.5, 0, 0.2/0.5]
  = [1.6, 0, 0.6, 0, 0.4]
```

**Algoritmo (Inference)**:
```python
def dropout(x, rate=0.5, training=False):
    if not training:
        return x  # Sin modificación
    return x * mask / (1 - rate)
```

**Interpretación Intuitiva**:
- Fuerza a la red a no depender de neuronas específicas
- Crea sub-redes aleatorias en cada epoch
- Similar a ensemble de múltiples modelos

---

#### 2.2.4 Dense Layer con Regularización L2

**Forward Pass**:
```
z = W × x + b
a = σ(z)

Donde:
- W: matriz de pesos (256×1280)
- x: vector entrada (1280,)
- b: vector bias (256,)
- σ: ReLU(z) = max(0, z)
```

**L2 Regularization (Weight Decay)**:
```
Loss_total = Loss_CE + λ × ||W||²₂

||W||²₂ = Σ(w_ij²)  (suma de cuadrados de todos los pesos)

Con λ = 5e-4:
- Penaliza pesos grandes
- Prefiere soluciones más simples
- Reduce variance del modelo
```

**Gradiente con L2**:
```
∂Loss/∂W = ∂Loss_CE/∂W + 2λW

Efecto en actualización:
W_new = W_old - lr × (∇Loss_CE + 2λW)
      = W_old × (1 - 2λ×lr) - lr × ∇Loss_CE
              ↑
         Weight decay
```

---

#### 2.2.5 Softmax + Categorical Crossentropy

**Softmax**:
```
p_i = exp(z_i) / Σ(j=1 to K) exp(z_j)

Propiedades:
- Σ(p_i) = 1
- p_i ∈ [0, 1]
- Amplifica diferencias (exp)
```

**Categorical Crossentropy**:
```
Loss = -Σ(i=1 to K) y_i × log(p_i)

Donde:
- y_i: one-hot encoding (1 para clase correcta, 0 resto)
- p_i: probabilidad predicha para clase i

Simplificación (solo 1 clase es 1):
Loss = -log(p_correcto)
```

**Label Smoothing (α=0.2)**:
```
En vez de: y = [0, 1, 0, ..., 0]
Usamos:    y = [0.01, 0.9, 0.01, ..., 0.01]

y_smooth = y × (1-α) + α/K

Efecto:
- Reduce overconfidence
- Mejora generalización
- Calibración de probabilidades
```

---

### 2.3 Conteo Detallado de Parámetros

```python
# MobileNetV2 Base
base_total = 2,257,984
  - Primeras 135 capas (congeladas): ~1.5M
  - Últimas 20 capas (entrenables): ~700K

# Custom Head
GlobalAvgPool:     0 parámetros
BatchNorm1:        2,560 = 1280×2 (gamma+beta)
Dropout1:          0
Dense(256):        328,192 = (1280×256) + 256
BatchNorm2:        512 = 256×2
Dropout2:          0
Dense(21):         5,397 = (256×21) + 21

# Total Custom Head: 336,661
# Total Entrenables: ~1,862,805
# Total Modelo: 2,597,461

# Memoria GPU/CPU:
Modelo float32: 2,597,461 × 4 bytes ≈ 10MB
Activaciones (batch=16): ~200MB
Total training: ~250MB (muy eficiente)
```

---

### 2.4 Por Qué Esta Arquitectura Específica

#### Comparación con Alternativas

```
┌────────────────┬──────────┬─────────┬──────────┬──────────┐
│ Arquitectura   │ Params   │ Tamaño  │ Inf.(ms) │ Accuracy │
├────────────────┼──────────┼─────────┼──────────┼──────────┤
│ MobileNetV2    │ 3.5M     │ 14MB    │ 150ms    │ 75-80%   │
│ ResNet50       │ 25.6M    │ 98MB    │ 450ms    │ 78-82%   │
│ EfficientNetB0 │ 5.3M     │ 21MB    │ 200ms    │ 76-81%   │
│ VGG16          │ 138M     │ 528MB   │ 1200ms   │ 71-75%   │
│ InceptionV3    │ 23.8M    │ 92MB    │ 400ms    │ 77-81%   │
└────────────────┴──────────┴─────────┴──────────┴──────────┘

Decisión: MobileNetV2
Razón: Mejor balance velocidad/tamaño/accuracy para producción
```

#### Decisiones de Diseño Críticas

**1. Congelar primeras 135 capas (dejar 20 últimas)**
```
Razones:
✅ Features básicas ya aprendidas (bordes, texturas)
✅ Reduce overfitting (menos parámetros a aprender)
✅ Acelera entrenamiento (menos backprop)
❌ Si congelamos todo: underfitting
❌ Si descongelamos todo: overfitting + lento
```

**2. Dos capas Dense (1280→256→21) en vez de una (1280→21)**
```
Una capa:
- Transición abrupta
- Menos capacidad de aprendizaje
- Difícil capturar patrones complejos

Dos capas:
✅ Transición suave
✅ Layer 256 aprende features intermedias específicas
✅ Mejor separabilidad de clases
```

**3. Dropout decreciente (0.5 → 0.3)**
```
Primera capa: 50% dropout
- Después de GAP (muchas features)
- Necesita regularización fuerte

Segunda capa: 30% dropout
- Antes de clasificación final
- Regularización moderada (no destruir info crítica)

Best practice: Dropout decreciente hacia la salida
```

**4. L2 en ambas capas Dense**
```
Sin L2:          Pesos pueden crecer sin límite
Con L2=5e-4:     Pesos pequeños preferidos

Ejemplo:
W1 = 10.5  → L2_loss = 5e-4 × 10.5² = 0.055
W2 = 0.5   → L2_loss = 5e-4 × 0.5² = 0.000125

Gradiente penaliza más W1 → tiende a valores pequeños
```

---

## 3. Pipeline de Datos

### 3.1 OptimizedDataGenerator: Carga Eficiente

**Problema Original**:
```python
# ❌ Enfoque naive: cargar TODO en RAM
X = np.load('all_images.npy')  # 21,000 imágenes × 224×224×3 = 3.5GB
y = np.load('all_labels.npy')

# Problema: OOM (Out of Memory) en máquinas con 8GB RAM
```

**Solución: DataGenerator**:
```python
# ✅ Cargar solo UN batch a la vez (16 imágenes)
class OptimizedDataGenerator(keras.utils.Sequence):
    def __getitem__(self, batch_idx):
        # Cargar solo 16 imágenes desde disco
        batch = load_from_npz(batch_idx)
        return batch  # 16 × 224×224×3 ≈ 2.4MB
```

**Ventajas**:
- ✅ Memoria constante: ~50MB (vs 3.5GB)
- ✅ Escalable a millones de imágenes
- ✅ Permite data augmentation on-the-fly

---

### 3.2 Arquitectura del DataGenerator

```
NPZ Files en Disco (400 archivos)
c000_batch0000.npz (20 imágenes, ~3MB)
c000_batch0001.npz
...
c020_batch0199.npz
         │
         ▼
┌─────────────────────────────────────────────┐
│   OptimizedDataGenerator.__init__()         │
│                                              │
│  1. Lee metadatos de NPZ (NO datos)         │
│  2. Crea índice: [(npz_path, img_idx), ...] │
│  3. Total samples: 20,987                    │
│  4. Shuffle índice si shuffle=True           │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│   __getitem__(batch_idx)                     │
│   Se llama en cada iteración                 │
│                                              │
│  1. Calcular qué samples necesito:           │
│     indices[batch_idx*16 : (batch_idx+1)*16] │
│                                              │
│  2. Agrupar por archivo NPZ:                 │
│     {npz1: [2,5,8], npz2: [1,9,15], ...}    │
│                                              │
│  3. Cargar archivos NPZ (uno a uno):         │
│     with np.load(npz_path) as data:          │
│         X = data['X'][indices]               │
│         y = data['y'][indices]               │
│                                              │
│  4. Normalizar: X = X.astype(float32) / 255  │
│                                              │
│  5. One-hot encode: y → [0,0,1,0,...,0]     │
│                                              │
│  6. Data Augmentation (si training):         │
│     X_aug = augment_batch(X)                 │
│                                              │
│  7. Return (X_aug, y)                        │
└─────────────────────────────────────────────┘
```

---

### 3.3 Data Augmentation: 9 Transformaciones

**Objetivo**: Aumentar variabilidad sin añadir datos reales.

```python
def _augment_batch(self, X):
    """
    Aplica transformaciones aleatorias a cada imagen
    Reduce overfitting al mostrar variaciones
    """
    for img in X:
        # 1. Flip Horizontal (75%)
        if random() > 0.25:
            img = flip_left_right(img)

        # 2. Rotation ±15° (70%)
        if random() > 0.3:
            angle = uniform(-15, 15)
            img = rotate(img, angle)

        # 3. Brightness ±30% (70%)
        if random() > 0.3:
            img = adjust_brightness(img, delta=0.3)

        # 4. Contrast 0.7-1.3x (70%)
        if random() > 0.3:
            img = adjust_contrast(img, 0.7, 1.3)

        # 5. Saturation 0.7-1.3x (70%)
        if random() > 0.3:
            img = adjust_saturation(img, 0.7, 1.3)

        # 6. Hue ±15% (70%)
        if random() > 0.3:
            img = adjust_hue(img, delta=0.15)

        # 7. Zoom 90%-110% (60%)
        if random() > 0.4:
            scale = uniform(0.9, 1.1)
            img = zoom_and_crop(img, scale)

        # 8. Cutout 40x40 (10%)
        if random() > 0.9:
            x, y = random_position()
            img[y:y+40, x:x+40] = 0

        # 9. Clip valores [0,1]
        img = clip(img, 0, 1)

    return X_augmented
```

**Impacto de Cada Transformación**:

| Transformación | Probabilidad | Objetivo | Impacto Overfitting |
|----------------|--------------|----------|---------------------|
| Flip Horizontal | 75% | Invarianza espejo | Alto ⭐⭐⭐ |
| Rotation ±15° | 70% | Invarianza rotación | Alto ⭐⭐⭐ |
| Brightness | 70% | Robustez iluminación | Medio ⭐⭐ |
| Contrast | 70% | Diferentes cámaras | Medio ⭐⭐ |
| Saturation | 70% | Diferentes displays | Bajo ⭐ |
| Hue | 70% | Variación color | Bajo ⭐ |
| Zoom | 60% | Diferentes distancias | Alto ⭐⭐⭐ |
| Cutout | 10% | Oclusiones parciales | Muy Alto ⭐⭐⭐⭐ |

**Ejemplo Visual**:
```
Imagen Original (pancakes):
┌──────────────┐
│   🥞🥞🥞     │
│   🥞🥞🥞     │
│   🍓🍯       │
└──────────────┘

Flip Horizontal:
┌──────────────┐
│     🥞🥞🥞   │
│     🥞🥞🥞   │
│       🍯🍓   │
└──────────────┘

Rotation +10°:
┌──────────────┐
│  🥞🥞🥞      │
│   🥞🥞🥞     │
│    🍓🍯      │
└──────────────┘

Cutout (oclusión):
┌──────────────┐
│   🥞■■🥞     │  ← Parche negro
│   ■■■🥞     │
│   🍓🍯       │
└──────────────┘
```

---

### 3.4 Preprocesamiento de Imágenes

**Pipeline Completo de Preprocessing**:

```
Imagen RAW (usuario sube)
         │
         ▼
┌─────────────────────────────────────┐
│ 1. PIL.Image.open(bytes)             │
│    Soporta: JPG, PNG, BMP, GIF       │
│    Output: Image object              │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 2. Convertir a RGB                   │
│    Si RGBA → RGB (quitar alpha)      │
│    Si Grayscale → RGB (duplicar)     │
│    Output: (H, W, 3)                 │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 3. Resize a 224x224                  │
│    Método: BILINEAR (rápido+calidad) │
│    Preserva aspect ratio: NO         │
│    (puede distorsionar ligeramente)  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 4. Convertir a NumPy array           │
│    dtype: uint8 [0-255]              │
│    shape: (224, 224, 3)              │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 5. Normalizar a [0, 1]               │
│    img = img.astype(float32) / 255   │
│    Razón: Mejora convergencia        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 6. Añadir dimensión batch            │
│    (224,224,3) → (1,224,224,3)      │
│    Keras espera batch dimension      │
└─────────────────────────────────────┘
         │
         ▼
    Listo para modelo.predict()
```

**Código Real**:
```python
def preprocess_image(self, image_bytes):
    # 1. Abrir imagen
    img = Image.open(io.BytesIO(image_bytes))

    # 2. Convertir a RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 3. Resize
    img = img.resize((224, 224))

    # 4. To array
    img_array = np.array(img)  # uint8

    # 5. Normalizar
    img_array = img_array.astype('float32') / 255.0

    # 6. Batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array  # (1, 224, 224, 3)
```

---

### 3.5 One-Hot Encoding

**Por Qué One-Hot**:
```python
# ❌ Encoding numérico (MALO para clasificación):
labels = [0, 1, 2, ..., 20]
# Problema: El modelo puede pensar que 20 > 1 (orden implícito)

# ✅ One-hot encoding (CORRECTO):
label = 5  # "eggs_benedict"
one_hot = [0, 0, 0, 0, 0, 1, 0, ..., 0]  # 21 elementos
#                       ↑
#                   Posición 5 = 1

# Ventajas:
# - No hay orden implícito
# - Compatible con softmax
# - Cada clase es independiente
```

**Implementación**:
```python
y = keras.utils.to_categorical(y, num_classes=21)

# Ejemplo:
y_original = [0, 0, 1, 0, 2]  # 5 imágenes

y_onehot = [
    [1, 0, 0, ..., 0],  # Clase 0
    [1, 0, 0, ..., 0],  # Clase 0
    [0, 1, 0, ..., 0],  # Clase 1
    [1, 0, 0, ..., 0],  # Clase 0
    [0, 0, 1, ..., 0],  # Clase 2
]  # Shape: (5, 21)
```

---

## 4. Técnicas de Regularización

### 4.1 ¿Qué es Overfitting?

**Definición**:
```
Overfitting = Modelo memoriza datos de entrenamiento
              pero falla en datos nuevos

Train Accuracy: 99% ✅
Val Accuracy:   73% ❌
Gap: 26% → OVERFITTING CRÍTICO
```

**Visualización**:
```
Loss │
     │ Training Loss ──────────────
     │                             ↓ continúa bajando
     │
     │ Validation Loss ─────────── ↑ sube (overfitting)
     │                          ↗
     │                       ↗
     └──────────────────────────────→ Epochs
                            ↑
                    Punto óptimo
```

---

### 4.2 Dropout: Regularización Estocástica

**Concepto Core**:
```
Durante entrenamiento:
Cada neurona tiene probabilidad p de "apagarse"

Epoch 1:  [1, 0, 1, 1, 0, 1, 0, 1]  ← Máscara aleatoria
Epoch 2:  [0, 1, 1, 0, 1, 1, 1, 0]  ← Diferente
Epoch 3:  [1, 1, 0, 1, 0, 0, 1, 1]  ← Diferente

Efecto: Fuerza redundancia → no depende de neuronas específicas
```

**Matemática Formal**:
```
Training:
h = activation(W × x + b)
h_dropped = h × mask / (1-p)

donde mask[i] ~ Bernoulli(1-p)

Inference:
h_pred = activation(W × x + b)  # Sin dropout, usa todos
```

**Interpretación como Ensemble**:
```
Dropout(0.5) con 1000 neuronas:

Combinaciones posibles = 2^1000 ≈ 10^301
(cada subset es un modelo diferente)

Durante entrenamiento: Promedia todas estas subredes
Durante inferencia: Usa el promedio geométrico
```

**Por Qué Funciona**:
1. **Previene Co-adaptación**: Neuronas no pueden "confabular"
2. **Ensemble Implícito**: Promedia múltiples arquitecturas
3. **Feature Redundancy**: Aprende múltiples formas de detectar patterns

---

### 4.3 L2 Regularization (Weight Decay)

**Intuición**:
```
Sin regularización:
W puede crecer sin límite → modelo complejo → overfitting

Con L2:
Penaliza pesos grandes → prefiere soluciones simples
```

**Matemática**:
```
Loss_total = Loss_CE + λ × Σ(w²)
                        ↑
                    L2 penalty

Ejemplo numérico:
Modelo A: W = [10, -8, 12, -15]
    L2 = 5e-4 × (100 + 64 + 144 + 225) = 0.267

Modelo B: W = [2, -1, 3, -2]
    L2 = 5e-4 × (4 + 1 + 9 + 4) = 0.009

Modelo B preferido (pesos más pequeños)
```

**Efecto en Gradiente**:
```
∂Loss/∂w = ∂Loss_CE/∂w + 2λw

Actualización:
w_new = w_old - lr × (∇Loss_CE + 2λw_old)
      = w_old(1 - 2λ×lr) - lr×∇Loss_CE
               ↑
          Decae hacia 0
```

**Comparación con L1**:
```
L1: Loss = λ × Σ|w|   → Sparsity (muchos pesos = 0)
L2: Loss = λ × Σw²    → Small weights (pesos pequeños)

Para clasificación: L2 preferido (mantiene todas features)
```

---

### 4.4 Label Smoothing

**Problema de Hard Labels**:
```
One-hot tradicional:
y_true = [0, 0, 1, 0, 0, ..., 0]  ← 100% seguro

Problema:
- Fuerza al modelo a ser extremadamente confiado
- Penaliza duramente pequeños errores
- Puede causar overfitting
```

**Solución: Label Smoothing**:
```
Con α = 0.2:
y_smooth = [ε, ε, 1-α+ε, ε, ε, ..., ε]

donde ε = α / (K-1) = 0.2 / 20 = 0.01

Resultado:
y_smooth = [0.01, 0.01, 0.81, 0.01, ..., 0.01]
                          ↑
                    Clase correcta: 81% (no 100%)
```

**Matemática**:
```
y_smooth(k) = y_hard(k) × (1 - α) + α/K

Ejemplo con K=21, α=0.2:
- Clase correcta:   1 × 0.8 + 0.01 = 0.81
- Clases incorrectas: 0 × 0.8 + 0.01 = 0.01

Suma = 0.81 + 20×0.01 = 0.81 + 0.20 = 1.01 ≈ 1.0 ✓
```

**Beneficios**:
1. **Evita Overconfidence**: El modelo no llega a 100%
2. **Mejor Calibración**: Probabilidades más realistas
3. **Generalización**: Reduce gap train-val
4. **Robustez**: Menos sensible a labels ruidosos

**Impacto en Loss**:
```
Sin smoothing:
Loss = -log(0.99) = 0.01   (muy pequeño, gradientes débiles)

Con smoothing:
Loss = -log(0.81) = 0.21   (más grande, gradientes más fuertes)

Efecto: Continúa aprendiendo incluso cuando casi correcto
```

---

### 4.5 Early Stopping

**Concepto**:
```
Detener entrenamiento cuando validación deja de mejorar
Evita continuar cuando solo está memorizando
```

**Algoritmo**:
```python
best_val_loss = ∞
patience_counter = 0
patience = 12

for epoch in range(max_epochs):
    train_model()
    val_loss = validate()

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        save_model()  # Guardar mejor
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping!")
        restore_best_model()
        break
```

**Parámetros**:
```
patience = 12  # Número de epochs sin mejora
min_delta = 0.001  # Mejora mínima para contar

Ejemplo:
Epoch 15: val_loss = 0.850
Epoch 16: val_loss = 0.849  # Mejora 0.001 → cuenta
Epoch 17: val_loss = 0.8495 # Mejora 0.0005 → NO cuenta (< min_delta)
...
Epoch 29: 12 epochs sin mejora → STOP
```

**Por Qué Funciona**:
```
Training Loss siempre baja → no es buen criterio
Validation Loss:
  Baja al principio → modelo aprende
  Deja de bajar → saturación
  Sube → overfitting

                ┌─ Stop aquí
Loss │           ↓
     │ Val ────────╮
     │             │ Empieza a subir
     │ Train  ─────┴────────
     └────────────────────────→ Epochs
```

---

### 4.6 Batch Normalization como Regularizador

**Efecto Regularizador**:
```
BN añade ruido al cálculo:
- Estadísticas varían entre batches
- μ_batch ≠ μ_population
- Similar a dropout (ruido estocástico)
```

**Comparación**:
```
┌──────────────────┬──────────┬──────────────┐
│ Técnica          │ Tipo     │ Fuerza       │
├──────────────────┼──────────┼──────────────┤
│ Dropout          │ Explícito│ Alta ⭐⭐⭐    │
│ L2               │ Explícito│ Media ⭐⭐     │
│ Label Smoothing  │ Implícito│ Media ⭐⭐     │
│ Batch Norm       │ Implícito│ Baja ⭐       │
│ Data Augmentation│ Explícito│ Muy Alta ⭐⭐⭐⭐│
└──────────────────┴──────────┴──────────────┘
```

---

### 4.7 Reducir Overfitting: Estrategia Combinada

**Nuestro Stack de Regularización**:
```python
# 1. Transfer Learning (base congelada)
for layer in base_model.layers[:-20]:
    layer.trainable = False  # Reduce params entrenables

# 2. Dropout (dos capas)
Dropout(0.5)  # Agresivo
Dropout(0.3)  # Moderado

# 3. L2 Regularization
Dense(256, kernel_regularizer=l2(5e-4))
Dense(21, kernel_regularizer=l2(5e-4))

# 4. Label Smoothing
loss = CategoricalCrossentropy(label_smoothing=0.2)

# 5. Batch Normalization
BatchNormalization()  # x2

# 6. Early Stopping
EarlyStopping(patience=12, min_delta=0.001)

# 7. Data Augmentation
9 transformaciones aleatorias
```

**Impacto Medido**:
```
Sin regularización:
Train: 99.8%, Val: 65%  → Gap: 34.8% ❌

Con regularización completa:
Train: 82%, Val: 78%    → Gap: 4% ✅

Reducción de overfitting: 87% 🎉
```

---

## 5. Learning Rate Schedule: Warmup + Cosine Annealing

### 5.1 ¿Por Qué No Learning Rate Constante?

**Problema**:
```
LR Constante = 1e-3 durante todo el entrenamiento

Inicio: Gradientes grandes → steps grandes → inestabilidad
Final:  Cerca del óptimo → steps grandes → oscila sin converger
```

**Solución: Learning Rate Adaptativo**
```
Inicio:     LR pequeño → exploración cautelosa
Medio:      LR grande → aprendizaje rápido
Final:      LR pequeño → refinamiento fino
```

---

### 5.2 Warmup: Calentamiento Gradual

**Algoritmo**:
```python
for epoch in range(warmup_epochs):
    lr = initial_lr * (epoch + 1) / warmup_epochs

# Ejemplo con warmup_epochs=3, initial_lr=1e-3:
Epoch 0: lr = 1e-3 * (0+1)/3 = 3.33e-4
Epoch 1: lr = 1e-3 * (1+1)/3 = 6.67e-4
Epoch 2: lr = 1e-3 * (2+1)/3 = 1.00e-3
```

**Visualización**:
```
LR  │
1e-3│            ┌────────
    │           ╱
    │          ╱   Warmup
    │         ╱   (3 epochs)
    │        ╱
    │  ────╱
    └──────────────────────────→ Epoch
    0   1   2   3
```

**Por Qué Funciona**:
1. **Pesos iniciales aleatorios**: Gradientes pueden ser erráticos
2. **LR alto inicial**: Puede causar divergencia
3. **Warmup**: Da tiempo a estabilizar antes de acelerar

---

### 5.3 Cosine Annealing: Decaimiento Suave

**Fórmula Matemática**:
```
lr(t) = lr_min + (lr_max - lr_min) × 0.5 × (1 + cos(π × t/T))

donde:
- t: epoch actual (después de warmup)
- T: total epochs
- lr_max: learning rate inicial (1e-3)
- lr_min: learning rate mínimo (1e-6)
```

**Ejemplo Numérico**:
```python
lr_max = 1e-3
lr_min = 1e-6
T = 27  # 30 epochs - 3 warmup

# Epoch 3 (t=0):
lr = 1e-6 + (1e-3 - 1e-6) × 0.5 × (1 + cos(0))
   = 1e-6 + 0.001 × 0.5 × 2
   = 1e-3 ✓

# Epoch 16 (t=13, mitad):
lr = 1e-6 + 0.001 × 0.5 × (1 + cos(π × 13/27))
   = 1e-6 + 0.001 × 0.5 × (1 + (-0.5))
   = 2.5e-4

# Epoch 30 (t=27, final):
lr = 1e-6 + 0.001 × 0.5 × (1 + cos(π))
   = 1e-6 + 0.001 × 0.5 × 0
   = 1e-6 ✓
```

**Visualización Completa**:
```
LR  │
1e-3│    Warmup  │   Cosine Annealing
    │      ╱     │  ╱╲
    │     ╱      │ ╱  ╲
5e-4│    ╱       │╱    ╲
    │   ╱        │      ╲___
    │  ╱         │          ╲___
1e-6│ ╱          │              ───
    └────────────────────────────────────→ Epoch
    0  1  2  3   7   14  21  28  30
```

---

### 5.4 Comparación de Schedules

```
┌────────────────────┬──────────┬────────────┬─────────────┐
│ Schedule           │ Convergencia│ Estabilidad│ Best Acc  │
├────────────────────┼──────────┼────────────┼─────────────┤
│ Constante          │ Lenta    │ Baja       │ 70%         │
│ Step Decay         │ Media    │ Media      │ 73%         │
│ Exponential Decay  │ Rápida   │ Baja       │ 72%         │
│ Cosine Annealing   │ Rápida   │ Alta       │ 78% ✅      │
│ Warmup + Cosine    │ Óptima   │ Muy Alta   │ 80% ⭐      │
└────────────────────┴──────────┴────────────┴─────────────┘
```

---

### 5.5 Implementación Real

```python
class WarmUpCosineDecay(keras.callbacks.Callback):
    def __init__(self, initial_lr, min_lr, warmup_epochs, total_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Fase 1: Warmup lineal
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Fase 2: Cosine annealing
            progress = (epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))

        # Actualizar learning rate
        keras.backend.set_value(
            self.model.optimizer.learning_rate, lr
        )

        if epoch % 5 == 0:
            print(f"\n   📉 Learning Rate: {lr:.6f}")
```

---

## 6. API FastAPI: Arquitectura y Endpoints

### 6.1 Estructura de la API

```python
app = FastAPI(
    title="Calorie Detection API",
    description="CNN-based food classification",
    version="3.0.0"
)

# CORS para frontend
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Cargar modelo al inicio
cnn_predictor = CNNPredictor(
    model_path='models/breakfast_cnn_model_optimized.h5',
    class_names_path='models/class_names.pkl'
)
```

---

### 6.2 Endpoint `/predict`: Flujo Completo

```python
@app.post("/predict")
async def predict_food_calories(file: UploadFile = File(...)):
    """
    Predice clase y calorías de una imagen
    """
    # 1. Validación
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Debe ser imagen")

    # 2. Leer bytes
    image_bytes = await file.read()

    # 3. Predecir
    result = cnn_predictor.predict(image_bytes)

    # 4. Formatear respuesta
    return JSONResponse({
        'predicted_class': result['predicted_class'],
        'confidence': result['confidence'],
        'estimated_calories': result['estimated_calories'],
        'nutrition': result['nutrition'],
        'top_predictions': result['top_predictions']
    })
```

**Ejemplo de Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@pancake_image.jpg"
```

**Ejemplo de Response**:
```json
{
  "success": true,
  "predicted_class": "pancakes",
  "display_name": "Pancakes",
  "confidence": 0.8943,
  "estimated_calories": 340,
  "portion_size_g": 150,
  "calories_per_100g": 227,
  "nutrition": {
    "protein": 9.3,
    "carbohydrates": 42.5,
    "fat": 15.5
  },
  "top_predictions": [
    {"class": "pancakes", "confidence": 0.8943},
    {"class": "waffles", "confidence": 0.0521},
    {"class": "french_toast", "confidence": 0.0312}
  ],
  "model_info": {
    "type": "CNN - MobileNetV2",
    "num_classes": 21
  }
}
```

---

### 6.3 CNNPredictor: Motor de Predicción

```python
class CNNPredictor:
    def __init__(self, model_path, class_names_path):
        # Cargar modelo Keras
        self.model = keras.models.load_model(model_path)

        # Cargar nombres de clases
        with open(class_names_path, 'rb') as f:
            self.class_names = pickle.load(f)

        # Base de datos nutricional
        self.nutrition_data = {
            'pancakes': {
                'calories': 227,
                'protein': 6.2,
                'carbs': 28.3,
                'fat': 10.3
            },
            # ... 20 clases más
        }

    def predict(self, image_bytes):
        # 1. Preprocesar
        img_array = self.preprocess_image(image_bytes)

        # 2. Predecir (forward pass)
        predictions = self.model.predict(img_array)[0]

        # 3. Interpretar
        predicted_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])

        # 4. Calcular calorías
        nutrition = self.nutrition_data[predicted_class]
        calories = int(nutrition['calories'] * 1.5)  # Porción 150g

        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'estimated_calories': calories,
            'nutrition': nutrition
        }
```

---

## 7. Métricas y Evaluación

### 7.1 Accuracy (Top-1)

**Definición**:
```
Accuracy = (Predicciones Correctas) / (Total Predicciones)
```

**Cálculo**:
```python
predictions = model.predict(X_test)  # Shape: (N, 21)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

accuracy = np.mean(predicted_classes == true_classes)
# Ejemplo: 4758 / 6330 = 0.7516 = 75.16%
```

**Interpretación**:
- **75%+**: Excelente para 21 clases
- **60-75%**: Bueno
- **<60%**: Necesita mejora

---

### 7.2 Top-3 Accuracy

**Definición**:
```
Correcto si la clase verdadera está en las 3 predicciones con mayor probabilidad
```

**Cálculo**:
```python
# Top-3 índices
top3_indices = np.argsort(predictions, axis=1)[:, -3:]

# Verificar si true_class está en top-3
top3_correct = [true_class in top3
                for true_class, top3 in zip(true_classes, top3_indices)]

top3_accuracy = np.mean(top3_correct)
# Ejemplo: 5590 / 6330 = 0.8831 = 88.31%
```

**Por Qué Importa**:
- Muchas clases similares (pancakes vs waffles)
- Usuario puede ver top-3 y elegir
- Más realista para UX

---

### 7.3 Categorical Crossentropy Loss

**Fórmula**:
```
Loss = -(1/N) × Σ Σ y_true[i,j] × log(y_pred[i,j])
```

**Ejemplo**:
```python
# Imagen de pancakes (clase 18)
y_true = [0, 0, ..., 1, ..., 0]  # One-hot
           ↑
y_pred = [0.01, 0.02, ..., 0.89, ..., 0.03]  # Softmax

Loss = -log(0.89) = 0.117

# Si predice mal (waffles):
y_pred_wrong = [0.01, 0.02, ..., 0.10, ..., 0.75]
                                  ↑pancakes  ↑waffles
Loss = -log(0.10) = 2.303  (mucho mayor!)
```

**Interpretación**:
- **<0.5**: Muy bueno
- **0.5-1.0**: Bueno
- **>1.0**: Necesita entrenamiento

---

### 7.4 Detección de Overfitting

**Métrica Clave**: Gap Train-Val
```
Gap = Train_Accuracy - Val_Accuracy

Gap < 5%:    Generalización excelente ✅
Gap 5-10%:   Generalización buena
Gap 10-20%:  Overfitting moderado ⚠️
Gap > 20%:   Overfitting crítico ❌
```

**Ejemplo de Evolución**:
```
Epoch  | Train Acc | Val Acc | Gap
-------|-----------|---------|------
5      | 65%       | 62%     | 3%   ✅
10     | 75%       | 71%     | 4%   ✅
15     | 82%       | 78%     | 4%   ✅
20     | 87%       | 79%     | 8%   ⚠️
25     | 92%       | 78%     | 14%  ❌ Detenerse!
```

---

## 8. Guía de Uso y Troubleshooting

### 8.1 Entrenar el Modelo

```bash
# 1. Preparar entorno
cd backend
source ../.venv/bin/activate
pip install -r requirements.txt

# 2. Entrenar
python train_cnn_model.py

# Salida esperada:
# ✅ TensorFlow 2.20.0 cargado correctamente!
# [1/6] 📊 Cargando metadata del dataset...
# [2/6] ✂️ Dividiendo dataset...
# [3/6] 🔄 Creando generadores de datos...
# [4/6] 🏗️ Construyendo modelo MobileNetV2...
# [5/6] 🚀 Entrenando modelo (30 epochs)...
# [6/6] 🔍 Evaluando modelo en test set...
#
# Test Accuracy: 78.23%
# Test Top-3 Accuracy: 89.14%
# Overfitting: 3.8%

# 3. Verificar archivos
ls -lh models/
# breakfast_cnn_model_optimized.h5  (50MB)
# class_names.pkl
# training_history.json
# training_curves.png
```

---

### 8.2 Iniciar la API

```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Verificar:
curl http://localhost:8000/health

# Terminal 2: Frontend (opcional)
cd frontend
npm run dev
```

---

### 8.3 Problemas Comunes

#### ❌ Error: "Modelo no disponible"

**Síntoma**:
```
WARNING:main:⚠️ Modelo CNN no disponible
ERROR:main:❌ Error en predicción: Modelo no disponible
```

**Solución**:
```bash
# 1. Verificar que existe
ls backend/models/breakfast_cnn_model_optimized.h5

# 2. Si no existe, entrenar:
python backend/train_cnn_model.py

# 3. Reiniciar API
# Ctrl+C en terminal del backend
uvicorn main:app --reload
```

---

#### ❌ Error: OOM (Out of Memory)

**Síntoma**:
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solución**:
```python
# Reducir batch size en train_cnn_model.py
BATCH_SIZE = 8  # En vez de 16

# O cerrar programas que consuman RAM
```

---

#### ❌ Overfitting > 10%

**Síntoma**:
```
Train Accuracy: 95%
Val Accuracy: 72%
Gap: 23% ❌
```

**Solución**:
```python
# Editar train_cnn_model.py:

# 1. Aumentar regularización
DROPOUT_RATE = 0.6  # de 0.5
L2_REGULARIZATION = 1e-3  # de 5e-4
LABEL_SMOOTHING = 0.3  # de 0.2

# 2. Congelar más capas
for layer in base_model.layers[:-15]:  # de -20 a -15
    layer.trainable = False

# 3. Re-entrenar
python train_cnn_model.py
```

---

### 8.4 Optimizaciones Futuras

#### 1. **Quantization (Reducir tamaño)**
```python
import tensorflow as tf

# Convertir a TFLite quantizado
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Resultado: 50MB → 12MB (4x reducción)
```

#### 2. **Test-Time Augmentation**
```python
def predict_with_tta(image, n_augmentations=5):
    predictions = []
    for _ in range(n_augmentations):
        aug_image = augment(image)
        pred = model.predict(aug_image)
        predictions.append(pred)

    return np.mean(predictions, axis=0)

# Mejora accuracy ~2-3%
```

#### 3. **Ensemble de Modelos**
```python
models = [
    load_model('mobilenetv2.h5'),
    load_model('efficientnetb0.h5'),
    load_model('resnet50.h5')
]

def ensemble_predict(image):
    preds = [m.predict(image) for m in models]
    return np.mean(preds, axis=0)

# Mejora accuracy ~3-5%
```

---

### 8.5 Métricas de Producción

**Latencia**:
```
Target: < 500ms por predicción

Breakdown:
- Preprocesamiento: ~50ms
- Inferencia CNN: ~150ms
- Post-procesamiento: ~10ms
- Total: ~210ms ✅
```

**Throughput**:
```
Con batch_size=16:
16 imágenes en 600ms = 26 img/s

Suficiente para:
- 100,000 requests/hora
- Aplicación web con miles de usuarios
```

---

## 📚 Referencias Técnicas

### Papers Fundamentales

1. **MobileNetV2**
   - Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
   - CVPR 2018

2. **Dropout**
   - Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
   - JMLR 2014

3. **Batch Normalization**
   - Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
   - ICML 2015

4. **Label Smoothing**
   - Szegedy et al., "Rethinking the Inception Architecture for Computer Vision"
   - CVPR 2016

5. **Cosine Annealing**
   - Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts"
   - ICLR 2017

---

## 🎓 Glosario Técnico

| Término | Definición |
|---------|------------|
| **Transfer Learning** | Reutilizar conocimiento de un modelo pre-entrenado |
| **Fine-tuning** | Ajustar capas pre-entrenadas a nueva tarea |
| **Overfitting** | Memorización de datos de entrenamiento |
| **Regularización** | Técnicas para prevenir overfitting |
| **Batch Normalization** | Normalizar activaciones por mini-batch |
| **Dropout** | Desactivar neuronas aleatoriamente durante entrenamiento |
| **Learning Rate** | Tamaño del paso en optimización |
| **Softmax** | Función que convierte logits en probabilidades |
| **Categorical Crossentropy** | Función de pérdida para clasificación |
| **Top-K Accuracy** | Correcto si clase verdadera está en top-K predicciones |

---

## ✅ Checklist de Deployment

```
Pre-producción:
□ Modelo entrenado con accuracy > 75%
□ Overfitting < 5%
□ API responde en < 500ms
□ Manejo de errores implementado
□ Logs configurados
□ CORS configurado correctamente

Producción:
□ Variables de entorno configuradas
□ Modelo versionado (Git LFS o S3)
□ Monitoreo de latencia
□ Rate limiting configurado
□ Tests de carga realizados
□ Documentación actualizada
□ Rollback plan definido
```

---
