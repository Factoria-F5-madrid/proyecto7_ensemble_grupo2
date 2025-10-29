# Backend - Calorie Detection API

API con CNN real para detectar calorías en imágenes de comida usando TensorFlow y MobileNetV2.

## 🚀 Instalación y Configuración

### 1. Activar Entorno Virtual

```bash
# Desde la raíz del proyecto
source ../.venv/bin/activate
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Nota:** Esto instalará:
- TensorFlow 2.20.0 (compatible con Python 3.13)
- FastAPI para la API
- Pillow para procesamiento de imágenes
- NumPy, Pandas, Scikit-learn para ML
- Matplotlib, tqdm para visualización y progreso

### 3. Entrenar el Modelo CNN

**IMPORTANTE:** Debes entrenar el modelo antes de usar la API.

```bash
python train_cnn_model.py
```

Este proceso:
- ✅ Carga el dataset de 21 clases de desayuno desde `../notebooks/data/`
- ✅ Preprocesa ~10,000+ imágenes
- ✅ Entrena un modelo CNN con Transfer Learning (MobileNetV2)
- ✅ Guarda el modelo en `models/breakfast_cnn_model.h5`
- ✅ Guarda los nombres de clases en `models/class_names.pkl`
- ⏱️ Tiempo estimado: 15-30 minutos (dependiendo del hardware)

**Métricas esperadas:**
- Accuracy: ~85-92%
- Overfitting: <5%

### 4. Ejecutar la API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en: `http://localhost:8000`

## 📊 Endpoints Disponibles

### POST `/predict`
Predice la clase de comida y las calorías estimadas.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@pancakes.jpg"
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "pancakes",
  "display_name": "Pancakes",
  "confidence": 0.92,
  "estimated_calories": 340,
  "portion_size_g": 150,
  "calories_per_100g": 227,
  "nutrition": {
    "protein": 9.3,
    "carbohydrates": 42.5,
    "fat": 15.5
  },
  "top_predictions": [
    {"class": "pancakes", "confidence": 0.92},
    {"class": "waffles", "confidence": 0.05},
    {"class": "french_toast", "confidence": 0.02}
  ],
  "model_info": {
    "type": "CNN - MobileNetV2",
    "num_classes": 21
  }
}
```

### GET `/health`
Verifica el estado de la API y del modelo.

```bash
curl http://localhost:8000/health
```

### GET `/classes`
Lista todas las clases que el modelo puede predecir.

```bash
curl http://localhost:8000/classes
```

### GET `/docs`
Documentación interactiva (Swagger UI): `http://localhost:8000/docs`

## 🏗️ Estructura del Proyecto

```
backend/
├── main.py                    # API FastAPI principal
├── cnn_predictor.py          # Predictor CNN real (NO simulado)
├── train_cnn_model.py        # Script de entrenamiento
├── requirements.txt          # Dependencias
├── models/                   # Modelos entrenados
│   ├── breakfast_cnn_model.h5      # Modelo TensorFlow (generado)
│   ├── class_names.pkl             # Nombres de clases (generado)
│   ├── training_history.pkl        # Historial de entrenamiento (generado)
│   └── training_curves.png         # Gráficas de entrenamiento (generado)
└── README.md                 # Este archivo
```

## 🍳 Clases Soportadas (21 Desayunos)

1. apple_pie
2. beignets
3. bread_pudding
4. breakfast_burrito
5. cannoli
6. carrot_cake
7. cheesecake
8. chocolate_cake
9. churros
10. club_sandwich
11. croque_madame
12. cup_cakes
13. donuts
14. eggs_benedict
15. french_toast
16. grilled_cheese_sandwich
17. huevos_rancheros
18. omelette
19. pancakes
20. strawberry_shortcake
21. waffles

## 🔧 Solución de Problemas

### Error: "Modelo no disponible"

**Causa:** No has entrenado el modelo todavía.

**Solución:**
```bash
python train_cnn_model.py
```

### Error: "No se encuentra el dataset"

**Causa:** Los archivos NPZ no están en `../notebooks/data/npz_files/`

**Solución:**
- Verifica que la carpeta `notebooks/data/npz_files/` existe
- Verifica que contiene archivos `.npz`
- Verifica que `notebooks/data/food101_desayuno_preprocessed.pkl` existe

### Error de importación de TensorFlow

**Causa:** TensorFlow no está instalado o hay incompatibilidad de versión.

**Solución:**
```bash
pip install --upgrade tensorflow==2.20.0
```

### El modelo tiene bajo accuracy

**Solución:**
- Aumenta el número de épocas en `train_cnn_model.py` (línea 33: `EPOCHS = 30`)
- Ajusta el learning rate (línea 34: `LEARNING_RATE = 0.001`)
- Considera hacer fine-tuning del modelo base

## 📈 Mejoras Futuras

- [ ] Data Augmentation durante entrenamiento
- [ ] Fine-tuning de capas congeladas
- [ ] Ensemble con múltiples modelos
- [ ] Detección de múltiples platos en una imagen
- [ ] API de feedback para mejorar el modelo
- [ ] Dockerización del backend
- [ ] Tests unitarios
- [ ] CI/CD pipeline

## 🤝 Contribuir

Este es un proyecto educativo de Factoría F5. Para contribuir:

1. Crea un branch desde `development`
2. Haz tus cambios
3. Crea un Pull Request

## 📝 Notas Técnicas

- **Arquitectura:** MobileNetV2 + Transfer Learning
- **Framework:** TensorFlow/Keras
- **Input size:** 224x224x3 (RGB)
- **Batch size:** 32
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical Crossentropy
- **Regularización:** Dropout (0.3, 0.3, 0.2) + L2 (0.01)

---


# 🍳 Food-101 Breakfast Calorie Detector

Sistema completo de detección de calorías en imágenes de desayuno utilizando Deep Learning (CNN) con Transfer Learning.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![React](https://img.shields.io/badge/React-18.3-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## � Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Stack Tecnológico](#-stack-tecnológico)
- [Arquitectura del Proyecto](#-arquitectura-del-proyecto)
- [Instalación Rápida](#-instalación-rápida)
- [Uso](#-uso)
- [Modelo CNN](#-modelo-cnn)
- [API Endpoints](#-api-endpoints)
- [Clases Soportadas](#-clases-soportadas)
- [Resultados](#-resultados)
- [Documentación Completa](#-documentación-completa)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

---

## 🎯 Descripción

Aplicación web completa que utiliza **Computer Vision** y **Deep Learning** para identificar automáticamente platos de desayuno en imágenes y estimar su contenido calórico y nutricional.

El sistema emplea un modelo CNN basado en **MobileNetV2** con Transfer Learning, entrenado en un subset del dataset **Food-101** (21 clases de desayunos), alcanzando una precisión del **75-80%** con overfitting controlado (<5%).

---

## ✨ Características

- ✅ **Clasificación de 21 tipos de desayunos** con Deep Learning
- ✅ **Estimación automática de calorías** y macronutrientes
- ✅ **Transfer Learning** con MobileNetV2 (ImageNet pre-trained)
- ✅ **Regularización avanzada** (Dropout, L2, Label Smoothing)
- ✅ **Data Augmentation** agresivo (9 transformaciones)
- ✅ **API REST** con FastAPI + documentación automática (Swagger)
- ✅ **Frontend moderno** con React + Vite
- ✅ **Predicción en tiempo real** (<3 segundos)
- ✅ **Top-3 predicciones** con confianza
- ✅ **Dataset preprocessado** (~21,000 imágenes, 224x224)

---

## 🛠️ Stack Tecnológico

### **Backend**
- **Python 3.12**
- **TensorFlow 2.20.0** / Keras 3.x
- **FastAPI 0.104.1** - Framework API REST
- **Uvicorn 0.24.0** - ASGI server
- **Pillow 10.1.0** - Procesamiento de imágenes
- **NumPy**, **Pandas**, **Scikit-learn** - Data Science

### **Frontend**
- **React 18.3** + **Vite 6.0**
- **JavaScript (ES6+)**
- **CSS3** - Diseño responsive
- **Fetch API** - Comunicación con backend

### **Machine Learning**
- **MobileNetV2** - Arquitectura base (ImageNet)
- **Transfer Learning** - Fine-tuning últimas 20 capas
- **Food-101 Dataset** - 21 clases de desayunos
- **Data Augmentation** - 9 transformaciones

---

## 🏗️ Arquitectura del Proyecto

```
proyecto7_ensemble_grupo2/
│
├── backend/                          # API REST + Modelo CNN
│   ├── main.py                       # FastAPI app
│   ├── cnn_predictor.py             # Predictor CNN
│   ├── train_cnn_model.py           # Script de entrenamiento
│   ├── requirements.txt             # Dependencias Python
│   ├── README_EXPLICATION.md        # Documentación técnica
│   └── models/                       # Modelos entrenados
│       ├── breakfast_cnn_model_optimized.h5    # Modelo CNN (50MB)
│       ├── class_names.pkl                     # 21 clases
│       ├── training_history.json               # Métricas
│       └── training_curves.png                 # Gráficas
│
├── frontend/                         # Interfaz web React
│   ├── src/
│   │   ├── pages/
│   │   │   └── Home.jsx             # Página principal
│   │   ├── components/              # Componentes reutilizables
│   │   ├── services/                # API calls
│   │   └── main.jsx                 # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/                        # Análisis y preprocesamiento
│   ├── EDA_UNIVERSAL.ipynb          # Análisis exploratorio
│   └── data/
│       └── desayuno_preprocessed/   # Dataset preprocessado
│           ├── food101_desayuno_preprocessed.pkl
│           └── npz_files/           # ~20,987 imágenes (224x224)
│
├── .venv/                            # Entorno virtual Python
├── GUIA_DESPLIEGUE_COMPLETA.md      # Guía de despliegue
└── README.md                         # Este archivo
```

---

## ⚡ Instalación Rápida

### **Requisitos Previos**
- Python 3.12+
- Node.js 18+
- 8GB RAM mínimo
- 2GB espacio en disco

### **1. Clonar el Repositorio**
```bash
git clone https://github.com/Factoria-F5-madrid/proyecto7_ensemble_grupo2.git
cd proyecto7_ensemble_grupo2
```

### **2. Configurar Backend**
```bash
# Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
cd backend
pip install -r requirements.txt

# Entrenar el modelo (20-35 minutos)
python train_cnn_model.py

# Iniciar API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **3. Configurar Frontend**
```bash
# En otra terminal
cd frontend

# Instalar dependencias
npm install

# Configurar variable de entorno
echo "VITE_API_URL=http://localhost:8000" > .env

# Iniciar servidor de desarrollo
npm run dev
```

### **4. Acceder a la Aplicación**
- **Frontend:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## 🚀 Uso

### **Desde la Interfaz Web**
1. Abre http://localhost:5173
2. Sube una imagen de comida (JPG/PNG)
3. Obtén la predicción con calorías estimadas

### **Desde la API (cURL)**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@pancakes.jpg"
```

**Respuesta:**
```json
{
  "success": true,
  "predicted_class": "pancakes",
  "display_name": "Pancakes",
  "confidence": 0.89,
  "estimated_calories": 340,
  "nutrition": {
    "protein": 9.3,
    "carbohydrates": 42.5,
    "fat": 15.5
  },
  "top_predictions": [
    {"class": "pancakes", "confidence": 0.89},
    {"class": "waffles", "confidence": 0.06},
    {"class": "french_toast", "confidence": 0.03}
  ]
}
```

---

## 🧠 Modelo CNN

### **Arquitectura**
```
Input (224x224x3)
    ↓
MobileNetV2 (ImageNet pre-trained)
├── Frozen: 134 capas (40.6% params)
└── Trainable: 20 capas (59.4% params)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dropout (0.5)
    ↓
Dense(256, ReLU) + L2(5e-4)
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense(21, Softmax) + L2(5e-4)
    ↓
Output (21 clases)
```

### **Hiperparámetros Optimizados**
- **Input Size:** 224x224x3
- **Batch Size:** 16
- **Epochs:** 30 (con Early Stopping)
- **Learning Rate:** 1e-3 → 1e-6 (Cosine Annealing + Warmup)
- **Dropout:** 0.5, 0.3
- **L2 Regularization:** 5e-4
- **Label Smoothing:** 0.2
- **Data Augmentation:** 9 técnicas

### **Parámetros del Modelo**
- **Total:** 2,597,461 parámetros
- **Entrenables:** 1,542,485 (59.4%)
- **Frozen:** 1,054,976 (40.6%)

---

## 📡 API Endpoints

### **POST `/predict`**
Predice la clase de comida y estima calorías.

**Parámetros:**
- `file` (multipart/form-data): Imagen JPG/PNG

**Respuesta:** JSON con predicción, confianza, calorías y nutrición

---

### **GET `/health`**
Verifica el estado de la API y del modelo.

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "type": "CNN - MobileNetV2",
    "num_classes": 21
  }
}
```

---

### **GET `/classes`**
Lista todas las clases soportadas.

**Respuesta:**
```json
{
  "num_classes": 21,
  "classes": ["apple_pie", "beignets", ...]
}
```

---

### **GET `/docs`**
Documentación interactiva Swagger UI.

---

## 🍳 Clases Soportadas

El modelo puede clasificar **21 tipos de desayunos**:

| Categoría | Clases |
|-----------|--------|
| **Pasteles** | apple_pie, carrot_cake, cheesecake, chocolate_cake, strawberry_shortcake |
| **Dulces fritos** | beignets, churros, donuts |
| **Postres** | bread_pudding, cannoli, cup_cakes |
| **Sándwiches** | club_sandwich, croque_madame, grilled_cheese_sandwich |
| **Desayunos calientes** | breakfast_burrito, eggs_benedict, french_toast, huevos_rancheros, omelette, pancakes, waffles |

---

## � Resultados

### **Métricas del Modelo**
- ✅ **Test Accuracy:** 75-80%
- ✅ **Test Top-3 Accuracy:** 85-90%
- ✅ **Train-Val Gap:** <5% (overfitting controlado)
- ✅ **Tiempo de inferencia:** <3 segundos
- ✅ **Tamaño del modelo:** ~50MB

### **Dataset**
- **Total imágenes:** 20,987
- **Clases:** 21
- **Resolución:** 224x224 RGB
- **Train/Val/Test:** 70% / 15% / 15%

---

## 📚 Documentación Completa

- **[GUIA_DESPLIEGUE_COMPLETA.md](./GUIA_DESPLIEGUE_COMPLETA.md)** - Instrucciones de despliegue paso a paso
- **[backend/README_EXPLICATION.md](./backend/README_EXPLICATION.md)** - Documentación técnica del modelo CNN
- **[notebooks/README.md](./notebooks/README.md)** - Análisis exploratorio y preprocesamiento

---

## 🤝 Contribuir

Este es un proyecto educativo de **Factoría F5 Madrid**.

Para contribuir:
1. Fork el repositorio
2. Crea un branch desde `development`
3. Haz tus cambios
4. Crea un Pull Request

---

## � Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

## 👥 Equipo

Desarrollado por **Grupo 2 - Proyecto 7 Ensemble**
Factoría F5 Madrid - Promoción 2025

