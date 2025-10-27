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

**¡Ya no hay simulaciones! Todo es REAL.** 🎉
