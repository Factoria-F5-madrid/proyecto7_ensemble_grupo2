# 🍽️ Breakfast Predictor (con cálculo de calorías) 

> Proyecto de clasificación de alimentos y cálculo de calorías utilizando técnicas de Machine Learning y Deep Learning

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Tecnologías](#-tecnologías)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Modelos](#-modelos)
- [Dataset](#-dataset)
- [Resultados](#-resultados)
- [Equipo](#-equipo)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## 🎯 Descripción

Sistema inteligente de identificación de alimentos que permite a los usuarios subir fotografías de platos de comida para obtener:

- **Clasificación automática** del tipo de alimento
- **Cálculo de calorías** estimadas
- **Información nutricional** del plato identificado

El proyecto implementa un enfoque de **ensemble learning** combinando múltiples modelos de Machine Learning para mejorar la precisión de las predicciones.

## ✨ Características

- 🖼️ **Clasificación de imágenes** utilizando modelos pre-entrenados (ResNet50)
- 🤖 **Algoritmo de ML**: LightGBM
- 🧠 **Red neuronal** para comparación de resultados
- 📊 **Cálculo automático de calorías** basado en la clasificación
- 🌐 **Interfaz web intuitiva** para subir y analizar imágenes
- 📈 **API REST** para integración con otras aplicaciones
- 👩‍💻 **Project** para coordinar el trabajo: https://github.com/orgs/Factoria-F5-madrid/projects/48/views/

## 🛠️ Tecnologías

### Backend
- **Python 3.8+**
- **TensorFlow/Keras** - Deep Learning y feature extraction
- **LightGBM** - Gradient boosting optimizado
- **Scikit-learn** - Preprocesamiento
- **Flask/FastAPI** - API REST
- **NumPy/Pandas** - Manipulación de datos

### Frontend
- **React/Vue.js** - Interfaz de usuario
- **Axios** - Peticiones HTTP
- **CSS/Tailwind** - Estilos

### Otros
- **Kaggle/Colab** - Entrenamiento en GPU
- **Matplotlib/Seaborn** - Visualización de resultados

## 📁 Estructura del Proyecto

```
proyecto7_ensemble_grupo2/
│
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
│
├── frontend/
│   ├── src/
│   │   ├── components/      # Componentes React/Vue
│   │   ├── services/        # Servicios API
│   │   └── App.js           # Componente principal
│   └── public/              # Assets estáticos
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Lightgbm.ipynb
│ │
├── requirements.txt         # Dependencias Python
└── README.md

```

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- Node.js 14+ (para el frontend)
- GPU (recomendado para entrenamiento)

### Backend

```bash
# Clonar el repositorio
git clone https://github.com/Factoria-F5-madrid/proyecto7_ensemble_grupo2.git
cd proyecto7_ensemble_grupo2

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar el servidor
cd backend
python app.py
```


# 🧠 Backend CNN - Documentación Técnica 

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
│                         SISTEMA COMPLETO                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   FRONTEND  │─────▶│    BACKEND   │─────▶│  MODELO CNN    │
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

### 1.4 Stack Tecnológico

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


### Frontend

```bash
cd frontend

# Instalar dependencias
npm install

# Iniciar aplicación
npm start
```

La aplicación estará disponible en `http://localhost:3000`

## 💻 Uso

### Interfaz Web

1. Accede a la aplicación web
2. Sube una imagen de tu plato de comida
3. Espera el procesamiento (2-5 segundos)
4. Visualiza:
   - Tipo de alimento identificado
   - Porcentaje de confianza
   - Calorías estimadas
   - Información nutricional

### API REST

```python
import requests

# Endpoint de predicción
url = "http://localhost:5000/api/predict"

# Subir imagen
files = {'image': open('mi_plato.jpg', 'rb')}
response = requests.post(url, files=files)

# Respuesta
result = response.json()
print(f"Alimento: {result['class']}")
print(f"Confianza: {result['confidence']:.2%}")
print(f"Calorías: {result['calories']} kcal")
```

```python
from backend.utils.image_processor import load_and_preprocess_image
from backend.utils.calorie_calculator import get_calories
import pickle

# Cargar modelo
with open('backend/models/lightgbm_food_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Procesar imagen
image = load_and_preprocess_image('path/to/image.jpg')

# Predecir
prediction = model.predict(image)
food_class = class_names[prediction[0]]

# Calcular calorías
calories = get_calories(food_class)

print(f"Identificado: {food_class}")
print(f"Calorías: {calories} kcal")
```

## 📊 Dataset

### Origen
- **Fuente**: [Food-101 Dataset (Kaggle)](https://www.kaggle.com/dansbecker/food-101)
- **Clases**: 21 categorías de desayuno
- **Imágenes totales**: ~20,000
- **Split**: 80% entrenamiento, 20% test

### Clases de Alimentos

```
apple_pie, bread_pudding, breakfast_burrito, carrot_cake, 
cheese_plate, cheesecake, chicken_quesadilla, chicken_wings,
chocolate_cake, churros, club_sandwich, creme_brulee, 
croque_madame, cup_cakes, deviled_eggs, donuts, dumplings,
edamame, eggs_benedict, escargots, falafel
```

### Reducido a 21 clases (desayunos)

```
'apple_pie', 'beignets', 'bread_pudding', 'breakfast_burrito',
'cannoli', 'carrot_cake', 'cheesecake', 'chocolate_cake', 'churros', 'club_sandwich',
'croque_madame', 'cup_cakes', 'donuts', 'eggs_benedict',
'french_toast', 'grilled_cheese_sandwich', 'huevos_rancheros', 'omelette', 'pancakes',
'strawberry_shortcake', 'waffles'
```

## 👥 Equipo

Proyecto desarrollado por el **Grupo 2** de Factoría F5 Madrid:

- 👤 **Cristian Yeder** - [@github1](https://github.com/CristianYepes)
- 👤 **Lady** - [@github2](https://github.com/nikaLPFB)
- 👤 **María Dunaeva** - [@github3](https://github.com/MariaDunaeva1)
- 👤 **Yeder Pimentel** - [@github4](https://github.com/Yedpt)

## 🤝 Contribución

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- [Factoría F5 Madrid](https://factoriaf5.org/) por la formación y recursos
- [Kaggle](https://www.kaggle.com/) por el dataset y entorno de entrenamiento
- [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) por el dataset original

## 📞 Contacto

Para preguntas o sugerencias:
- 📧 Email: [tu-email@ejemplo.com]
- 🐛 Issues: [GitHub Issues](https://github.com/Factoria-F5-madrid/proyecto7_ensemble_grupo2/issues)

---
