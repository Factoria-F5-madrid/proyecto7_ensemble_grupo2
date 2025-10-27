# 🍽️ Food Classification & Calorie Calculator

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
- 🤖 **Múltiples algoritmos de ML**: LightGBM, XGBoost, Random Forest
- 🧠 **Red neuronal** para comparación de resultados
- 📊 **Cálculo automático de calorías** basado en la clasificación
- 🌐 **Interfaz web intuitiva** para subir y analizar imágenes
- 📈 **API REST** para integración con otras aplicaciones

## 🛠️ Tecnologías

### Backend
- **Python 3.8+**
- **TensorFlow/Keras** - Deep Learning y feature extraction
- **LightGBM** - Gradient boosting optimizado
- **XGBoost** - Extreme gradient boosting
- **Scikit-learn** - Random Forest y preprocesamiento
- **Flask/FastAPI** - API REST
- **NumPy/Pandas** - Manipulación de datos

### Frontend
- **React/Vue.js** - Interfaz de usuario
- **Axios** - Peticiones HTTP
- **CSS/Tailwind** - Estilos

### Otros
- **Kaggle** - Entrenamiento en GPU
- **Colab** - Entrenamiento en GPU
- **OpenCV** - Procesamiento de imágenes
- **Matplotlib/Seaborn** - Visualización de resultados

## 📁 Estructura del Proyecto

```
proyecto7_ensemble_grupo2/
│
├── backend/
│   ├── models/              # Modelos entrenados (.pkl, .h5)
│   ├── api/                 # Endpoints de la API
│   ├── utils/               # Funciones auxiliares
│   │   ├── calorie_calculator.py
│   │   └── image_processor.py
│   ├── data/                # Información nutricional
│   └── app.py               # Aplicación principal
│
├── frontend/
│   ├── src/
│   │   ├── components/      # Componentes React/Vue
│   │   ├── services/        # Servicios API
│   │   └── App.js           # Componente principal
│   └── public/              # Assets estáticos
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_lightgbm_training.ipynb
│   ├── 03_xgboost_training.ipynb
│   ├── 04_random_forest_training.ipynb
│   └── 05_neural_network_training.ipynb
│
├── data/
│   ├── raw/                 # Datos originales de Kaggle
│   ├── processed/           # Datos preprocesados
│   └── nutritional_info.csv # Información calórica
│
├── tests/                   # Tests unitarios
├── requirements.txt         # Dependencias Python
├── README.md
└── LICENSE

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

### Ejemplo con Python

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

## 🤖 Modelos

### 1. LightGBM
- **Feature extraction**: ResNet50 (ImageNet)
- **Accuracy**: ~65%
- **F1-Score**: 0.65
- **Ventajas**: Rápido, eficiente en memoria

### 2. XGBoost
- **Feature extraction**: ResNet50 (ImageNet)
- **Accuracy**: ~XX%
- **F1-Score**: X.XX
- **Ventajas**: Robusto contra overfitting

### 3. Random Forest
- **Feature extraction**: ResNet50 (ImageNet)
- **Accuracy**: ~XX%
- **F1-Score**: X.XX
- **Ventajas**: Interpretable, baseline sólido

### 4. Red Neuronal
- **Arquitectura**: Transfer Learning con ResNet50
- **Accuracy**: ~XX%
- **F1-Score**: X.XX
- **Ventajas**: Mejor captura de features complejos

### Ensemble Final
Combinación ponderada de los 4 modelos para maximizar precisión.

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

### Preprocesamiento
1. Resize a 224x224 píxeles
2. Normalización con ImageNet stats
3. Augmentation (rotación, flip, zoom)
4. Feature extraction con ResNet50

## 📈 Resultados

### Métricas Generales

| Modelo | Accuracy | F1-Score | Tiempo Inferencia |
|--------|----------|----------|-------------------|
| LightGBM | 65.46% | 0.6529 | ~50ms |
| XGBoost | XX.XX% | X.XXXX | ~XXms |
| Random Forest | XX.XX% | X.XXXX | ~XXms |
| Neural Network | XX.XX% | X.XXXX | ~XXms |
| **Ensemble** | **XX.XX%** | **X.XXXX** | ~XXms |

### Clases con Mejor Performance
1. **Churros**: F1 = 0.825
2. **Donuts**: F1 = 0.XXX
3. **Club Sandwich**: F1 = 0.XXX

### Clases con Peor Performance
1. **Apple Pie**: F1 = 0.403
2. **XXX**: F1 = 0.XXX
3. **XXX**: F1 = 0.XXX

### Visualizaciones


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




### 2. Environment Setup

You can choose between using a Dev Container (recommended for a consistent environment) or a standard Python virtual environment.

#### Option A: Using Dev Containers (Recommended)

This method uses Docker to create a fully configured and isolated development environment.

1.  Open the cloned project folder in **Visual Studio Code**.
2.  VS Code will automatically detect the Dev Container configuration (`.devcontainer/devcontainer.json`) and show a notification in the bottom-right corner.
3.  Click on **"Reopen in Container"**.
4.  Wait for VS Code to build the Docker image and start the container. This might take a few minutes on the first run.

Once the container is running, you will have a terminal in a ready-to-use environment. To install the project's dependencies, run:

```bash
pip install -r requirements.txt
```

> **Pro-tip:** You can automate this step by uncommenting the `postCreateCommand` line in the `.devcontainer/devcontainer.json` file.

#### Option B: Using a Python Virtual Environment

If you prefer not to use Docker, you can set up a local virtual environment.

1.  **Create a virtual environment:**

    From the project's root directory, run the following command. We'll name the environment `venv`.

    ```bash
    python3 -m venv venv
    ```

2.  **Activate the virtual environment:**

    -   **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```

    -   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

    Your terminal prompt should now be prefixed with `(venv)`, indicating the environment is active.

3.  **Install dependencies:**

    With the virtual environment active, install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

Once your environment is set up, you can run the backend and frontend services. The recommended way is using Docker Compose, which orchestrates both services.

