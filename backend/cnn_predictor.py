"""
Predictor CNN Real - Carga y usa el modelo TensorFlow entrenado
"""
import os
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
import io

# Configurar TensorFlow antes de importarlo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
    print(f"✅ TensorFlow {tf.__version__} disponible")
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠️  TensorFlow no disponible - usando modo fallback")

class CNNPredictor:
    def __init__(self, model_path='models/breakfast_cnn_model_optimized.h5',
                 class_names_path='models/class_names.pkl'):
        """
        Inicializa el predictor CNN con el modelo entrenado

        Args:
            model_path: Ruta al archivo .h5 del modelo
            class_names_path: Ruta al archivo pickle con nombres de clases
        """
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.model = None
        self.class_names = []
        self.img_size = (224, 224)

        # Base de datos de calorías y nutrición (por 100g)
        self.nutrition_data = {
            'apple_pie': {'calories': 237, 'protein': 2.4, 'carbs': 34.0, 'fat': 11.0},
            'beignets': {'calories': 347, 'protein': 6.8, 'carbs': 35.2, 'fat': 20.5},
            'bread_pudding': {'calories': 153, 'protein': 4.2, 'carbs': 22.7, 'fat': 5.8},
            'breakfast_burrito': {'calories': 231, 'protein': 11.2, 'carbs': 25.8, 'fat': 9.4},
            'cannoli': {'calories': 213, 'protein': 4.6, 'carbs': 25.2, 'fat': 10.8},
            'carrot_cake': {'calories': 415, 'protein': 4.0, 'carbs': 45.0, 'fat': 25.0},
            'cheesecake': {'calories': 321, 'protein': 5.5, 'carbs': 25.9, 'fat': 22.5},
            'chocolate_cake': {'calories': 371, 'protein': 4.9, 'carbs': 50.7, 'fat': 16.9},
            'churros': {'calories': 117, 'protein': 1.7, 'carbs': 12.3, 'fat': 6.8},
            'club_sandwich': {'calories': 282, 'protein': 15.8, 'carbs': 28.4, 'fat': 12.6},
            'croque_madame': {'calories': 298, 'protein': 16.2, 'carbs': 18.5, 'fat': 18.9},
            'cup_cakes': {'calories': 305, 'protein': 3.2, 'carbs': 45.8, 'fat': 12.4},
            'donuts': {'calories': 452, 'protein': 4.9, 'carbs': 51.3, 'fat': 25.6},
            'eggs_benedict': {'calories': 230, 'protein': 12.8, 'carbs': 8.2, 'fat': 16.4},
            'french_toast': {'calories': 166, 'protein': 5.9, 'carbs': 18.8, 'fat': 7.4},
            'grilled_cheese_sandwich': {'calories': 291, 'protein': 12.8, 'carbs': 24.2, 'fat': 16.8},
            'huevos_rancheros': {'calories': 144, 'protein': 8.9, 'carbs': 8.7, 'fat': 8.2},
            'omelette': {'calories': 154, 'protein': 10.6, 'carbs': 1.2, 'fat': 11.7},
            'pancakes': {'calories': 227, 'protein': 6.2, 'carbs': 28.3, 'fat': 10.3},
            'strawberry_shortcake': {'calories': 344, 'protein': 4.4, 'carbs': 49.8, 'fat': 14.6},
            'waffles': {'calories': 291, 'protein': 7.9, 'carbs': 33.4, 'fat': 14.7}
        }

        # Cargar modelo automáticamente
        self.load_model()

    def load_model(self):
        """Carga el modelo CNN entrenado y los nombres de clases"""
        if not HAS_TENSORFLOW:
            print("⚠️  TensorFlow no disponible. Instala con: pip install tensorflow")
            return False

        try:
            # Verificar que existen los archivos
            if not self.model_path.exists():
                print(f"❌ Modelo no encontrado en: {self.model_path}")
                print(f"   Ejecuta primero: python train_cnn_model.py")
                return False

            if not self.class_names_path.exists():
                print(f"❌ Archivo de clases no encontrado en: {self.class_names_path}")
                return False

            # Cargar modelo
            print(f"🔄 Cargando modelo desde: {self.model_path}")
            self.model = keras.models.load_model(str(self.model_path))

            # Cargar nombres de clases
            with open(self.class_names_path, 'rb') as f:
                self.class_names = pickle.load(f)

            print(f"✅ Modelo CNN cargado correctamente")
            print(f"   • Clases: {len(self.class_names)}")
            print(f"   • Input shape: {self.model.input_shape}")

            return True

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            import traceback
            traceback.print_exc()
            return False

    def preprocess_image(self, image_bytes):
        """
        Preprocesa la imagen para el modelo CNN

        Args:
            image_bytes: Bytes de la imagen

        Returns:
            numpy array procesado listo para predicción
        """
        try:
            # Abrir imagen desde bytes
            img = Image.open(io.BytesIO(image_bytes))

            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Redimensionar a 224x224 (tamaño esperado por MobileNetV2)
            img = img.resize(self.img_size)

            # Convertir a array numpy
            img_array = np.array(img)

            # Normalizar (0-255 -> 0-1)
            img_array = img_array.astype('float32') / 255.0

            # Añadir dimensión de batch
            img_array = np.expand_dims(img_array, axis=0)

            return img_array

        except Exception as e:
            print(f"❌ Error preprocesando imagen: {e}")
            raise

    def predict(self, image_bytes):
        """
        Realiza la predicción sobre la imagen

        Args:
            image_bytes: Bytes de la imagen a clasificar

        Returns:
            dict con la predicción y calorías estimadas
        """
        # Verificar que el modelo está cargado
        if self.model is None:
            return {
                'error': 'Modelo no disponible',
                'message': 'El modelo CNN no está entrenado. Ejecuta train_cnn_model.py primero',
                'predicted_class': 'unknown',
                'confidence': 0.0,
                'calories': 0
            }

        try:
            # Preprocesar imagen
            img_array = self.preprocess_image(image_bytes)

            # Hacer predicción
            predictions = self.model.predict(img_array, verbose=0)[0]

            # Obtener clase predicha y confianza
            predicted_idx = np.argmax(predictions)
            predicted_class = self.class_names[predicted_idx]
            confidence = float(predictions[predicted_idx])

            # Top 3 predicciones
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = [
                {
                    'class': self.class_names[idx],
                    'confidence': float(predictions[idx])
                }
                for idx in top_indices
            ]

            # Obtener información nutricional
            nutrition = self.nutrition_data.get(predicted_class, {
                'calories': 250, 'protein': 8.0, 'carbs': 30.0, 'fat': 12.0
            })

            # Calcular calorías para porción estándar (150g)
            portion_size_g = 150
            estimated_calories = int(nutrition['calories'] * portion_size_g / 100)

            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'calories_per_100g': nutrition['calories'],
                'estimated_calories': estimated_calories,
                'portion_size_g': portion_size_g,
                'nutrition': {
                    'protein': round(nutrition['protein'] * portion_size_g / 100, 1),
                    'carbohydrates': round(nutrition['carbs'] * portion_size_g / 100, 1),
                    'fat': round(nutrition['fat'] * portion_size_g / 100, 1)
                },
                'top_predictions': top_predictions,
                'model_info': {
                    'type': 'CNN - MobileNetV2',
                    'num_classes': len(self.class_names)
                }
            }

            print(f"🔮 Predicción: {predicted_class} (confianza: {confidence:.3f})")

            return result

        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'predicted_class': 'unknown',
                'confidence': 0.0,
                'calories': 0
            }

    def get_model_status(self):
        """Obtiene el estado del modelo"""
        if self.model is None:
            return {
                'loaded': False,
                'message': 'Modelo no cargado. Ejecuta train_cnn_model.py primero'
            }

        return {
            'loaded': True,
            'type': 'CNN - MobileNetV2',
            'num_classes': len(self.class_names),
            'classes': self.class_names,
            'input_shape': list(self.model.input_shape[1:])
        }
