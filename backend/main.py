#!/usr/bin/env python3
"""
API para Detección de Calorías con CNN Real
Backend con modelo TensorFlow MobileNetV2 entrenado
"""

import os
import logging
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Importar predictor CNN REAL
from cnn_predictor import CNNPredictor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Calorie Detection API",
    description="API para detectar calorías en imágenes de comida usando CNN (MobileNetV2)",
    version="3.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar predictor CNN
logger.info("🔄 Inicializando predictor CNN...")
cnn_predictor = CNNPredictor(
    model_path='models/breakfast_cnn_model.h5',
    class_names_path='models/class_names.pkl'
)

# Verificar estado del modelo
model_status = cnn_predictor.get_model_status()

if model_status['loaded']:
    logger.info("✅ Modelo CNN cargado correctamente")
    logger.info(f"   • Tipo: {model_status['type']}")
    logger.info(f"   • Clases: {model_status['num_classes']}")
else:
    logger.warning("⚠️  Modelo CNN no disponible")
    logger.warning("   • Ejecuta primero: python train_cnn_model.py")
    logger.warning("   • La API funcionará pero con mensaje de error")



@app.post("/predict")
async def predict_food_calories(file: UploadFile = File(...)):
    """
    Endpoint principal: Detectar calorías en imagen de comida usando CNN

    Args:
        file: Imagen de comida (JPG, PNG, etc.)

    Returns:
        JSON con predicción, calorías y información nutricional
    """
    # Validar archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        logger.info(f"📸 Procesando imagen: {file.filename}")

        # Leer bytes de la imagen
        image_bytes = await file.read()

        # Predecir con CNN
        result = cnn_predictor.predict(image_bytes)

        # Verificar si hubo error
        if 'error' in result:
            logger.error(f"❌ Error en predicción: {result['error']}")
            raise HTTPException(
                status_code=500,
                detail=result.get('message', 'Error en predicción')
            )

        logger.info(f"✅ Predicción: {result['predicted_class']} ({result['confidence']:.2%})")

        # Formatear respuesta
        response = {
            'success': True,
            'predicted_class': result['predicted_class'],
            'display_name': result['predicted_class'].replace('_', ' ').title(),
            'confidence': result['confidence'],
            'estimated_calories': result['estimated_calories'],
            'portion_size_g': result.get('portion_size_g', 150),
            'calories_per_100g': result['calories_per_100g'],
            'nutrition': result['nutrition'],
            'top_predictions': result.get('top_predictions', []),
            'model_info': result['model_info']
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/health")
async def health_check():
    """Verificar estado de la API y del modelo"""
    model_status = cnn_predictor.get_model_status()

    return {
        'status': 'healthy',
        'api_version': '3.0.0',
        'model_loaded': model_status['loaded'],
        'model_info': model_status
    }

@app.get("/classes")
async def get_available_classes():
    """Obtener todas las clases que el modelo puede predecir"""
    model_status = cnn_predictor.get_model_status()

    if not model_status['loaded']:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta train_cnn_model.py primero"
        )

    return {
        'total_classes': model_status['num_classes'],
        'classes': model_status['classes']
    }

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        'message': 'Calorie Detection API',
        'version': '3.0.0',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'classes': '/classes (GET)',
            'docs': '/docs (GET)'
        }
    }

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🚀 Iniciando Calorie Detection API v3.0")
    logger.info("=" * 70)

    # Verificar modelo
    if model_status['loaded']:
        logger.info(f"✅ Modelo CNN listo para predicciones")
    else:
        logger.warning("⚠️  Modelo no disponible - La API retornará errores")
        logger.warning("   Solución: Ejecuta 'python train_cnn_model.py'")

    logger.info("=" * 70)

    # Iniciar servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
