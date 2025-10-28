import axios from 'axios';

// Configuración base de la API
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// Servicio para predicción de calorías
export const predictCalories = async (imageFile) => {
  try {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await api.post('/predict', formData);
    return response.data;
  } catch (error) {
    console.error('Error en predicción de calorías:', error);
    
    if (error.response) {
      throw new Error(error.response.data.detail || 'Error del servidor');
    } else if (error.request) {
      throw new Error('No se pudo conectar con el servidor');
    } else {
      throw new Error('Error inesperado');
    }
  }
};

export default {
  predictCalories,
};

