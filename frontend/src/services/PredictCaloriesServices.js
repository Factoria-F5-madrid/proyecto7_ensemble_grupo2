// aqui van las peticiones a la api 

import axios from "axios";

const BASE_URL = "http://localhost:8000";

export async function predictCalories(data) {
    const response = await axios.post(`${BASE_URL}/predict-calories`, data);
    return response.data;
}