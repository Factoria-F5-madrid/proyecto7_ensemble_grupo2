// frontend/src/services/predictCaloriesServices.js
import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
  headers: {
    Accept: "application/json",
  },
});

/**
 * Uploads an image file and asks the backend to predict the dish and calories.
 * - Expects backend endpoint POST /predict-breed (multipart/form-data, field "file")
 * - onUploadProgress is optional and receives progress percentage 0-100
 *
 * @param {File} file
 * @param {(percent: number) => void} [onUploadProgress]
 * @returns {Promise<Object>} response.data
 */
export async function predictCalories(file, onUploadProgress) {
  if (!file) throw new Error("No file provided");

  const form = new FormData();
  // field name "file" to match your backend; change if backend expects another name
  form.append("file", file);

  try {
    const res = await api.post("/predict-breed", form, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (evt) => {
        if (!evt.lengthComputable) return;
        const percent = Math.round((evt.loaded * 100) / evt.total);
        if (typeof onUploadProgress === "function") onUploadProgress(percent);
      },
    });
    return res.data;
  } catch (err) {
    // Normalize the error to make it easier for the UI to display
    const message =
      err?.response?.data?.message ||
      err?.response?.data ||
      err?.message ||
      "Upload failed";
    const status = err?.response?.status || null;
    const error = new Error(message);
    error.status = status;
    throw error;
  }
}

