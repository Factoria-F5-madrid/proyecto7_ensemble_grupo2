import React, { useRef, useState, useEffect } from "react";
import hero from "../assets/hero.png";
import scanBg from "../assets/comida.jpg";
import { predictCalories } from "../services/PredictCaloriesServices";

export default function Home() {
  const scanRef = useRef(null);
  const fileInputRef = useRef(null);

  // UI state for upload / prediction
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [predicting, setPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [isDragActive, setIsDragActive] = useState(false);

  // cleanup preview URL on unmount / when file changes
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, []);

  const scrollToScan = (e) => {
    e?.preventDefault();
    if (!scanRef.current) return;

    const NAV_OFFSET = 72;
    const top = scanRef.current.getBoundingClientRect().top + window.pageYOffset;
    window.scrollTo({
      top: Math.max(top - NAV_OFFSET, 0),
      behavior: "smooth",
    });
  };

  // helper to set file + preview and cleanup previous preview
  const setFileAndPreview = (file) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const url = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(url);
    setPredictionResult(null);
    setErrorMsg("");
    setUploadProgress(0);
  };

  // handle file selection (file input)
  const onSelectFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFileAndPreview(f);
  };

  // drag & drop handlers
  const onDropFile = (e) => {
    e.preventDefault();
    setIsDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    setFileAndPreview(f);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragActive(false);
  };

  // remove selected file and revoke preview url
  const clearSelectedFile = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setUploadProgress(0);
    setPredictionResult(null);
    setErrorMsg("");
  };

  // call backend predict service
  const onPredict = async () => {
    if (!selectedFile) {
      setErrorMsg("Por favor selecciona una imagen primero.");
      return;
    }

    try {
      setPredicting(true);
      setErrorMsg("");
      setUploadProgress(0);
      setPredictionResult(null);

      const data = await predictCalories(selectedFile, (percent) => {
        setUploadProgress(percent);
      });

      setPredictionResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
      setErrorMsg(err?.message || "Error predicting the image");
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      {/* HERO */}
      <header id="home" className="relative">
        <div
          className="h-screen bg-cover bg-center flex items-center"
          style={{ backgroundImage: `url(${hero})` }}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-black/40 to-black/30"></div>

          <div className="container mx-auto px-6 lg:px-20 relative z-10">
            <div className="max-w-3xl text-center mx-auto py-24">
              <h1 className="text-5xl md:text-6xl font-serif text-black leading-tight drop-shadow-lg">
                Calorie Detection <span className="block">AI App</span>
              </h1>
              <p className="mt-6 text-lg text-black/90">
                Detecta automáticamente las calorías de tus comidas usando inteligencia artificial
              </p>

              <div className="mt-8 flex items-center justify-center gap-4">
                <button
                  onClick={scrollToScan}
                  className="px-6 py-3 rounded-full bg-red-600 text-white font-medium shadow-lg hover:bg-red-700 transition"
                >
                  Escanear Comida
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* SCAN SECTION */}
      <main className="container mx-auto px-6 lg:px-20 -mt-24 relative z-10">
        <section
          ref={scanRef}
          id="scan"
          className="relative bg-cover bg-center rounded-xl overflow-hidden shadow-lg"
          style={{ backgroundImage: `url(${scanBg})` }}
        >
          <div className="absolute inset-0 bg-black/50"></div>

          <div className="relative z-10 py-16 px-8 text-center">
            <h2 className="text-4xl font-serif text-white mb-4">
              Detecta las Calorías de tu Comida
            </h2>
            <p className="text-lg text-white/90 mb-8 max-w-2xl mx-auto">
              Sube una imagen de tu desayuno y nuestro modelo CNN entrenado con datos EDA 
              te dirá cuántas calorías contiene y información nutricional detallada.
            </p>

            <div
              className="bg-white/90 backdrop-blur-md rounded-xl shadow-md p-8"
              onDrop={onDropFile}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
            >
              <div
                className={`relative border-2 border-dashed ${
                  selectedFile ? "border-amber-400" : isDragActive ? "border-amber-300 bg-amber-50" : "border-gray-200"
                } rounded-lg p-6 flex flex-col items-center gap-4 transition`}
              >
                {/* Preview or icon */}
                {previewUrl ? (
                  <>
                    <img
                      src={previewUrl}
                      alt="preview"
                      className="w-32 h-32 object-cover rounded-lg shadow-md border"
                    />
                    <button
                      onClick={clearSelectedFile}
                      className="absolute -top-2 -right-2 bg-white rounded-full p-1 shadow text-gray-600 hover:bg-gray-100"
                      aria-label="Remove image"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M6.28 5.22a.75.75 0 011.06 0L10 7.88l2.66-2.66a.75.75 0 111.06 1.06L11.06 8.94l2.66 2.66a.75.75 0 11-1.06 1.06L10 10l-2.66 2.66a.75.75 0 11-1.06-1.06L8.94 8.94 6.28 6.28a.75.75 0 010-1.06z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </>
                ) : (
                  <>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-10 w-10 text-gray-400"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="1.5"
                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                      />
                    </svg>
                    <p className="text-gray-500">Arrastra y suelta tu imagen aquí</p>
                  </>
                )}

                <label
                  htmlFor="file"
                  className="cursor-pointer px-4 py-2 bg-gray-100 border rounded-md text-sm hover:bg-gray-200"
                >
                  Selecciona una imagen
                </label>
                <input
                  id="file"
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={onSelectFile}
                />
              </div>

              <div className="mt-4">
                {selectedFile && (
                  <p className="text-sm text-gray-600">Archivo seleccionado: {selectedFile.name}</p>
                )}

                {uploadProgress > 0 && uploadProgress < 100 && (
                  <div className="mt-3 w-full bg-gray-200 rounded-full h-3">
                    <div
                      className="h-3 rounded-full bg-amber-400"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                )}
                {uploadProgress === 100 && predicting && (
                  <p className="mt-2 text-sm text-gray-500">Procesando con CNN...</p>
                )}
                {errorMsg && <p className="mt-2 text-sm text-red-600">{errorMsg}</p>}
              </div>

              <div className="mt-8 flex justify-center">
                <button
                  onClick={onPredict}
                  disabled={predicting}
                  className="px-8 py-3 rounded-full bg-amber-400 text-white font-medium shadow hover:bg-amber-500 transition disabled:opacity-60"
                  type="button"
                >
                  {predicting ? "Analizando..." : "Detectar Calorías"}
                </button>
              </div>

              {/* Resultado de la predicción */}
              {predictionResult && (
                <div className="mt-6 text-left bg-white p-4 rounded-md border">
                  <h3 className="font-semibold mb-2 text-lg">🍽️ Resultado de la Predicción</h3>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-gray-700">Plato detectado:</span>
                      <span className="text-lg font-semibold text-amber-600">
                        {predictionResult.predicted_class || predictionResult.label || 'Desconocido'}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-gray-700">Calorías estimadas:</span>
                      <span className="text-2xl font-bold text-red-600">
                        {predictionResult.calories || predictionResult.estimated_calories || 'N/A'} kcal
                      </span>
                    </div>
                    
                    {predictionResult.confidence && (
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-700">Confianza:</span>
                        <span className="text-lg font-medium text-green-600">
                          {(predictionResult.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}
                    
                    {/* Información nutricional */}
                    {predictionResult.nutrition_info && (
                      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                        <h4 className="font-medium text-gray-700 mb-2">Información Nutricional:</h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          {predictionResult.nutrition_info.protein && (
                            <div>Proteínas: <span className="font-medium">{predictionResult.nutrition_info.protein}g</span></div>
                          )}
                          {predictionResult.nutrition_info.carbs && (
                            <div>Carbohidratos: <span className="font-medium">{predictionResult.nutrition_info.carbs}g</span></div>
                          )}
                          {predictionResult.nutrition_info.fat && (
                            <div>Grasas: <span className="font-medium">{predictionResult.nutrition_info.fat}g</span></div>
                          )}
                          {predictionResult.nutrition_info.fiber && (
                            <div>Fibra: <span className="font-medium">{predictionResult.nutrition_info.fiber}g</span></div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Top predicciones */}
                    {predictionResult.top_predictions && predictionResult.top_predictions.length > 1 && (
                      <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                        <h4 className="font-medium text-gray-700 mb-2">Otras posibilidades:</h4>
                        <div className="space-y-1 text-sm">
                          {predictionResult.top_predictions.slice(1, 4).map((pred, idx) => (
                            <div key={idx} className="flex justify-between">
                              <span>{pred.class || pred.label}</span>
                              <span className="text-gray-600">{(pred.confidence * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <details className="mt-4">
                    <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
                      Ver respuesta completa del servidor
                    </summary>
                    <pre className="mt-2 text-xs text-gray-700 overflow-auto bg-gray-100 p-2 rounded">
                      {JSON.stringify(predictionResult, null, 2)}
                    </pre>
                  </details>
                </div>
              )}

              <p className="mt-6 text-xs text-gray-400 text-center">
                Formatos soportados: JPG, PNG. Tamaño máximo recomendado: 5MB.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
