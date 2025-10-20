import React, { useRef, useState, useEffect } from "react";
import hero from "../assets/hero.png";
import scanBg from "../assets/comida.jpg"; // 🔸 pon tu imagen difuminada aquí
import { predictCalories } from "../services/PredictCaloriesServices";

export default function Home() {
  const scanRef = useRef(null);
  const fileInputRef = useRef(null);

  // UI state for upload / prediction
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null); // preview image URL
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const scrollToScan = (e) => {
    e?.preventDefault();
    if (!scanRef.current) return;

    // Ajusta si tu nav tiene otra altura
    const NAV_OFFSET = 72;
    const top = scanRef.current.getBoundingClientRect().top + window.pageYOffset;
    window.scrollTo({
      top: Math.max(top - NAV_OFFSET, 0),
      behavior: "smooth",
    });
  };

  // helper to set file + preview and cleanup previous preview
  const setFileAndPreview = (file) => {
    // revoke previous
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

  // open native file dialog
  // const triggerFileSelect = () => {
  //   fileInputRef.current?.click();
  // };

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
    // show visual cue
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

      // adapta según la respuesta del backend (label, calories, probabilities, etc.)
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
                Best food <span className="block">calories app</span>
              </h1>
              <p className="mt-6 text-lg text-black/90">
                Discover our calorie prediction app
              </p>

              <div className="mt-8 flex items-center justify-center gap-4">
                {/* smooth scroll to scan section */}
                <button
                  onClick={scrollToScan}
                  className="px-6 py-3 rounded-full bg-red-600 text-white font-medium shadow-lg hover:bg-red-700 transition"
                >
                  Scan your food
                </button>

                <a
                  href="#browse"
                  className="px-6 py-3 rounded-full bg-white/90 text-gray-800 font-medium shadow hover:brightness-95 transition"
                >
                  Explore Menu
                </a>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* BROWSE CARDS */}
      <main className="container mx-auto px-6 lg:px-20 -mt-24 relative z-10">
        <section id="browse" className="bg-white rounded-xl shadow-lg py-12 px-6">
          <h2 className="text-3xl font-serif text-center mb-10">Browse Our app</h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Card 1 */}
            <article className="p-6 border rounded-lg bg-white">
              <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                {/* icon */}
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 8c3.866 0 7 1.79 7 4v4a1 1 0 01-1 1H6a1 1 0 01-1-1v-4c0-2.21 3.134-4 7-4z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 8V4" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Comidas Saludables</h3>
              <p className="text-sm text-gray-500 mb-4">In the new era of technology we look in the future with certainty and pride for our life.</p>
              <a className="text-sm text-red-600 font-medium" href="#explore">Explore Menu</a>
            </article>

            {/* Card 2 */}
            <article className="p-6 border rounded-lg bg-white">
              <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 10h16M4 14h16" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Comidas Populares</h3>
              <p className="text-sm text-gray-500 mb-4">In the new era of technology we look in the future with certainty and pride for our life.</p>
              <a className="text-sm text-red-600 font-medium" href="#explore">Explore Menu</a>
            </article>

            {/* Card 3 */}
            <article className="p-6 border rounded-lg bg-white">
              <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M12 2l3 7h-6l3-7zM5 13h14l-1 9H6l-1-9z" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Comidas Especiales</h3>
              <p className="text-sm text-gray-500 mb-4">In the new era of technology we look in the future with certainty and pride for our life.</p>
              <a className="text-sm text-red-600 font-medium" href="#explore">Explore Menu</a>
            </article>

            {/* Card 4 */}
            <article className="p-6 border rounded-lg bg-white">
              <div className="w-14 h-14 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-600" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M3 7h18M5 7v14h14V7" />
                </svg>
              </div>
              <h3 className="font-semibold text-lg mb-2">Comida internacionales</h3>
              <p className="text-sm text-gray-500 mb-4">In the new era of technology we look in the future with certainty and pride for our life.</p>
              <a className="text-sm text-red-600 font-medium" href="#explore">Explore Menu</a>
            </article>
          </div>
        </section>

        {/* 🔸 NUEVA SECCIÓN: Descubre tu plato de comida */}
        <section
          id="scan"
          ref={scanRef}
          className="relative py-20 mt-20 rounded-2xl overflow-hidden"
        >
          {/* Imagen de fondo difuminada solo en esta sección */}
          <div className="absolute inset-0">
            <img
              src={scanBg}
              alt="background blur"
              className="w-full h-full object-cover filter blur-sm brightness-75"
            />
            <div className="absolute inset-0 bg-white/60"></div>
          </div>

          {/* Contenido principal */}
          <div className="relative z-10 max-w-3xl mx-auto text-center px-6">
            <h2 className="text-3xl md:text-4xl font-serif text-gray-800 mb-4">
              Descubre tu plato de comida
            </h2>
            <p className="text-gray-600 mb-8">
              Sube una imagen de tu plato y nuestra IA predecirá su tipo y
              calorías con precisión asombrosa.
            </p>

            {/* Card de carga: acepta drag & drop y muestra preview */}
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
                    {/* Clear (X) button */}
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
                        d="M7 16V8a1 1 0 011-1h3.28a1 1 0 01.77.36l.94 1.17A2 2 0 0014 9h3a1 1 0 011 1v5a2 2 0 01-2 2H9a2 2 0 01-2-2z"
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
                  <p className="text-sm text-gray-600">Selected: {selectedFile.name}</p>
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
                  <p className="mt-2 text-sm text-gray-500">Processing...</p>
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
                  {predicting ? "Predicting..." : "Predecir Plato"}
                </button>
              </div>

              {/* Resultado de la predicción (si lo hay) */}
              {predictionResult && (
                <div className="mt-6 text-left bg-white p-4 rounded-md border">
                  <h3 className="font-semibold mb-2">Resultado</h3>
                  <pre className="text-xs text-gray-700 overflow-auto">
                    {JSON.stringify(predictionResult, null, 2)}
                  </pre>
                </div>
              )}

              <p className="mt-6 text-xs text-gray-400 text-center">
                Aceptamos archivos JPG, PNG. Tamaño máximo recomendado 5MB.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
