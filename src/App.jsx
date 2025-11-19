import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, Brain, Sparkles, X } from 'lucide-react';

const EmosiKU = () => {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [model, setModel] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);

  const emotions = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise'];

  // Load TensorFlow.js model
  const loadModel = async () => {
    try {
      setIsLoading(true);
      
      // Load TensorFlow.js library
      if (!window.tf) {
        await new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0';
          script.onload = resolve;
          script.onerror = reject;
          document.head.appendChild(script);
        });
      }
      
      // GUNAKAN JSDELIVR CDN - Support CORS dan lebih cepat!
      // Format: https://cdn.jsdelivr.net/gh/username/repo@branch/path/to/file
      const modelUrl = 'https://cdn.jsdelivr.net/gh/erenmenn/EmosiKu-model@main/model.json';
      
      console.log('Loading model from jsDelivr CDN:', modelUrl);
      
      const loadedModel = await window.tf.loadLayersModel(modelUrl);
      
      setModel(loadedModel);
      setModelLoaded(true);
      setIsLoading(false);
      console.log('✅ Model loaded successfully!');
      console.log('Model input shape:', loadedModel.inputs[0].shape);
      console.log('Model output shape:', loadedModel.outputs[0].shape);
    } catch (error) {
      console.error('❌ Error loading model:', error);
      console.error('Error details:', error.message);
      
      // Tampilkan error yang lebih informatif
      let errorMsg = 'Gagal memuat model.\n\n';
      if (error.message.includes('404')) {
        errorMsg += '❌ File tidak ditemukan.\n\nPastikan:\n1. Repository PUBLIC\n2. File model.json ada di root folder\n3. File weights (.bin) ada di folder yang sama';
      } else if (error.message.includes('CORS') || error.message.includes('fetch')) {
        errorMsg += '❌ CORS Error.\n\nSolusi: Gunakan jsDelivr CDN (sudah diupdate di kode)';
      } else {
        errorMsg += error.message;
      }
      
      alert(errorMsg);
      setIsLoading(false);
      setModelLoaded(false);
    }
  };

  useEffect(() => {
  Promise.resolve().then(() => loadModel());
}, []);

  const convertToGrayscale = (imgElement) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Resize to model input size (biasanya 48x48 atau 224x224)
    const targetSize = 48; // Sesuaikan dengan input size model Anda
    canvas.width = targetSize;
    canvas.height = targetSize;
    
    ctx.drawImage(imgElement, 0, 0, targetSize, targetSize);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Convert to grayscale
    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      data[i] = avg;
      data[i + 1] = avg;
      data[i + 2] = avg;
    }
    
    ctx.putImageData(imageData, 0, 0);
    return canvas;
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        setPrediction(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const predictEmotion = async () => {
    if (!image) {
      alert('Silakan upload gambar terlebih dahulu!');
      return;
    }
    
    if (!model) {
      alert('Model belum siap. Tunggu hingga model selesai dimuat.');
      return;
    }
    
    setIsLoading(true);
    console.log('🔍 Starting prediction...');
    
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = async () => {
        try {
          console.log('📷 Image loaded, size:', img.width, 'x', img.height);
          
          // Konversi ke grayscale dan resize
          const grayscaleCanvas = convertToGrayscale(img);
          console.log('⚫ Converted to grayscale:', grayscaleCanvas.width, 'x', grayscaleCanvas.height);
          
          // Prediksi menggunakan model
          const tensor = window.tf.browser.fromPixels(grayscaleCanvas, 1) // 1 channel = grayscale
            .toFloat()
            .div(255.0) // Normalisasi 0-1
            .expandDims(0); // Tambah batch dimension [1, 48, 48, 1]
          
          console.log('📊 Tensor shape:', tensor.shape);
          
          const predictionTensor = await model.predict(tensor);
          const predictions = await predictionTensor.data();
          
          console.log('🎯 Raw predictions:', predictions);
          
          // Cleanup memory
          tensor.dispose();
          predictionTensor.dispose();
          
          // Map hasil prediksi ke emotions
          const results = emotions.map((emotion, idx) => ({
            emotion,
            confidence: predictions[idx] * 100
          })).sort((a, b) => b.confidence - a.confidence);
          
          console.log('✅ Final results:', results);
          
          setPrediction(results);
          setIsLoading(false);
        } catch (predError) {
          console.error('❌ Prediction error:', predError);
          alert('Error saat prediksi: ' + predError.message);
          setIsLoading(false);
        }
      };
      
      img.onerror = () => {
        console.error('❌ Error loading image');
        alert('Gagal memuat gambar');
        setIsLoading(false);
      };
      
      img.src = image;
    } catch (error) {
      console.error('❌ Error in predictEmotion:', error);
      alert('Terjadi kesalahan: ' + error.message);
      setIsLoading(false);
    }
  };

  const resetAll = () => {
    setImage(null);
    setPrediction(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black text-white p-4 md:p-8">
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Header */}
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8 md:mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="w-10 h-10 md:w-12 md:h-12 text-yellow-500" />
            <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-yellow-400 via-yellow-500 to-yellow-600 bg-clip-text text-transparent">
              EmosiKU.id
            </h1>
            <Sparkles className="w-10 h-10 md:w-12 md:h-12 text-yellow-500" />
          </div>
          <p className="text-gray-400 text-base md:text-lg">
            Deteksi Emosi dengan Kecerdasan Buatan
          </p>
          {modelLoaded ? (
            <div className="mt-3 inline-flex items-center gap-2 px-4 py-2 bg-green-500/20 border border-green-500/30 rounded-full text-green-400 text-sm">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              Model Siap Digunakan
            </div>
          ) : (
            <div className="mt-3 inline-flex items-center gap-2 px-4 py-2 bg-yellow-500/20 border border-yellow-500/30 rounded-full text-yellow-400 text-sm">
              <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
              Memuat Model dari GitHub...
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="max-w-2xl mx-auto mb-8">
          {/* Upload Section */}
          <div className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-yellow-500/20 rounded-2xl p-6 md:p-8 shadow-2xl">
            <h2 className="text-xl md:text-2xl font-bold text-yellow-500 mb-6 flex items-center gap-2">
              <Upload className="w-6 h-6" />
              Upload Gambar Wajah
            </h2>
            
            {!image ? (
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-3 border-dashed border-yellow-500/40 rounded-xl p-12 md:p-16 text-center cursor-pointer hover:border-yellow-500/70 hover:bg-yellow-500/5 transition-all duration-300"
              >
                <Camera className="w-20 h-20 md:w-24 md:h-24 mx-auto mb-4 text-yellow-500" />
                <p className="text-gray-300 text-lg md:text-xl mb-2">Klik untuk memilih gambar</p>
                <p className="text-gray-500 text-sm">Format: JPG, PNG (Otomatis dikonversi ke grayscale)</p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-xl overflow-hidden border-2 border-yellow-500/30">
                  <img src={image} alt="Uploaded" className="w-full h-80 object-cover" />
                  <div className="absolute top-2 left-2 bg-black/70 px-3 py-1 rounded-full text-xs text-yellow-500 border border-yellow-500/30">
                    Gambar Anda
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={predictEmotion}
                    disabled={isLoading || !modelLoaded}
                    className="py-4 bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700 rounded-xl font-bold text-black transition-all duration-300 shadow-lg shadow-yellow-500/30 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {isLoading ? (
                      <>
                        <div className="w-5 h-5 border-3 border-black border-t-transparent rounded-full animate-spin"></div>
                        Menganalisis...
                      </>
                    ) : (
                      <>
                        <Brain className="w-5 h-5" />
                        Deteksi Emosi
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={resetAll}
                    className="py-4 bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 rounded-xl text-red-400 font-semibold transition-all duration-300 flex items-center justify-center gap-2"
                  >
                    <X className="w-5 h-5" />
                    Hapus
                  </button>
                </div>
              </div>
            )}
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
          </div>
        </div>

        {/* Results Section */}
        {prediction && (
          <div className="max-w-2xl mx-auto bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm border border-yellow-500/20 rounded-2xl p-6 md:p-8 shadow-2xl animate-fadeIn">
            <h2 className="text-2xl md:text-3xl font-bold text-yellow-500 mb-6 flex items-center gap-2">
              <Sparkles className="w-7 h-7" />
              Hasil Deteksi Emosi
            </h2>
            
            {/* Top Emotion - Highlighted */}
            <div className="mb-6 bg-gradient-to-r from-yellow-500/20 to-yellow-600/20 border-2 border-yellow-500/50 rounded-xl p-6">
              <div className="text-center">
                <p className="text-gray-400 text-sm mb-2">Emosi Terdeteksi</p>
                <h3 className="text-4xl md:text-5xl font-bold text-yellow-400 mb-2">
                  {prediction[0].emotion}
                </h3>
                <p className="text-3xl md:text-4xl font-bold text-yellow-500">
                  {prediction[0].confidence.toFixed(1)}%
                </p>
              </div>
            </div>
            
            {/* All Emotions */}
            <div className="space-y-3">
              <p className="text-gray-400 text-sm mb-3">Detail Semua Emosi:</p>
              {prediction.map((result, idx) => (
                <div key={idx} className="bg-black/30 rounded-xl p-4 border border-yellow-500/10 hover:border-yellow-500/30 transition-all duration-300">
                  <div className="flex justify-between items-center mb-2">
                    <span className={`font-semibold text-lg ${idx === 0 ? 'text-yellow-400' : 'text-gray-300'}`}>
                      {result.emotion}
                    </span>
                    <span className={`font-bold ${idx === 0 ? 'text-yellow-400' : 'text-gray-400'}`}>
                      {result.confidence.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        idx === 0 
                          ? 'bg-gradient-to-r from-yellow-400 to-yellow-600' 
                          : 'bg-gradient-to-r from-gray-500 to-gray-600'
                      }`}
                      style={{ width: `${result.confidence}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Instructions */}
        <div className="mt-8 max-w-4xl mx-auto bg-gradient-to-br from-yellow-500/10 to-yellow-600/10 backdrop-blur-sm border border-yellow-500/30 rounded-2xl p-6 md:p-8">
          <h3 className="text-xl font-bold text-yellow-500 mb-4">📋 Cara Menggunakan</h3>
          <ol className="space-y-2 text-gray-300 mb-6">
            <li className="flex gap-2"><span className="text-yellow-500 font-bold">1.</span> Upload gambar wajah yang ingin dianalisis</li>
            <li className="flex gap-2"><span className="text-yellow-500 font-bold">2.</span> Gambar otomatis dikonversi ke grayscale saat prediksi</li>
            <li className="flex gap-2"><span className="text-yellow-500 font-bold">3.</span> Klik tombol "Deteksi Emosi" untuk menganalisis</li>
            <li className="flex gap-2"><span className="text-yellow-500 font-bold">4.</span> Lihat hasil emosi dengan persentase confidence</li>
          </ol>
          
          <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg mb-4">
            <p className="text-sm text-green-400 font-semibold mb-2">
              ✅ Model Configuration:
            </p>
            <div className="space-y-1 text-xs text-gray-300">
              <p>• <strong>Source:</strong> jsDelivr CDN (CORS-enabled)</p>
              <p>• <strong>URL:</strong> <code className="bg-black/50 px-2 py-1 rounded text-green-400">cdn.jsdelivr.net/gh/erenmenn/EmosiKu-model</code></p>
              <p>• <strong>Emotions:</strong> Angry, Fear, Happy, Sad, Surprise</p>
              <p>• <strong>Input:</strong> 48x48 Grayscale Image</p>
            </div>
          </div>
          
          <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <p className="text-sm text-blue-400 mb-2">
              <strong>🔧 Troubleshooting:</strong>
            </p>
            <div className="space-y-1 text-xs text-gray-300">
              <p>• Pastikan repository GitHub <strong>PUBLIC</strong></p>
              <p>• File model.json dan weights (.bin) harus di root folder</p>
              <p>• Jika error, cek Console browser (F12) untuk detail</p>
              <p>• Tunggu beberapa detik untuk model loading (pertama kali)</p>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
      `}</style>
    </div>
  );
};

export default EmosiKU;