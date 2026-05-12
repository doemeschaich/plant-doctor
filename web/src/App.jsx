import { useState } from 'react'

const API_URL = 'http://localhost:8000'

function App() {
  const [imageFile, setImageFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileChange = (event) => {
    const file = event.target.files[0]
    if (!file) return
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setPredictions(null)
    setError(null)
  }

  const handleSubmit = async () => {
    if (!imageFile) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', imageFile)
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) throw new Error(`API error: ${response.status}`)
      const data = await response.json()
      setPredictions(data.predictions)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // "Tomato___Late_blight" -> "Tomato — Late blight"
  const prettyClass = (cls) =>
    cls.replace(/___/g, ' — ').replace(/_/g, ' ')

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 py-8 px-4">
      <div className="max-w-md mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-emerald-800 mb-2">
            🌿 Plant Doctor
          </h1>
          <p className="text-emerald-700">
            Upload a leaf photo to diagnose diseases
          </p>
        </header>

        <div className="bg-white rounded-2xl shadow-xl p-6">
          <input
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileChange}
            className="block w-full text-sm text-slate-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-emerald-100 file:text-emerald-700
              hover:file:bg-emerald-200
              cursor-pointer"
          />

          {imagePreview && (
            <img
              src={imagePreview}
              alt="Preview"
              className="mt-4 w-full rounded-xl object-cover max-h-80"
            />
          )}

          <button
            onClick={handleSubmit}
            disabled={!imageFile || loading}
            className="mt-4 w-full py-3 px-6 rounded-xl
              bg-emerald-600 hover:bg-emerald-700
              disabled:bg-slate-300 disabled:cursor-not-allowed
              text-white font-semibold transition-colors"
          >
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>

        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl">
            Error: {error}
          </div>
        )}

        {predictions && (
          <div className="mt-6 bg-white rounded-2xl shadow-xl p-6">
            <h2 className="text-xl font-bold text-slate-800 mb-4">
              Results
            </h2>
            <div className="space-y-3">
              {predictions.map((pred, idx) => (
                <div
                  key={idx}
                  className={`flex items-center justify-between p-3 rounded-xl ${
                    idx === 0
                      ? 'bg-emerald-50 border-2 border-emerald-300'
                      : 'bg-slate-50'
                  }`}
                >
                  <span className="text-sm font-medium text-slate-700">
                    {prettyClass(pred.class)}
                  </span>
                  <span
                    className={`text-sm font-bold ${
                      idx === 0 ? 'text-emerald-700' : 'text-slate-500'
                    }`}
                  >
                    {(pred.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App