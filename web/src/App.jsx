import { useState } from 'react'
import './App.css'

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

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()
      setPredictions(data.predictions)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>🌿 Plant Doctor</h1>
      <p>Upload a photo of a plant leaf to diagnose diseases</p>

      <div className="upload-section">
        <input
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleFileChange}
        />

        {imagePreview && (
          <div className="preview">
            <img src={imagePreview} alt="Preview" />
          </div>
        )}

        <button onClick={handleSubmit} disabled={!imageFile || loading}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>

      {error && <div className="error">Error: {error}</div>}

      {predictions && (
        <div className="results">
          <h2>Results</h2>
          {predictions.map((pred, idx) => (
            <div key={idx} className="prediction">
              <span className="class-name">{pred.class}</span>
              <span className="confidence">
                {(pred.confidence * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default App