import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/predict'

const CLASS_COPY = {
  potential_repair_needed: {
    tone: 'attention',
    label: 'Potential repair needed',
    heading: 'Hmm… this roof might need some love.',
    body: 'Something caught the AI’s attention. Might be worth taking a closer look.',
  },
  no_obvious_repair_needed: {
    tone: 'clear',
    label: 'No obvious repair needed',
    heading: 'Looks pretty good!',
    body: 'Nothing obvious jumped out. That’s a good sign!',
  },
}

const PROBABILITY_ORDER = ['potential_repair_needed', 'no_obvious_repair_needed']

function classLabel(key) {
  return CLASS_COPY[key]?.label || 'Other reading'
}

function friendlyError(response, payload) {
  const detail = payload?.detail
  const message = typeof detail === 'string'
    ? detail
    : Array.isArray(detail)
      ? detail.map((item) => item?.msg).filter(Boolean).join(' ')
      : ''

  if (response?.status === 503 || /model\.pkl/i.test(message)) {
    return 'Roof screening is temporarily unavailable. Please try again later.'
  }
  if (response?.status === 400) {
    return message || 'Please choose a photo of a roof and try again.'
  }
  if (/^inference failed/i.test(message)) {
    return "We couldn't read this photo. Try another image with a clearer view of the roof."
  }
  if (message) return message
  return "We couldn't analyze this photo. Please try again."
}

function App() {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  useEffect(() => {
    if (!file) {
      setPreviewUrl('')
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const chooseFile = (selected) => {
    if (!selected) return
    if (!selected.type.startsWith('image/')) {
      setError('Please choose a JPEG, PNG, WebP, or GIF photo.')
      return
    }
    setError('')
    setResult(null)
    setFile(selected)
  }

  const onFileChange = (event) => {
    chooseFile(event.target.files?.[0])
  }

  const onDragOver = (event) => {
    event.preventDefault()
    if (!loading) setDragging(true)
  }

  const onDragLeave = (event) => {
    event.preventDefault()
    setDragging(false)
  }

  const onDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    if (loading) return
    chooseFile(event.dataTransfer.files?.[0])
  }

  const onPredict = async () => {
    if (!file || loading) return

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      })

      let payload = null
      try {
        payload = await response.json()
      } catch {
        payload = null
      }

      if (!response.ok) {
        throw new Error(friendlyError(response, payload))
      }

      setResult(payload)
    } catch (err) {
      const offline = err?.message === 'Failed to fetch'
      setError(offline
        ? "We couldn't reach Roof Inspector. Check your connection and try again."
        : (err.message || "We couldn't analyze this photo. Please try again."))
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setResult(null)
    setError('')
    setDragging(false)
    if (inputRef.current) inputRef.current.value = ''
  }

  const copy = result
    ? CLASS_COPY[result.predicted_label] || {
      tone: 'unknown',
      heading: 'Here’s what the check found.',
      body: 'The screening finished. The readings below show how the two outcomes compared.',
    }
    : null
  const confidence = result?.probabilities?.[result.predicted_label]
  const probabilityRows = result?.probabilities
    ? PROBABILITY_ORDER
      .filter((key) => key in result.probabilities)
      .concat(Object.keys(result.probabilities).filter((key) => !PROBABILITY_ORDER.includes(key)))
      .map((key) => [key, result.probabilities[key]])
    : []

  return (
    <div className="page">
      <header className="header">
        <div className="brand">
          <span className="mark" aria-hidden="true">
            <svg viewBox="0 0 32 32">
              <path d="M5 15.5 16 6l11 9.5" />
              <path d="M9 15.2V25h14v-9.8" />
            </svg>
          </span>
          <span className="brandName">Roof Inspector</span>
        </div>
      </header>

      <main className="main">
        <p className="kicker">Let our AI tell you if your roof needs a repair.</p>
        <h1>Snap a photo, upload it, and see what the AI thinks.</h1>

        <section className="panel" aria-label="Roof photo screening">
          <label
            className={`dropzone${dragging ? ' isDragging' : ''}${previewUrl ? ' hasPreview' : ''}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={onFileChange}
              disabled={loading}
            />
            {previewUrl ? (
              <img src={previewUrl} alt="Selected roof" className="preview" />
            ) : (
              <span className="dropCopy">
                <span className="dropArt" aria-hidden="true">
                  <svg viewBox="0 0 128 80">
                    <circle className="sun" cx="104" cy="18" r="7" />
                    <path className="roof" d="M16 40 64 14l48 26" />
                    <path className="house" d="M30 40v28h68V40" />
                    <path className="door" d="M58 68V50h14v18" />
                  </svg>
                </span>
                <span className="dropTitle">Drop a roof photo here</span>
                <span className="dropOr">or choose an image</span>
                <span className="formats">JPEG, PNG, WebP, or GIF</span>
              </span>
            )}
          </label>

          {file && previewUrl && !result && (
            <p className="fileName">{file.name}</p>
          )}

          {!result && (
            <button
              type="button"
              className="analyze"
              onClick={onPredict}
              disabled={loading || !file}
            >
              {loading ? 'Analyzing...' : 'Inspect This Roof →'}
            </button>
          )}

          {error && (
            <p className="error" role="alert">{error}</p>
          )}

          {result && copy && (
            <article className={`result result-${copy.tone}`} aria-live="polite">
              <div className="resultTop">
                <span className="resultMark" aria-hidden="true">
                  {copy.tone === 'clear' ? (
                    <svg viewBox="0 0 24 24"><path d="M6 12.5 10 16.5 18 8" /></svg>
                  ) : (
                    <svg viewBox="0 0 24 24"><path d="M5 13.5 12 7l7 6.5" /><path d="M8 13.2V19h8v-5.8" /></svg>
                  )}
                </span>
                <div>
                  <h2>{copy.heading}</h2>
                  {typeof confidence === 'number' && (
                    <p className="confidence">{(confidence * 100).toFixed(1)}% confidence</p>
                  )}
                </div>
              </div>
              <p className="resultBody">{copy.body}</p>
              {probabilityRows.length > 0 && (
                <dl className="probs">
                  {probabilityRows.map(([key, probability]) => (
                    <div className="probRow" key={key}>
                      <dt>{classLabel(key)}</dt>
                      <dd>{(probability * 100).toFixed(1)}%</dd>
                    </div>
                  ))}
                </dl>
              )}
              <p className="disclaimer">
                Roof Inspector is an AI screening tool, not a professional roof inspection. Results may be affected by image quality, lighting, roof angle, and damage that is not visible in the photo.
              </p>
              <button type="button" className="another" onClick={reset}>
                Analyze another photo
              </button>
            </article>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
