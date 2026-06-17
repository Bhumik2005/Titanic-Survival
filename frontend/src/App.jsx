import { useState } from "react"
import Navbar from "./components/Navbar"
import Form from "./components/Form"
import Result from "./components/Result"

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePredict = async (formData) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      })

      if (!response.ok) throw new Error("Prediction failed")

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError("Could not connect to the backend. Make sure it's running.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white">
      <Navbar />
      <main className="max-w-5xl mx-auto px-6 py-12">
        {/* Hero */}
        <div className="text-center mb-14">
          <p className="text-xs uppercase tracking-[0.3em] text-[#6B7280] mb-4 font-mono">
            Machine Learning · 1912
          </p>
          <h1 className="text-5xl font-bold text-white mb-4 leading-tight">
            Would you have<br />
            <span className="text-[#60A5FA]">survived?</span>
          </h1>
          <p className="text-[#9CA3AF] text-lg max-w-xl mx-auto">
            Enter your passenger details and let the model predict your fate aboard the RMS Titanic.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <Form onPredict={handlePredict} loading={loading} />
          <Result result={result} error={error} loading={loading} />
        </div>
      </main>
    </div>
  )
}
