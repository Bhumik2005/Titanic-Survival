export default function Result({ result, error, loading }) {
  if (loading) {
    return (
      <div className="bg-[#0f1117] border border-[#1F2937] rounded-2xl p-6 flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-[#60A5FA] border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-[#6B7280] text-sm">Running prediction...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-[#0f1117] border border-[#1F2937] rounded-2xl p-6 flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <p className="text-4xl mb-4">⚠️</p>
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="bg-[#0f1117] border border-[#1F2937] rounded-2xl p-6 flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <p className="text-5xl mb-4">🌊</p>
          <p className="text-[#6B7280] text-sm">
            Fill in the passenger details and click<br />
            <span className="text-[#60A5FA]">Predict My Fate</span> to see the result.
          </p>
        </div>
      </div>
    )
  }

  const survived = result.survived
  const probability = result.probability

  return (
    <div className="bg-[#0f1117] border border-[#1F2937] rounded-2xl p-6 space-y-6">
      <h2 className="text-white font-semibold text-lg">Prediction Result</h2>

      {/* Main verdict */}
      <div
        className={`rounded-xl p-6 text-center border ${
          survived
            ? "bg-green-950/40 border-green-700/40"
            : "bg-red-950/40 border-red-700/40"
        }`}
      >
        <p className="text-5xl mb-3">{survived ? "🛟" : "💀"}</p>
        <p
          className={`text-2xl font-bold mb-1 ${
            survived ? "text-green-400" : "text-red-400"
          }`}
        >
          {survived ? "Survived" : "Did Not Survive"}
        </p>
        <p className="text-[#9CA3AF] text-sm">
          Model: {result.model_used === "logistic" ? "Logistic Regression" : "Random Forest"}
        </p>
      </div>

      {/* Probability bar */}
      <div>
        <div className="flex justify-between text-xs text-[#6B7280] mb-2">
          <span>Survival Probability</span>
          <span className="text-white font-medium">{probability}%</span>
        </div>
        <div className="h-2.5 bg-[#1F2937] rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              survived ? "bg-green-400" : "bg-red-400"
            }`}
            style={{ width: `${probability}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-[#6B7280] mt-1">
          <span>0%</span>
          <span>100%</span>
        </div>
      </div>

      {/* Feature importance (Random Forest only) */}
      {result.feature_importance && (
        <div>
          <p className="text-xs font-medium text-[#9CA3AF] uppercase tracking-wider mb-3">
            Feature Importance
          </p>
          <div className="space-y-2.5">
            {result.feature_importance.map((f) => (
              <div key={f.feature}>
                <div className="flex justify-between text-xs text-[#9CA3AF] mb-1">
                  <span>{f.feature}</span>
                  <span>{(f.importance * 100).toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-[#1F2937] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#60A5FA] rounded-full"
                    style={{ width: `${f.importance * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Fun fact */}
      <div className="border-t border-[#1F2937] pt-4">
        <p className="text-xs text-[#6B7280]">
          {survived
            ? "✓ You would have made it onto a lifeboat."
            : "✗ You would have gone down with the ship."}
        </p>
      </div>
    </div>
  )
}
