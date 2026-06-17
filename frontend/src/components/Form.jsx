import { useState } from "react"

const inputClass =
  "w-full bg-[#111827] border border-[#1F2937] rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none focus:border-[#60A5FA] transition-colors"

const labelClass = "block text-xs font-medium text-[#9CA3AF] uppercase tracking-wider mb-1.5"

export default function Form({ onPredict, loading }) {
  const [form, setForm] = useState({
    pclass: 1,
    sex: "male",
    age: 25,
    sibsp: 0,
    parch: 0,
    fare: 50,
    embarked: "S",
    model: "logistic",
  })

  const handle = (e) => {
    const { name, value } = e.target
    setForm((prev) => ({
      ...prev,
      [name]: ["age", "fare", "sibsp", "parch", "pclass"].includes(name)
        ? Number(value)
        : value,
    }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    onPredict(form)
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-[#0f1117] border border-[#1F2937] rounded-2xl p-6 space-y-5"
    >
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-white font-semibold text-lg">Passenger Details</h2>
        <span className="text-xs text-[#6B7280] font-mono">RMS Titanic · 1912</span>
      </div>

      {/* Model selector */}
      <div>
        <label className={labelClass}>Model</label>
        <div className="flex gap-3">
          {[
            { value: "logistic", label: "Logistic Regression" },
            { value: "random_forest", label: "Random Forest" },
          ].map((m) => (
            <button
              key={m.value}
              type="button"
              onClick={() => setForm((p) => ({ ...p, model: m.value }))}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors border ${
                form.model === m.value
                  ? "bg-[#60A5FA] text-[#0a0a0f] border-[#60A5FA]"
                  : "bg-transparent text-[#9CA3AF] border-[#1F2937] hover:border-[#60A5FA]"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Passenger Class */}
      <div>
        <label className={labelClass}>Passenger Class</label>
        <div className="flex gap-3">
          {[1, 2, 3].map((c) => (
            <button
              key={c}
              type="button"
              onClick={() => setForm((p) => ({ ...p, pclass: c }))}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors border ${
                form.pclass === c
                  ? "bg-[#60A5FA] text-[#0a0a0f] border-[#60A5FA]"
                  : "bg-transparent text-[#9CA3AF] border-[#1F2937] hover:border-[#60A5FA]"
              }`}
            >
              {c === 1 ? "1st" : c === 2 ? "2nd" : "3rd"}
            </button>
          ))}
        </div>
      </div>

      {/* Sex */}
      <div>
        <label className={labelClass}>Sex</label>
        <div className="flex gap-3">
          {["male", "female"].map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setForm((p) => ({ ...p, sex: s }))}
              className={`flex-1 py-2 rounded-lg text-sm font-medium capitalize transition-colors border ${
                form.sex === s
                  ? "bg-[#60A5FA] text-[#0a0a0f] border-[#60A5FA]"
                  : "bg-transparent text-[#9CA3AF] border-[#1F2937] hover:border-[#60A5FA]"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Age + Fare */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className={labelClass}>Age: {form.age}</label>
          <input
            type="range"
            name="age"
            min={1}
            max={80}
            value={form.age}
            onChange={handle}
            className="w-full accent-[#60A5FA]"
          />
        </div>
        <div>
          <label className={labelClass}>Fare: £{form.fare}</label>
          <input
            type="range"
            name="fare"
            min={0}
            max={500}
            value={form.fare}
            onChange={handle}
            className="w-full accent-[#60A5FA]"
          />
        </div>
      </div>

      {/* SibSp + Parch */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className={labelClass}>Siblings / Spouses</label>
          <input
            type="number"
            name="sibsp"
            min={0}
            max={10}
            value={form.sibsp}
            onChange={handle}
            className={inputClass}
          />
        </div>
        <div>
          <label className={labelClass}>Parents / Children</label>
          <input
            type="number"
            name="parch"
            min={0}
            max={10}
            value={form.parch}
            onChange={handle}
            className={inputClass}
          />
        </div>
      </div>

      {/* Embarked */}
      <div>
        <label className={labelClass}>Port of Embarkation</label>
        <div className="flex gap-3">
          {[
            { value: "C", label: "Cherbourg" },
            { value: "Q", label: "Queenstown" },
            { value: "S", label: "Southampton" },
          ].map((e) => (
            <button
              key={e.value}
              type="button"
              onClick={() => setForm((p) => ({ ...p, embarked: e.value }))}
              className={`flex-1 py-2 rounded-lg text-xs font-medium transition-colors border ${
                form.embarked === e.value
                  ? "bg-[#60A5FA] text-[#0a0a0f] border-[#60A5FA]"
                  : "bg-transparent text-[#9CA3AF] border-[#1F2937] hover:border-[#60A5FA]"
              }`}
            >
              {e.label}
            </button>
          ))}
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full py-3 bg-[#60A5FA] hover:bg-[#3B82F6] text-[#0a0a0f] font-bold rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm tracking-wide"
      >
        {loading ? "Predicting..." : "Predict My Fate"}
      </button>
    </form>
  )
}
