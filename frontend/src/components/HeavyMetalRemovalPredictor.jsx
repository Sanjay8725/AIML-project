import { useMemo, useState } from "react";

const ADSORBENT_OPTIONS = [
  "Activated Carbon",
  "Biochar",
  "Chitosan composite",
];

const METAL_OPTIONS = ["Pb", "Cd", "Hg", "As", "Cr", "Cu", "Ni", "Zn"];

const INITIAL_FORM = {
  Adsorbent: "Activated Carbon",
  Metal: "Pb",
  "Dosage (g/L)": "",
  "Temp (°C)": "",
  pH: "",
  "Time (min)": "",
  RPM: "",
  "C0 (mg/L)": "",
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

function toNumber(value) {
  if (value === "" || value === null || value === undefined) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export default function HeavyMetalRemovalPredictor() {
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const progressValue = useMemo(() => {
    const raw = Number(prediction ?? 0);
    if (!Number.isFinite(raw)) {
      return 0;
    }
    return Math.max(0, Math.min(100, raw));
  }, [prediction]);

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setPrediction(null);

    const payload = {
      Adsorbent: formData.Adsorbent,
      Metal: formData.Metal,
      "Dosage (g/L)": toNumber(formData["Dosage (g/L)"]),
      "Temp (°C)": toNumber(formData["Temp (°C)"]),
      pH: toNumber(formData.pH),
      "Time (min)": toNumber(formData["Time (min)"]),
      RPM: toNumber(formData.RPM),
      "C0 (mg/L)": toNumber(formData["C0 (mg/L)"]),
    };

    const hasInvalid = Object.entries(payload)
      .filter(([key]) => key !== "Adsorbent" && key !== "Metal")
      .some(([, value]) => value === null);

    if (hasInvalid) {
      setError("Please enter valid numeric values for all process parameters.");
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = await response.json();
      const predictedValue =
        data.predicted_removal_percentage ??
        data["predicted_removal_percentage (%)"] ??
        data.removal_percentage ??
        data.prediction;

      if (!Number.isFinite(Number(predictedValue))) {
        throw new Error("API response did not include a numeric prediction.");
      }

      setPrediction(Number(predictedValue));
    } catch (submitError) {
      setError(submitError.message || "Unable to fetch prediction.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section className="min-h-screen px-4 py-10 text-slate-900">
      <div className="mx-auto max-w-6xl">
        <header className="mb-8 rounded-2xl border border-slate-300 bg-gradient-to-r from-slate-200 via-slate-100 to-emerald-100 p-6 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-600">
            Linear Regression Dashboard
          </p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900 md:text-4xl">
            Heavy Metal Removal Predictor
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-700 md:text-base">
            Enter adsorption experiment parameters to estimate removal efficiency.
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-3">
          <form
            onSubmit={handleSubmit}
            className="lg:col-span-2 rounded-2xl border border-slate-300 bg-white p-6 shadow-sm"
          >
            <div className="mb-4">
              <h2 className="text-lg font-semibold text-slate-900">Input Parameters</h2>
              <p className="text-sm text-slate-600">
                Provide all 8 model features from the adsorption dataset.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <label className="flex flex-col gap-1">
                <span className="text-sm font-medium text-slate-700">Adsorbent</span>
                <select
                  name="Adsorbent"
                  value={formData.Adsorbent}
                  onChange={handleInputChange}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-emerald-200 transition focus:ring-2"
                >
                  {ADSORBENT_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex flex-col gap-1">
                <span className="text-sm font-medium text-slate-700">Metal</span>
                <select
                  name="Metal"
                  value={formData.Metal}
                  onChange={handleInputChange}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-emerald-200 transition focus:ring-2"
                >
                  {METAL_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex flex-col gap-1">
                <span className="text-sm font-medium text-slate-700">Dosage (g/L)</span>
                <input
                  type="number"
                  step="any"
                  name="Dosage (g/L)"
                  value={formData["Dosage (g/L)"]}
                  onChange={handleInputChange}
                  placeholder="e.g. 1.5"
                  className="rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-emerald-200 transition focus:ring-2"
                />
              </label>

              <label className="flex flex-col gap-1">
                <span className="text-sm font-medium text-slate-700">Temp (°C)</span>
                <input
                  type="number"
                  step="any"
                  name="Temp (°C)"
                  value={formData["Temp (°C)"]}
                  onChange={handleInputChange}
                  placeholder="e.g. 30"
                  className="rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-emerald-200 transition focus:ring-2"
                />
              </label>

              <label className="flex flex-col gap-1">
                <span className="text-sm font-medium text-slate-700">pH</span>
                <input
                  type="number"
                  step="any"
                  name="pH"
                  value={formData.pH}
                  onChange={handleInputChange}
                  placeholder="e.g. 6"
                  className="rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-emerald-200 transition focus:ring-2"
                />
              </label>

              <label className="flex flex-col gap-1">
                <span className="text-sm font-medium text-slate-700">Time (min)</span>
                <input
                  type="number"
                  step="any"
                  name="Time (min)"
                  value={formData["Time (min)"]}
                  onChange={handleInputChange}
                  placeholder="e.g. 90"
                  className="rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-emerald-200 transition focus:ring-2"
                />
              </label>

              <label className="flex flex-col gap-1">
                <span className="text-sm font-medium text-slate-700">RPM</span>
                <input
                  type="number"
                  step="any"
                  name="RPM"
                  value={formData.RPM}
                  onChange={handleInputChange}
                  placeholder="e.g. 150"
                  className="rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-emerald-200 transition focus:ring-2"
                />
              </label>

              <label className="flex flex-col gap-1">
                <span className="text-sm font-medium text-slate-700">C0 (mg/L)</span>
                <input
                  type="number"
                  step="any"
                  name="C0 (mg/L)"
                  value={formData["C0 (mg/L)"]}
                  onChange={handleInputChange}
                  placeholder="e.g. 50"
                  className="rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-emerald-200 transition focus:ring-2"
                />
              </label>
            </div>

            <div className="mt-6 flex items-center gap-3">
              <button
                type="submit"
                disabled={isLoading}
                className="rounded-lg bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isLoading ? "Predicting..." : "Predict Removal %"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setFormData(INITIAL_FORM);
                  setPrediction(null);
                  setError("");
                }}
                className="rounded-lg border border-slate-300 bg-white px-5 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
              >
                Reset
              </button>
            </div>

            {error ? (
              <p className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
                {error}
              </p>
            ) : null}
          </form>

          <aside className="rounded-2xl border border-emerald-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900">Prediction Output</h2>
            <p className="mt-1 text-sm text-slate-600">
              Estimated removal performance from the trained model.
            </p>

            <div className="mt-6 rounded-xl border border-emerald-200 bg-emerald-50 p-5">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-700">
                Success
              </p>
              <p className="mt-2 text-4xl font-bold text-emerald-800">
                {prediction !== null ? `${progressValue.toFixed(1)}%` : "--"}
              </p>
              <p className="mt-1 text-sm text-emerald-700">Predicted Removal Percentage</p>

              <div className="mt-5">
                <div className="mb-2 flex items-center justify-between text-xs text-slate-600">
                  <span>0%</span>
                  <span>100%</span>
                </div>
                <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-emerald-700 transition-all duration-700"
                    style={{ width: `${progressValue}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="mt-5 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-600">
              API endpoint: {API_BASE_URL}/predict
            </div>
          </aside>
        </div>
      </div>
    </section>
  );
}
