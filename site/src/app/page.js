"use client";

import Image from "next/image";

const base = process.env.NODE_ENV === "production" ? "/Code2_AI" : "";



import { useMemo, useState } from "react";
import { Client } from "@gradio/client";

export default function Page() {
  const [input, setInput] = useState(
    "Suma r1 y r2 y guarda en r3\nGuarda r3 en la dirección 0345"
  );
  const [output, setOutput] = useState("");
  const [latency, setLatency] = useState(null);
  const [loading, setLoading] = useState(false);

  const [modelChoice, setModelChoice] = useState("onnx"); // "onnx" | "fp32"
  const [beams, setBeams] = useState(4);

  // 🔁 Cambia esto por tu Space real (usuario/nombre-space)
  const SPACE_ID = useMemo(() => "coyoteMMK/Code2_AI", []);

  // 🔁 Endpoint de Gradio. Normalmente es "/predict"
  // Si en "Use via API" te sale otro, cámbialo aquí.
  const ENDPOINT = useMemo(() => "/generar", []);

  async function run() {
    if (!input.trim()) return;

    setLoading(true);
    setOutput("");
    setLatency(null);

    try {
      const t0 = performance.now();

      const client = await Client.connect(SPACE_ID);

      const modelo_sel =
        modelChoice === "onnx" ? "ONNX INT8 (rápido)" : "Full FP32 (más pesado)";

      // Debe coincidir con tu función gradio:
      // generar(instruccion, modelo_sel, beams) -> [salida, tiempo]
      const result = await client.predict(ENDPOINT, [input, modelo_sel, beams]);

      const t1 = performance.now();

      const salida = String(result?.data?.[0] ?? "");
      const tiempoStr = String(result?.data?.[1] ?? "");

      setOutput(salida);

      const parsed = parseFloat(tiempoStr.replace("s", "").trim());
      if (!Number.isNaN(parsed)) setLatency(parsed);
      else setLatency((t1 - t0) / 1000);
    } catch (e) {
      setOutput("❌ Error llamando al Space:\n" + (e?.message ?? String(e)));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-5xl px-6 py-10">
        <h1 className="text-3xl font-bold tracking-tight">
          CODE-2 Translator (GitHub Pages)
        </h1>
        <p className="mt-2 text-zinc-300">
          Web estática (Next.js + Tailwind) que llama a un Space de Hugging Face
          para traducir NL → CODE-2.
        </p>

        <div className="mt-6 flex flex-wrap items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-zinc-300">Modelo</span>
            <select
              className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm"
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
            >
              <option value="onnx">ONNX INT8</option>
              <option value="fp32">Full FP32</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-zinc-300">Beams</span>
            <input
              className="w-20 rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm"
              type="number"
              min={1}
              max={8}
              value={beams}
              onChange={(e) => setBeams(parseInt(e.target.value || "4", 10))}
            />
          </div>

          <button
            onClick={run}
            disabled={loading}
            className="ml-auto rounded-lg bg-emerald-500 px-4 py-2 text-sm font-semibold text-black hover:bg-emerald-400 disabled:opacity-60"
          >
            {loading ? "Generando..." : "Traducir"}
          </button>

          {latency !== null && (
            <div className="text-sm text-zinc-200">
              ⏱ {latency.toFixed(3)} s
            </div>
          )}
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
            <div className="mb-2 text-sm text-zinc-300">Entrada (NL)</div>
            <textarea
              className="h-72 w-full resize-none rounded-lg border border-zinc-700 bg-zinc-950 p-3 text-sm outline-none"
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
            <div className="mb-2 text-sm text-zinc-300">Salida (CODE-2)</div>
            <textarea
              className="h-72 w-full resize-none rounded-lg border border-zinc-700 bg-zinc-950 p-3 text-sm outline-none"
              value={output}
              readOnly
            />
          </div>
        </div>

        <p className="mt-6 text-xs text-zinc-400">
          Nota: GitHub Pages no ejecuta modelos. La inferencia ocurre en Hugging
          Face Spaces.
        </p>
      </div>
    </main>
  );
}
