"use client";

import { useMemo, useState, useEffect } from "react";
import { Client } from "@gradio/client";

export default function Page() {
  const [input, setInput] = useState("");
  const [displayedOutput, setDisplayedOutput] = useState("");
  const [fullOutput, setFullOutput] = useState("");
  const [latency, setLatency] = useState(null);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "assistant",
      content: "Hola! 👋 Soy Code-2 Translator. Cuéntame qué instrucciones quieres convertir a ensamblador.",
    },
  ]);

  const [modelChoice, setModelChoice] = useState("onnx");
  const [beams, setBeams] = useState(4);

  const SPACE_ID = useMemo(() => "coyoteMMK/Code2_AI", []);
  const ENDPOINT = useMemo(() => "/generar", []);

  // Efecto de escritura letra a letra
  useEffect(() => {
    if (!loading && fullOutput && displayedOutput !== fullOutput) {
      let index = 0;
      const interval = setInterval(() => {
        if (index <= fullOutput.length) {
          setDisplayedOutput(fullOutput.slice(0, index));
          index++;
        } else {
          clearInterval(interval);
        }
      }, 15);

      return () => clearInterval(interval);
    }
  }, [fullOutput, displayedOutput, loading]);

  async function run() {
    if (!input.trim()) return;

    // Agregar mensaje del usuario al chat
    const userMessage = {
      id: messages.length + 1,
      type: "user",
      content: input,
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setDisplayedOutput("");
    setFullOutput("");
    setLatency(null);

    try {
      const t0 = performance.now();

      const client = await Client.connect(SPACE_ID);

      const modelo_sel =
        modelChoice === "onnx" ? "ONNX INT8 (rápido)" : "Full FP32 (más pesado)";

      const result = await client.predict(ENDPOINT, [input, modelo_sel, beams]);

      const t1 = performance.now();

      const salida = String(result?.data?.[0] ?? "");
      const tiempoStr = String(result?.data?.[1] ?? "");

      setFullOutput(salida);

      const parsed = parseFloat(tiempoStr.replace("s", "").trim());
      if (!Number.isNaN(parsed)) setLatency(parsed);
      else setLatency((t1 - t0) / 1000);

      // Agregar mensaje del asistente
      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          {
            id: prev.length + 1,
            type: "assistant",
            content: salida,
            latency: !Number.isNaN(parsed) ? parsed : (t1 - t0) / 1000,
            model: modelo_sel,
          },
        ]);
      }, 100);
    } catch (e) {
      const errorMsg = "❌ Error: " + (e?.message ?? String(e));
      setFullOutput(errorMsg);
      setMessages((prev) => [
        ...prev,
        {
          id: prev.length + 1,
          type: "assistant",
          content: errorMsg,
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
      run();
    }
  };

  return (
    <main className="flex h-screen flex-col bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-slate-100">
      {/* Header */}
      <div className="border-b border-purple-500/30 bg-black/50 px-6 py-4 backdrop-blur">
        <div className="mx-auto max-w-4xl">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            ✨ Code-2 Translator
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Convierte instrucciones en lenguaje natural a ensamblador CODE-2
          </p>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-4xl space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-2xl rounded-2xl px-5 py-3 ${
                  msg.type === "user"
                    ? "bg-gradient-to-br from-purple-600 to-purple-700 text-white rounded-br-none"
                    : msg.isError
                    ? "bg-red-900/40 border border-red-500/50 text-red-200 rounded-bl-none"
                    : "bg-slate-800/80 border border-purple-500/30 text-slate-100 rounded-bl-none"
                }`}
              >
                <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
                  {msg.content}
                </p>
                {msg.latency && (
                  <p className="mt-2 text-xs text-slate-400">
                    ⏱ {msg.latency.toFixed(3)}s • {msg.model}
                  </p>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-none bg-slate-800/80 border border-purple-500/30 px-5 py-3">
                <div className="flex gap-2">
                  <div className="h-2 w-2 rounded-full bg-purple-400 animate-bounce" />
                  <div className="h-2 w-2 rounded-full bg-purple-400 animate-bounce delay-100" />
                  <div className="h-2 w-2 rounded-full bg-purple-400 animate-bounce delay-200" />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Settings & Input Bar */}
      <div className="border-t border-purple-500/30 bg-black/50 px-6 py-4 backdrop-blur">
        <div className="mx-auto max-w-4xl space-y-4">
          {/* Controles */}
          <div className="flex items-center gap-3 rounded-xl bg-slate-800/40 border border-purple-500/20 p-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-slate-400">Modelo</span>
              <select
                className="rounded-lg bg-slate-700 border border-slate-600 px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-purple-500"
                value={modelChoice}
                onChange={(e) => setModelChoice(e.target.value)}
                disabled={loading}
              >
                <option value="onnx">⚡ ONNX INT8 (Rápido)</option>
                <option value="fp32">🎯 Full FP32 (Preciso)</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-slate-400">Beams</span>
              <input
                className="w-16 rounded-lg bg-slate-700 border border-slate-600 px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-purple-500"
                type="number"
                min={1}
                max={8}
                value={beams}
                onChange={(e) => setBeams(parseInt(e.target.value || "4", 10))}
                disabled={loading}
              />
            </div>
          </div>

          {/* Input */}
          <div className="flex gap-3">
            <textarea
              className="flex-1 rounded-xl bg-slate-800 border border-purple-500/30 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-purple-500 resize-none max-h-24"
              placeholder="Escribe tu instrucción aquí... (Ctrl + Enter para enviar)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading}
              rows={2}
            />
            <button
              onClick={run}
              disabled={loading || !input.trim()}
              className="self-end rounded-xl bg-gradient-to-br from-purple-600 to-pink-600 px-6 py-3 font-semibold text-white hover:from-purple-500 hover:to-pink-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:shadow-lg hover:shadow-purple-500/50"
            >
              {loading ? "📝" : "Enviar"}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
