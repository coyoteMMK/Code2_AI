"use client";

import { useMemo, useState, useEffect, useRef } from "react";
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

  const textareaRef = useRef(null);
  const chatEndRef = useRef(null);

  const [modelChoice, setModelChoice] = useState("onnx");
  const [beams, setBeams] = useState(1);
  const [temperature, setTemperature] = useState(0.2);
  const [topP, setTopP] = useState(0.6);

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

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

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
        modelChoice === "onnx" ? "ONNX INT8 (rápido)" : "Full FP32 (más preciso)";

      const result = await client.predict(ENDPOINT, [
        input,
        modelo_sel,
        beams,
        temperature,
        topP,
      ]);

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

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey) {
      e.preventDefault();
      run();
    }
  };

  return (
    <main className="relative flex h-screen flex-col bg-gradient-to-br from-slate-950 via-slate-900 to-stone-900 text-slate-100">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(75%_60%_at_20%_0%,rgba(148,163,184,0.12),transparent_70%)]" />
      {/* Header */}
      <div className="relative border-b border-white/10 bg-black/40 px-6 py-4 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 rounded-full border border-white/10 bg-slate-800/60 px-3 py-1.5">
            <span className="text-xs font-medium text-slate-300">Modelo</span>
            <select
              className="rounded-full bg-slate-900/60 px-2 py-1 text-xs text-slate-100 outline-none"
              value={modelChoice}
              onChange={(e) => setModelChoice(e.target.value)}
              disabled={loading}
            >
              <option value="onnx">ONNX INT8 (rapido)</option>
              <option value="fp32">Full FP32 (preciso)</option>
            </select>
          </div>

          <div className="flex min-w-[200px] flex-1 items-center justify-center">
            <div className="flex items-center gap-3">
              <img
                src={`${process.env.NEXT_PUBLIC_BASE_PATH || ""}/code2-logo.svg`}
                alt="Code-2"
                className="h-8 w-8"
              />
              <div>
                <h1 className="text-xl font-semibold text-slate-100">Code-2 Translator</h1>
                <p className="text-sm text-slate-400">
                  Traduce lenguaje natural a ensamblador CODE-2
                </p>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 rounded-full border border-white/10 bg-slate-800/60 px-3 py-1.5">
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-medium text-slate-300">Beams</span>
              <input
                className="w-14 rounded-full bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100 outline-none"
                type="number"
                min={1}
                max={8}
                value={beams}
                onChange={(e) => setBeams(parseInt(e.target.value || "1", 10))}
                disabled={loading}
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-medium text-slate-300">Temp</span>
              <input
                className="w-16 rounded-full bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100 outline-none"
                type="number"
                min={0.1}
                max={2.0}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value || "0.2"))}
                disabled={loading}
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-medium text-slate-300">Top-p</span>
              <input
                className="w-16 rounded-full bg-slate-900/60 px-2 py-1 text-[11px] text-slate-100 outline-none"
                type="number"
                min={0.1}
                max={1.0}
                step={0.1}
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value || "0.6"))}
                disabled={loading}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="relative flex-1 overflow-y-auto px-6 py-6 soft-scroll">
        <div className="mx-auto max-w-5xl space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-2xl rounded-2xl px-5 py-3 ${
                  msg.type === "user"
                    ? "bg-slate-700/80 text-slate-100 rounded-br-none"
                    : msg.isError
                    ? "bg-rose-950/40 border border-rose-400/40 text-rose-100 rounded-bl-none"
                    : "bg-slate-800/70 border border-white/10 text-slate-100 rounded-bl-none"
                }`}
              >
                <p className="whitespace-pre-wrap break-words text-base leading-relaxed">
                  {msg.content}
                </p>
                {msg.latency && (
                  <p className="mt-2 text-sm text-slate-400">
                    ⏱ {msg.latency.toFixed(3)}s • {msg.model}
                  </p>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-none bg-slate-800/70 border border-white/10 px-5 py-3">
                <div className="flex gap-2">
                  <div className="h-2 w-2 rounded-full bg-slate-400 animate-bounce" />
                  <div className="h-2 w-2 rounded-full bg-slate-400 animate-bounce delay-100" />
                  <div className="h-2 w-2 rounded-full bg-slate-400 animate-bounce delay-200" />
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
      </div>

      {/* Settings & Input Bar */}
      <div className="relative px-6 py-4 mb-3">
        <div className="mx-auto max-w-5xl">
          <div className="flex items-end gap-3 rounded-2xl border border-white/10 bg-slate-950/40 p-2">
            <textarea
              ref={textareaRef}
              className="soft-scroll flex-1 rounded-2xl bg-slate-900/70 border border-white/10 px-4 py-2 text-base leading-6 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-slate-300 resize-none max-h-40 min-h-[44px] overflow-y-auto"
              placeholder="Escribe tu instruccion aqui... (Enter para enviar, Shift/Ctrl+Enter para nueva linea)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              rows={1}
            />
            <button
              onClick={run}
              disabled={loading || !input.trim()}
              className="self-end rounded-2xl bg-slate-100 px-4 py-3 text-sm font-semibold text-slate-900 shadow-sm hover:bg-white disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300 disabled:shadow-none transition-colors"
            >
              {loading ? (
                "..."
              ) : (
                <svg
                  className="h-5 w-5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
