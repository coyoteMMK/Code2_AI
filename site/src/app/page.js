"use client";

import { useMemo, useState, useEffect, useRef } from "react";
import Image from "next/image";

export default function Page() {
  const publicBasePath = "";
  const WELCOME_TEXT =
    "Hola! 👋 Soy Code-2 Translator. Cuéntame qué instrucciones quieres convertir a ensamblador.";
  const DEFAULT_SETTINGS = {
    modelChoice: "fp32",
    beams: 1,
    temperature: 0.2,
    topP: 0.6,
  };

  const [input, setInput] = useState("");
  const [displayedOutput, setDisplayedOutput] = useState("");
  const [fullOutput, setFullOutput] = useState("");
  const [latency, setLatency] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingHint, setLoadingHint] = useState("Generando...");
  const [serverPaused, setServerPaused] = useState(false);
  const [warmingUp, setWarmingUp] = useState(true);
  const [warmupStatus, setWarmupStatus] = useState("Preparando...");
  const [warmupVisible, setWarmupVisible] = useState(true);
  const [warmupFadingOut, setWarmupFadingOut] = useState(false);
  const [typingMessageId, setTypingMessageId] = useState(null);
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "assistant",
      content: "",
      isWelcome: true,
      isTyping: true,
    },
  ]);

  const textareaRef = useRef(null);
  const chatEndRef = useRef(null);
  const paramsMenuRef = useRef(null);
  const paramsToggleRef = useRef(null);

  const [modelChoice, setModelChoice] = useState(DEFAULT_SETTINGS.modelChoice);
  const [beams, setBeams] = useState(DEFAULT_SETTINGS.beams);
  const [temperature, setTemperature] = useState(DEFAULT_SETTINGS.temperature);
  const [topP, setTopP] = useState(DEFAULT_SETTINGS.topP);
  const [showParamsMenu, setShowParamsMenu] = useState(false);
  const [isInputMultiline, setIsInputMultiline] = useState(false);

  // En Vercel, la API route maneja la conexión al Space
  const API_ENDPOINT = "/api/generate";

  const normalizeInteger = (value, min, max, fallback) => {
    const parsed = Number.parseInt(String(value), 10);
    if (Number.isNaN(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  const normalizeFloat = (value, min, max, fallback) => {
    const parsed = Number.parseFloat(String(value).replace(",", "."));
    if (Number.isNaN(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  const adjustInteger = (value, delta, min, max, fallback) => {
    const base = normalizeInteger(value, min, max, fallback);
    return normalizeInteger(base + delta, min, max, fallback);
  };

  const adjustFloat = (value, delta, min, max, fallback, decimals = 1) => {
    const base = normalizeFloat(value, min, max, fallback);
    const next = Math.min(max, Math.max(min, base + delta));
    return Number(next.toFixed(decimals));
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const isRecoverableSpaceError = (error) => {
    const msg = String(error?.message ?? error ?? "").toLowerCase();

    return (
      msg.includes("sleep") ||
      msg.includes("loading") ||
      msg.includes("building") ||
      msg.includes("starting") ||
      msg.includes("503") ||
      msg.includes("connection errored out") ||
      msg.includes("space is not ready") ||
      msg.includes("fetch failed") ||
      msg.includes("not healthy") ||
      msg.includes("runtime error") ||
      msg.includes("queue is full") ||
      msg.includes("connection refused")
    );
  };

  const isOwnerPausedError = (error) => {
    const msg = String(error?.message ?? error ?? "").toLowerCase();
    return (
      msg.includes("paused by its owner") ||
      msg.includes("space is paused") ||
      msg.includes("this space is paused")
    );
  };

  async function callGenerateAPI(payload, setHint, maxAttempts = 6) {
    let lastError;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        if (attempt === 1) {
          setHint?.("Conectando con Hugging Face...");
        } else {
          setHint?.(`⏳ Esperando a que el Space esté listo... intento ${attempt}/${maxAttempts}`);
        }

        const response = await fetch(API_ENDPOINT, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const errorMsg = errorData.error || `Error ${response.status}`;

          if (response.status === 503) {
            // Space durmiendo o pausado, reintenta
            lastError = new Error(errorMsg);
            if (errorMsg.includes("pausado")) {
              throw lastError; // Error no recuperable
            }
            if (attempt < maxAttempts) {
              await sleep(Math.min(2000 * attempt, 8000));
              continue;
            }
          }

          throw new Error(errorMsg);
        }

        const result = await response.json();
        setHint?.("Servidor activo. Respuesta generada.");
        return result;
      } catch (error) {
        lastError = error;

        if (isOwnerPausedError(error)) {
          throw error;
        }

        if (!isRecoverableSpaceError(error) || attempt === maxAttempts) {
          throw error;
        }

        await sleep(Math.min(2000 * attempt, 8000));
      }
    }

    throw lastError;
  }

  const handleBeamsChange = (rawValue) => {
    if (rawValue === "") {
      setBeams("");
      return;
    }

    if (!/^\d+$/.test(rawValue)) return;

    const parsed = Number.parseInt(rawValue, 10);
    setBeams(parsed > 8 ? "8" : rawValue);
  };

  const handleTemperatureChange = (rawValue) => {
    if (rawValue === "") {
      setTemperature("");
      return;
    }

    if (!/^\d*([.,]\d*)?$/.test(rawValue)) return;

    const normalizedRaw = rawValue.replace(",", ".");
    const parsed = Number.parseFloat(normalizedRaw);

    if (!Number.isNaN(parsed) && parsed > 1) {
      setTemperature("1");
      return;
    }

    setTemperature(normalizedRaw);
  };

  const handleTopPChange = (rawValue) => {
    if (rawValue === "") {
      setTopP("");
      return;
    }

    if (!/^\d*([.,]\d*)?$/.test(rawValue)) return;

    const normalizedRaw = rawValue.replace(",", ".");
    const parsed = Number.parseFloat(normalizedRaw);

    if (!Number.isNaN(parsed) && parsed > 1) {
      setTopP("1");
      return;
    }

    setTopP(normalizedRaw);
  };

  useEffect(() => {
    if (loading || !fullOutput || !typingMessageId) return;

    let index = 0;
    setDisplayedOutput("");

    const interval = setInterval(() => {
      index += 1;

      if (index <= fullOutput.length) {
        setDisplayedOutput(fullOutput.slice(0, index));
        return;
      }

      clearInterval(interval);
    }, 15);

    return () => clearInterval(interval);
  }, [fullOutput, loading, typingMessageId]);

  useEffect(() => {
    if (!typingMessageId) return;

    setMessages((prev) =>
      prev.map((message) =>
        message.id === typingMessageId
          ? {
              ...message,
              content: displayedOutput,
            }
          : message
      )
    );

    if (fullOutput && displayedOutput === fullOutput) {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === typingMessageId
            ? {
                ...message,
                content: fullOutput,
                isTyping: false,
              }
            : message
        )
      );
      setTypingMessageId(null);
      setDisplayedOutput("");
      setFullOutput("");
    }
  }, [displayedOutput, fullOutput, typingMessageId]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight + 2}px`;

      const hasLineBreak = input.includes("\n");
      const height = textareaRef.current.scrollHeight;

      setIsInputMultiline((prev) => {
        if (input.length === 0) return false;

        const shouldEnterMultiline = hasLineBreak || height > 56 || input.length > 80;
        const canReturnSingleLine = !hasLineBreak && height <= 52 && input.length < 60;

        if (prev) return !canReturnSingleLine;
        return shouldEnterMultiline;
      });
    }
  }, [input]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (!showParamsMenu) return;

    const handlePointerDown = (event) => {
      const target = event.target;

      if (paramsMenuRef.current?.contains(target) || paramsToggleRef.current?.contains(target)) {
        return;
      }

      setShowParamsMenu(false);
    };

    document.addEventListener("mousedown", handlePointerDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
    };
  }, [showParamsMenu]);

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      if (index <= WELCOME_TEXT.length) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === 1 && m.isWelcome
              ? {
                  ...m,
                  content: WELCOME_TEXT.slice(0, index),
                  isTyping: true,
                }
              : m
          )
        );
        index++;
      } else {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === 1 && m.isWelcome
              ? {
                  ...m,
                  isTyping: false,
                }
              : m
          )
        );
        clearInterval(interval);
      }
    }, 18);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let hideReadyMessageTimer;
    let fadeOutTimer;

    const setReadyStatus = () => {
      if (cancelled) return;
      setWarmupVisible(true);
      setWarmupFadingOut(false);
      setWarmupStatus("✅ Servidor listo");
      clearTimeout(hideReadyMessageTimer);
      clearTimeout(fadeOutTimer);

      hideReadyMessageTimer = setTimeout(() => {
        if (cancelled) return;
        setWarmupFadingOut(true);

        fadeOutTimer = setTimeout(() => {
          if (!cancelled) {
            setWarmupStatus("");
            setWarmupVisible(false);
            setWarmupFadingOut(false);
          }
        }, 350);
      }, 3000);
    };

    async function warmupSpace() {
      try {
        setWarmingUp(true);
        setWarmupVisible(true);
        setWarmupFadingOut(false);
        setWarmupStatus("Conectando con Hugging Face...");

        // Llamada mínima para despertar el Space
        try {
          const response = await fetch(`${API_ENDPOINT}?action=warmup`, {
            method: "GET",
          });

          if (!response.ok && response.status !== 503) {
            throw new Error(`Warmup failed: ${response.status}`);
          }
        } catch {
          // No bloqueamos la UI si falla el warmup
        }

        if (!cancelled) {
          setReadyStatus();
        }
      } catch (e) {
        if (!cancelled) {
          if (isOwnerPausedError(e)) {
            setWarmupVisible(true);
            setWarmupFadingOut(false);
            setWarmupStatus(
              "⚠️ El Space está pausado por el owner. Hay que reiniciarlo en Hugging Face."
            );
          } else {
            setWarmupVisible(true);
            setWarmupFadingOut(false);
            setWarmupStatus(
              "⚠️ No pude confirmar el estado del servidor. Se reintentará al enviar una consulta."
            );
          }
        }
      } finally {
        if (!cancelled) {
          setTimeout(() => {
            setWarmingUp(false);
          }, 1200);
        }
      }
    }

    warmupSpace();

    return () => {
      cancelled = true;
      clearTimeout(hideReadyMessageTimer);
      clearTimeout(fadeOutTimer);
    };
  }, [modelChoice]);

  const resetSettings = () => {
    setModelChoice(DEFAULT_SETTINGS.modelChoice);
    setBeams(DEFAULT_SETTINGS.beams);
    setTemperature(DEFAULT_SETTINGS.temperature);
    setTopP(DEFAULT_SETTINGS.topP);
  };

  const commitSettings = () => {
    setBeams(normalizeInteger(beams, 1, 8, DEFAULT_SETTINGS.beams));
    setTemperature(normalizeFloat(temperature, 0.1, 1, DEFAULT_SETTINGS.temperature));
    setTopP(normalizeFloat(topP, 0.1, 1, DEFAULT_SETTINGS.topP));
  };

  const copyResult = async (messageId, content) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 1400);
    } catch {
      setCopiedMessageId(null);
    }
  };

  async function run() {
    if (!input.trim()) return;

    const userPrompt = input;
    const userMessageId = Date.now();

    const requestBeams = normalizeInteger(beams, 1, 8, DEFAULT_SETTINGS.beams);
    const requestTemperature = normalizeFloat(temperature, 0.1, 1, DEFAULT_SETTINGS.temperature);
    const requestTopP = normalizeFloat(topP, 0.1, 1, DEFAULT_SETTINGS.topP);

    setBeams(requestBeams);
    setTemperature(requestTemperature);
    setTopP(requestTopP);

    const userMessage = {
      id: userMessageId,
      type: "user",
      content: userPrompt,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setLoadingHint("Conectando...");
    setServerPaused(false);
    setDisplayedOutput("");
    setFullOutput("");
    setLatency(null);

    try {
      const t0 = performance.now();

      const modelo_sel =
        modelChoice === "onnx" ? "ONNX INT8 (rápido)" : "Full FP32 (más preciso)";

      const result = await callGenerateAPI(
        {
          input: userPrompt,
          modelChoice,
          beams: requestBeams,
          temperature: requestTemperature,
          topP: requestTopP,
        },
        (msg) => {
          setLoadingHint(msg);
          if (/arrancando|despertando|esperando|durmiendo/i.test(msg)) {
            setServerPaused(true);
          } else {
            setServerPaused(false);
          }
        },
        6
      );

      const t1 = performance.now();

      const salida = String(result?.data?.[0] ?? "");
      const tiempoStr = String(result?.data?.[1] ?? "");
      const parsed = parseFloat(tiempoStr.replace("s", "").trim());
      const finalLatency = !Number.isNaN(parsed) ? parsed : (t1 - t0) / 1000;

      const assistantMessageId = userMessageId + 1;

      setMessages((prev) => [
        ...prev,
        {
          id: assistantMessageId,
          type: "assistant",
          content: "",
          latency: finalLatency,
          model: modelo_sel,
          isTyping: true,
        },
      ]);

      setTypingMessageId(assistantMessageId);
      setFullOutput(salida);
      setLatency(finalLatency);
    } catch (e) {
      const rawError = String(e?.message ?? e ?? "");
      const isOwnerPaused = isOwnerPausedError(e);
      const isRecoverable = isRecoverableSpaceError(e);

      const errorMsg = isOwnerPaused
        ? "⚠️ Space en pausa en HF."
        : isRecoverable
        ? "⏳ Space iniciando. Intenta de nuevo."
        : "❌ Error: " + rawError;

      setMessages((prev) => [
        ...prev,
        {
          id: userMessageId + 1,
          type: "assistant",
          content: errorMsg,
          isError: !isRecoverable,
        },
      ]);
    } finally {
      setLoading(false);
      setLoadingHint("Generando...");
      setServerPaused(false);
    }
  }

  const handleKeyDown = (e) => {
    if (e.key !== "Enter") return;

    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      e.preventDefault();

      const textarea = textareaRef.current;
      const start = textarea?.selectionStart ?? input.length;
      const end = textarea?.selectionEnd ?? input.length;
      const nextValue = `${input.slice(0, start)}\n${input.slice(end)}`;

      setInput(nextValue);

      requestAnimationFrame(() => {
        if (textareaRef.current) {
          const nextCaret = start + 1;
          textareaRef.current.selectionStart = nextCaret;
          textareaRef.current.selectionEnd = nextCaret;
        }
      });
      return;
    }

    e.preventDefault();
    run();
  };

  return (
    <main className="relative flex h-dvh flex-col overflow-hidden bg-linear-to-br from-slate-950 via-slate-900 to-stone-900 text-slate-100">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(75%_60%_at_20%_0%,rgba(148,163,184,0.12),transparent_70%)]" />

      <div className="relative border-b border-white/10 bg-black/40 px-3 py-3 sm:px-6 sm:py-4 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 lg:flex-nowrap">
          <div className="flex min-w-0 flex-wrap items-center gap-3">
            <Image
              src={`${publicBasePath}/code2-logo.svg`}
              alt="Code-2"
              width={40}
              height={40}
              className="h-8 w-8 sm:h-10 sm:w-10"
            />

            <h1 className="text-xl sm:text-3xl font-semibold text-slate-100">Code-2 Translator</h1>
          </div>

          {warmupVisible && (warmingUp || warmupStatus || warmupFadingOut) && (
            <div
              className={`order-3 flex basis-full justify-center lg:pointer-events-none lg:absolute lg:left-1/2 lg:top-1/2 lg:z-10 lg:w-auto lg:-translate-x-1/2 lg:-translate-y-1/2 ${
                warmupFadingOut ? "opacity-0" : "opacity-100"
              } transition-opacity duration-300`}
            >
              <div className="inline-flex w-fit max-w-[92vw] sm:max-w-2xl rounded-xl border border-cyan-400/30 bg-cyan-950/30 px-3 sm:px-6 py-2 text-xs sm:text-sm text-cyan-100 lg:pointer-events-auto">
                {warmupStatus}
              </div>
            </div>
          )}

          <div className="ml-auto flex w-full sm:w-auto items-center justify-between sm:justify-start gap-3 rounded-2xl bg-slate-900/55 px-3 sm:px-4 py-2 shadow-[0_0_0_1px_rgba(255,255,255,0.05)] backdrop-blur-sm">
            <div className="flex flex-col leading-none">
              <span className="text-[10px] font-medium uppercase tracking-[0.18em] text-slate-300">
                Modelo
              </span>
            </div>

            <div className="relative">
              <select
                className="appearance-none rounded-xl bg-slate-800/80 py-2 pl-3 sm:pl-4 pr-9 sm:pr-10 text-xs sm:text-sm font-medium text-slate-100 outline-none ring-1 ring-white/8 transition focus:ring-2 focus:ring-cyan-400/50 disabled:cursor-not-allowed disabled:opacity-60"
                value={modelChoice}
                onChange={(e) => setModelChoice(e.target.value)}
                disabled={loading}
              >
                <option value="onnx">ONNX INT8 · rápido</option>
                <option value="fp32">Full FP32 · preciso</option>
              </select>

              <svg
                className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
                viewBox="0 0 20 20"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                aria-hidden="true"
              >
                <path d="M6 8l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      <div className="relative min-h-0 flex-1 overflow-y-auto px-3 py-4 sm:px-6 sm:py-6 soft-scroll">
        <div className="mx-auto max-w-5xl space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`relative max-w-[90%] sm:max-w-2xl rounded-2xl px-3 sm:px-5 py-2.5 sm:py-3 ${
                  msg.type === "user"
                    ? "bg-slate-700/80 text-slate-100 rounded-br-none"
                    : msg.isError
                    ? "bg-rose-950/40 border border-rose-400/40 text-rose-100 rounded-bl-none"
                    : "bg-slate-800/70 border border-white/10 text-slate-100 rounded-bl-none"
                }`}
              >
                {msg.type === "assistant" && !msg.isError && !msg.isWelcome && msg.content?.trim() && (
                  <button
                    onClick={() => copyResult(msg.id, msg.content)}
                    className="absolute right-2 top-2 sm:right-3 sm:top-3 inline-flex h-8 w-8 items-center justify-center rounded-md border border-white/15 bg-slate-900/80 text-slate-200 transition-colors hover:bg-slate-800"
                    title={copiedMessageId === msg.id ? "Copiado" : "Copiar"}
                    aria-label={copiedMessageId === msg.id ? "Copiado" : "Copiar respuesta"}
                  >
                    <Image
                      src={`${publicBasePath}/copy.svg`}
                      alt="Copiar"
                      width={16}
                      height={16}
                      className="h-4 w-4"
                    />
                  </button>
                )}

                <p
                  className={`whitespace-pre-wrap wrap-break-word text-sm sm:text-base leading-relaxed ${
                    msg.type === "assistant" && !msg.isError && !msg.isWelcome && msg.content?.trim()
                      ? "pr-10 sm:pr-12"
                      : ""
                  }`}
                >
                  {msg.content}
                  {msg.isTyping && (
                    <span
                      className="ml-0.5 inline-block h-5 w-0.5 animate-pulse align-middle bg-cyan-300"
                      aria-hidden="true"
                    />
                  )}
                </p>

                {(msg.type === "assistant" && !msg.isError) || msg.latency ? (
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    {msg.latency && (
                      <p className="text-xs sm:text-sm text-slate-400">
                        ⏱ {msg.latency.toFixed(3)}s • {msg.model}
                      </p>
                    )}
                  </div>
                ) : null}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-none bg-slate-800/70 border border-white/10 px-4 sm:px-5 py-3 max-w-[90%] sm:max-w-2xl">
                <div className="flex gap-2">
                  <div className="h-2 w-2 rounded-full bg-slate-400 animate-bounce" />
                  <div className="h-2 w-2 rounded-full bg-slate-400 animate-bounce delay-100" />
                  <div className="h-2 w-2 rounded-full bg-slate-400 animate-bounce delay-200" />
                </div>
                <p className="mt-2 text-sm text-slate-300">{loadingHint}</p>
                {serverPaused && (
                  <p className="mt-1 text-xs text-amber-300">
                    Consejo: la primera petición tras inactividad suele tardar más.
                  </p>
                )}
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>
      </div>

      <div className="relative px-3 pb-3 sm:px-6 sm:pb-4">
        <div className="relative mx-auto max-w-5xl">
          {showParamsMenu && (
            <div className="absolute bottom-full left-0 z-30 mb-3 w-[min(75vw,15rem)] translate-x-0 sm:w-fit max-w-[calc(100vw-1.5rem)] sm:max-w-[calc(100vw-3rem)] rounded-3xl border border-white/10 bg-slate-950/92 p-2 sm:p-3 shadow-[0_24px_80px_rgba(2,6,23,0.65)] backdrop-blur-xl">
              <div
                ref={paramsMenuRef}
                className="flex flex-col sm:flex-row sm:flex-wrap items-stretch sm:items-end gap-2 sm:gap-3"
              >
                <label className="flex w-full sm:w-auto min-w-0 sm:min-w-33 flex-col items-center gap-2 rounded-2xl bg-slate-900/60 px-3 py-3 text-center shadow-[0_0_0_1px_rgba(255,255,255,0.05)]">
                  <span className="w-full text-center text-[10px] font-medium uppercase tracking-[0.18em] text-slate-400">
                    Beams
                  </span>
                  <div className="flex items-center gap-1 rounded-xl bg-slate-800/85 p-1 ring-1 ring-white/8 transition focus-within:ring-2 focus-within:ring-cyan-400/50">
                    <button
                      type="button"
                      onClick={() =>
                        setBeams((prev) => adjustInteger(prev, -1, 1, 8, DEFAULT_SETTINGS.beams))
                      }
                      disabled={loading}
                      className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-base font-semibold text-slate-200 transition disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <span className="text-2xl leading-none symbol-hover">−</span>
                    </button>

                    <input
                      className="no-number-spinner h-8 w-16 bg-transparent text-center text-sm font-medium text-slate-100 outline-none"
                      type="text"
                      inputMode="numeric"
                      min={1}
                      max={8}
                      step={1}
                      value={beams}
                      onChange={(e) => handleBeamsChange(e.target.value)}
                      onBlur={() => setBeams(normalizeInteger(beams, 1, 8, DEFAULT_SETTINGS.beams))}
                      disabled={loading}
                    />

                    <button
                      type="button"
                      onClick={() =>
                        setBeams((prev) => adjustInteger(prev, 1, 1, 8, DEFAULT_SETTINGS.beams))
                      }
                      disabled={loading}
                      className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-base font-semibold text-slate-200 transition disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <span className="text-2xl leading-none symbol-hover">+</span>
                    </button>
                  </div>
                </label>

                <label className="flex w-full sm:w-auto min-w-0 sm:min-w-33 flex-col items-center gap-2 rounded-2xl bg-slate-900/60 px-3 py-3 text-center shadow-[0_0_0_1px_rgba(255,255,255,0.05)]">
                  <span className="w-full text-center text-[10px] font-medium uppercase tracking-[0.18em] text-slate-400">
                    Temp
                  </span>
                  <div className="flex items-center gap-1 rounded-xl bg-slate-800/85 p-1 ring-1 ring-white/8 transition focus-within:ring-2 focus-within:ring-cyan-400/50">
                    <button
                      type="button"
                      onClick={() =>
                        setTemperature((prev) =>
                          adjustFloat(prev, -0.1, 0.1, 1, DEFAULT_SETTINGS.temperature)
                        )
                      }
                      disabled={loading}
                      className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-base font-semibold text-slate-200 transition disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <span className="text-2xl leading-none symbol-hover">−</span>
                    </button>

                    <input
                      className="no-number-spinner h-8 w-16 bg-transparent text-center text-sm font-medium text-slate-100 outline-none"
                      type="text"
                      inputMode="decimal"
                      min={0.1}
                      max={1}
                      step={0.1}
                      value={temperature}
                      onChange={(e) => handleTemperatureChange(e.target.value)}
                      onBlur={() =>
                        setTemperature(
                          normalizeFloat(temperature, 0.1, 1, DEFAULT_SETTINGS.temperature)
                        )
                      }
                      disabled={loading}
                    />

                    <button
                      type="button"
                      onClick={() =>
                        setTemperature((prev) =>
                          adjustFloat(prev, 0.1, 0.1, 1, DEFAULT_SETTINGS.temperature)
                        )
                      }
                      disabled={loading}
                      className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-base font-semibold text-slate-200 transition disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <span className="text-2xl leading-none symbol-hover">+</span>
                    </button>
                  </div>
                </label>

                <label className="flex w-full sm:w-auto min-w-0 sm:min-w-33 flex-col items-center gap-2 rounded-2xl bg-slate-900/60 px-3 py-3 text-center shadow-[0_0_0_1px_rgba(255,255,255,0.05)]">
                  <span className="w-full text-center text-[10px] font-medium uppercase tracking-[0.18em] text-slate-400">
                    Top-p
                  </span>
                  <div className="flex items-center gap-1 rounded-xl bg-slate-800/85 p-1 ring-1 ring-white/8 transition focus-within:ring-2 focus-within:ring-cyan-400/50">
                    <button
                      type="button"
                      onClick={() =>
                        setTopP((prev) => adjustFloat(prev, -0.1, 0.1, 1, DEFAULT_SETTINGS.topP))
                      }
                      disabled={loading}
                      className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-base font-semibold text-slate-200 transition disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <span className="text-2xl leading-none symbol-hover">−</span>
                    </button>

                    <input
                      className="no-number-spinner h-8 w-16 bg-transparent text-center text-sm font-medium text-slate-100 outline-none"
                      type="text"
                      inputMode="decimal"
                      min={0.1}
                      max={1}
                      step={0.1}
                      value={topP}
                      onChange={(e) => handleTopPChange(e.target.value)}
                      onBlur={() => setTopP(normalizeFloat(topP, 0.1, 1, DEFAULT_SETTINGS.topP))}
                      disabled={loading}
                    />

                    <button
                      type="button"
                      onClick={() =>
                        setTopP((prev) => adjustFloat(prev, 0.1, 0.1, 1, DEFAULT_SETTINGS.topP))
                      }
                      disabled={loading}
                      className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-base font-semibold text-slate-200 transition disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <span className="text-2xl leading-none symbol-hover">+</span>
                    </button>
                  </div>
                </label>

                <div className="flex w-full sm:w-auto min-h-21 min-w-0 sm:min-w-40 items-center justify-center rounded-2xl bg-slate-900/60 px-3 py-3 shadow-[0_0_0_1px_rgba(255,255,255,0.05)]">
                  <button
                    onClick={resetSettings}
                    disabled={loading}
                    className="inline-flex h-12 w-full items-center justify-center rounded-2xl bg-slate-800/85 px-4 text-sm font-semibold text-slate-100 shadow-[0_0_0_1px_rgba(255,255,255,0.08)] transition hover:bg-slate-700/85 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    Valores por defecto
                  </button>
                </div>
              </div>
            </div>
          )}

          <div className="rounded-2xl border border-white/10 bg-slate-950/40 p-2">
            <div className={isInputMultiline ? "flex flex-col gap-2" : "flex items-end gap-2 sm:gap-3"}>
              {!isInputMultiline && (
                <button
                  ref={paramsToggleRef}
                  onClick={() => {
                    commitSettings();
                    setShowParamsMenu((prev) => !prev);
                  }}
                  disabled={loading}
                  className="self-end inline-flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-xl border border-white/15 bg-slate-900/70 text-slate-100 transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                  title={showParamsMenu ? "Ocultar parámetros" : "Mostrar parámetros"}
                  aria-label={showParamsMenu ? "Ocultar parámetros" : "Mostrar parámetros"}
                >
                  <span className="text-2xl leading-none symbol-hover">{showParamsMenu ? "−" : "+"}</span>
                </button>
              )}

              <textarea
                ref={textareaRef}
                className={`soft-scroll rounded-2xl bg-slate-900/70 border border-white/10 px-3 sm:px-4 py-2 text-sm sm:text-base leading-6 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-slate-300 resize-none max-h-34 sm:max-h-40 min-h-10 sm:min-h-11 overflow-y-auto ${
                  isInputMultiline ? "w-full" : "flex-1"
                }`}
                placeholder="Escribe aquí..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loading}
                rows={1}
              />

              {isInputMultiline ? (
                <div className="flex items-center justify-between px-1">
                  <button
                    ref={paramsToggleRef}
                    onClick={() => {
                      commitSettings();
                      setShowParamsMenu((prev) => !prev);
                    }}
                    disabled={loading}
                    className="inline-flex h-10 w-10 sm:h-11 sm:w-11 items-center justify-center rounded-xl border border-white/15 bg-slate-900/70 text-slate-100 transition-colors hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                    title={showParamsMenu ? "Ocultar parámetros" : "Mostrar parámetros"}
                    aria-label={showParamsMenu ? "Ocultar parámetros" : "Mostrar parámetros"}
                  >
                    <span className="text-2xl leading-none symbol-hover">{showParamsMenu ? "−" : "+"}</span>
                  </button>

                  <div className="flex-1" />

                  <button
                    onClick={run}
                    disabled={loading || !input.trim()}
                    className="rounded-2xl bg-slate-100 px-3 sm:px-4 py-2.5 sm:py-3 text-sm font-semibold text-slate-900 shadow-sm hover:bg-white disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300 disabled:shadow-none transition-colors"
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
              ) : (
                <button
                  onClick={run}
                  disabled={loading || !input.trim()}
                  className="self-end rounded-2xl bg-slate-100 px-3 sm:px-4 py-2.5 sm:py-3 text-sm font-semibold text-slate-900 shadow-sm hover:bg-white disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-300 disabled:shadow-none transition-colors"
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
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
