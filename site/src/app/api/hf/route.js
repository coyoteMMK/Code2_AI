import { NextResponse } from "next/server";
import { Client } from "@gradio/client";

export const runtime = "nodejs";

const SPACE_ID = "coyoteMMK/Code2_AI";
const SPACE_URL = "https://coyotemmk-code2-ai.hf.space";
const ENDPOINT = "/generar";
const DEFAULT_PROMPT = "mov eax, 1";
const DEFAULT_BEAMS = 1;
const DEFAULT_TEMPERATURE = 0.2;
const DEFAULT_TOP_P = 0.6;
const HF_TOKEN = process.env.HF_TOKEN;

async function connectSpaceClient() {
  const references = [SPACE_ID, SPACE_URL];
  let lastError;

  for (const reference of references) {
    try {
      if (HF_TOKEN) {
        return await Client.connect(reference, { token: HF_TOKEN });
      }

      return await Client.connect(reference);
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError ?? new Error("No se pudo conectar al Space");
}

function normalizeProxyError(error) {
  const message = String(error?.message ?? error ?? "Error desconocido");
  const lowerMessage = message.toLowerCase();

  if (lowerMessage.includes("could not resolve app config")) {
    return {
      status: 503,
      error:
        "El Space respondió sin configuración válida de Gradio (app config). Suele pasar cuando está caído o arrancando con error (HTTP 500). Revisa el estado en Hugging Face y reinícialo.",
    };
  }

  if (lowerMessage.includes("paused") || lowerMessage.includes("sleeping")) {
    return {
      status: 503,
      error: "El Space está pausado o durmiendo. Intenta de nuevo en unos segundos.",
    };
  }

  return {
    status: 502,
    error: message,
  };
}

export async function POST(request) {
  let body = {};

  try {
    body = await request.json();
  } catch {
    body = {};
  }

  const warmup = Boolean(body?.warmup);
  const modelChoice = body?.modelChoice === "onnx" ? "onnx" : "fp32";
  const modelLabel = modelChoice === "onnx" ? "ONNX INT8 (rápido)" : "Full FP32 (más preciso)";
  const prompt = warmup ? DEFAULT_PROMPT : String(body?.prompt ?? "");
  const beams = Number.isFinite(body?.beams) ? body.beams : DEFAULT_BEAMS;
  const temperature = Number.isFinite(body?.temperature) ? body.temperature : DEFAULT_TEMPERATURE;
  const topP = Number.isFinite(body?.topP) ? body.topP : DEFAULT_TOP_P;

  const startedAt = Date.now();

  try {
    const client = await connectSpaceClient();
    const result = await client.predict(ENDPOINT, [
      prompt,
      modelLabel,
      beams,
      temperature,
      topP,
    ]);

    return NextResponse.json({
      ok: true,
      data: result?.data ?? [],
      latency: `${((Date.now() - startedAt) / 1000).toFixed(3)}s`,
      warmup,
    });
  } catch (error) {
    const mappedError = normalizeProxyError(error);

    return NextResponse.json(
      {
        ok: false,
        error: mappedError.error,
        warmup,
      },
      { status: mappedError.status }
    );
  }
}