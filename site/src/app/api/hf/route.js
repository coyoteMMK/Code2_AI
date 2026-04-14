import { NextResponse } from "next/server";
import { Client } from "@gradio/client";

export const runtime = "nodejs";

const SPACE_ID = "coyoteMMK/Code2_AI";
const ENDPOINT = "/generar";
const DEFAULT_PROMPT = "mov eax, 1";
const DEFAULT_BEAMS = 1;
const DEFAULT_TEMPERATURE = 0.2;
const DEFAULT_TOP_P = 0.6;
const HF_TOKEN = process.env.HF_TOKEN;

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
    const client = HF_TOKEN
      ? await Client.connect(SPACE_ID, { token: HF_TOKEN })
      : await Client.connect(SPACE_ID);
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
    const message = String(error?.message ?? error ?? "Error desconocido");
    const lowerMessage = message.toLowerCase();
    const status = lowerMessage.includes("paused") ? 503 : 502;

    return NextResponse.json(
      {
        ok: false,
        error: message,
        warmup,
      },
      { status }
    );
  }
}