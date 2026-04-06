export async function POST(request) {
  try {
    const { input, modelChoice, beams, temperature, topP } = await request.json();

    // Validación básica
    if (!input?.trim()) {
      return Response.json(
        { error: "Input vacío" },
        { status: 400 }
      );
    }

    // Construir el modelo_sel igual al frontend
    const modelo_sel = modelChoice === "onnx" 
      ? "ONNX INT8 (rápido)" 
      : "Full FP32 (más preciso)";

    // Payload para el Space de Hugging Face
    const hfPayload = [input, modelo_sel, beams, temperature, topP];

    // URL del Space (público, no necesita token)
    const spaceUrl = "https://coyoteMMK-Code2_AI.hf.space/api/predict";

    // Hacer la petición al Space
    const response = await fetch(spaceUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ data: hfPayload }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`HF Space error (${response.status}):`, errorText);

      // Mapear errores comunes del Space
      if (response.status === 503 || errorText.includes("sleep")) {
        return Response.json(
          { error: "El Space está durmiendo. Intenta de nuevo en unos segundos." },
          { status: 503 }
        );
      }

      if (errorText.includes("paused")) {
        return Response.json(
          { error: "El Space está pausado por el owner." },
          { status: 503 }
        );
      }

      throw new Error(`HF Space: ${response.status} ${errorText}`);
    }

    const result = await response.json();
    return Response.json(result);
  } catch (error) {
    console.error("API error:", error);
    return Response.json(
      { error: error.message || "Error en el servidor" },
      { status: 500 }
    );
  }
}

// Endpoint para warmup (mantener el Space despierto)
export async function GET(request) {
  try {
    const url = new URL(request.url);
    const action = url.searchParams.get("action");

    if (action === "warmup") {
      // Llamada mínima para despertar el Space
      const spaceUrl = "https://coyoteMMK-Code2_AI.hf.space/api/predict";

      const response = await fetch(spaceUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          data: ["mov eax, 1", "Full FP32 (más preciso)", 1, 0.2, 0.6],
        }),
      });

      if (!response.ok) {
        console.warn(`Warmup returned ${response.status}`);
        return Response.json(
          { success: false, message: "Warmup attempted" },
          { status: response.status }
        );
      }

      return Response.json({ success: true, message: "Space warmed up" });
    }

    return Response.json({ error: "Action no especificada" }, { status: 400 });
  } catch (error) {
    console.error("Warmup error:", error);
    return Response.json({ success: false, error: error.message }, { status: 500 });
  }
}
