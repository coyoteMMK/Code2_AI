import os
import re
import time
import gradio as gr

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from optimum.onnxruntime import ORTModelForSeq2SeqLM


# =========================
# RUTAS (repo)
# =========================
MODEL_FP32_DIR = os.getenv("MODEL_FP32_DIR", "models/full_fp32/")
MODEL_ONNX_INT8_DIR = os.getenv("MODEL_ONNX_INT8_DIR", "models/onnx_int8_dynamic/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# NORMALIZACIÓN
# =========================
def normalizar_salida(texto: str) -> str:
    # 1) convierte "\n" literal a salto real
    texto = texto.replace("\\n", "\n")

    # 2) elimina tokens típicos si aparecen
    texto = re.sub(r"<pad>|</s>|<s>", "", texto)

    # 3) quita espacios antes/después del salto de línea
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n[ \t]+", "\n", texto)

    # 4) quita sangría al inicio de cada línea (evita " ST ...")
    lineas = [ln.lstrip() for ln in texto.splitlines()]

    # 5) colapsa espacios múltiples dentro de cada línea
    lineas = [re.sub(r" +", " ", ln).strip() for ln in lineas]

    # 6) elimina líneas vacías sueltas
    return "\n".join([ln for ln in lineas if ln != ""]).strip()


# =========================
# CARGA DE MODELOS (una sola vez)
# =========================
print("🔧 Cargando tokenizers...")
tokenizer_fp32 = T5Tokenizer.from_pretrained(MODEL_FP32_DIR)
tokenizer_onnx = T5Tokenizer.from_pretrained(MODEL_ONNX_INT8_DIR)

print(f"⚙️ Cargando modelo FP32 (PyTorch) en {DEVICE}...")
model_fp32 = T5ForConditionalGeneration.from_pretrained(MODEL_FP32_DIR).to(DEVICE)
model_fp32.eval()

print("⚙️ Cargando modelo ONNX INT8 (CPU)...")
model_onnx = ORTModelForSeq2SeqLM.from_pretrained(MODEL_ONNX_INT8_DIR)


# =========================
# INFERENCIA
# =========================
@torch.inference_mode()
def generar(instruccion: str, modelo_sel: str, beams: int):
    if not instruccion or not instruccion.strip():
        return "⚠️ Entrada vacía.", "0.0000 s"

    # Selección
    if modelo_sel == "ONNX INT8 dinámico (CPU)":
        tokenizer = tokenizer_onnx
        model = model_onnx
        is_onnx = True
    else:
        tokenizer = tokenizer_fp32
        model = model_fp32
        is_onnx = False

    # Tokenizar
    inputs = tokenizer(instruccion, return_tensors="pt", truncation=True, padding=True)

    # Inferencia + tiempo
    t0 = time.perf_counter()

    if is_onnx:
        # ORTModel: CPU (inputs torch ok)
        out_ids = model.generate(
            **inputs,
            max_length=512,
            num_beams=beams,
            do_sample=False,
        )
    else:
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out_ids = model.generate(
            **inputs,
            max_length=512,
            num_beams=beams,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    t1 = time.perf_counter()

    texto = tokenizer.decode(
        out_ids[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    salida = normalizar_salida(texto)

    return salida, f"{(t1 - t0):.4f} s"


# =========================
# UI (Gradio)
# =========================
demo = gr.Interface(
    fn=generar,
    inputs=[
        gr.Textbox(lines=8, label="🧠 Instrucciones en lenguaje natural (múltiples líneas)"),
        gr.Radio(
            choices=["ONNX INT8 dinámico (CPU)", "PyTorch FP32 (GPU/CPU)"],
            value="ONNX INT8 dinámico (CPU)",
            label="⚙️ Modelo",
        ),
        gr.Slider(1, 6, value=4, step=1, label="🔎 Beam search (más beams = más precisión, más lento)"),
    ],
    outputs=[
        gr.Textbox(lines=10, label="🖥️ Ensamblado CODE-2"),
        gr.Textbox(label="⏱️ Tiempo de inferencia"),
    ],
    title="🟢 CODE-2 Translator (FP32 vs ONNX INT8)",
    description="Traduce lenguaje natural → ensamblado CODE-2 preservando saltos de línea y midiendo el tiempo de inferencia.",
    examples=[
        ["Suma r1 y r2 y guarda en r3\nGuarda r3 en la dirección 0345", "ONNX INT8 dinámico (CPU)", 4],
        ["Carga el valor de la dirección 0342 en r1\nGuarda el contenido de r1 en la dirección 0456", "ONNX INT8 dinámico (CPU)", 4],
        ["Guarda el contenido de rF en la dirección 0002", "PyTorch FP32 (GPU/CPU)", 4],
    ],
)

if __name__ == "__main__":
    demo.launch()
