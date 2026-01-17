import re
import time
import gradio as gr

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# -------------------------
# Rutas en tu PC (local)
# -------------------------
MODEL_ONNX_DIR = "../models/full_fp32"
MODEL_FP32_DIR = "../models/onnx_int8_dynamic"

print("🔧 Cargando tokenizers...")
tokenizer_onnx = T5Tokenizer.from_pretrained(MODEL_ONNX_DIR)
tokenizer_fp32 = T5Tokenizer.from_pretrained(MODEL_FP32_DIR)

print("⚙️ Cargando modelos...")
# ONNX INT8 (CPU)
model_onnx = ORTModelForSeq2SeqLM.from_pretrained(MODEL_ONNX_DIR)

# Full fine-tuning (PyTorch)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_fp32 = T5ForConditionalGeneration.from_pretrained(MODEL_FP32_DIR).to(device)
model_fp32.eval()


def normalizar_salida(texto: str) -> str:
    texto = texto.replace("\\n", "\n")
    texto = re.sub(r"<pad>|</s>|<s>", "", texto)
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n[ \t]+", "\n", texto)

    lineas = [ln.lstrip() for ln in texto.splitlines()]
    lineas = [re.sub(r" +", " ", ln).strip() for ln in lineas]
    return "\n".join([ln for ln in lineas if ln != ""]).strip()


def generar(instruccion: str, modelo_sel: str, beams: int):
    if not instruccion or not instruccion.strip():
        return "⚠️ Entrada vacía.", "0.0000 s"

    if modelo_sel == "ONNX INT8 (rápido, CPU)":
        tokenizer = tokenizer_onnx
        model = model_onnx
        is_onnx = True
    else:
        tokenizer = tokenizer_fp32
        model = model_fp32
        is_onnx = False

    inputs = tokenizer(instruccion, return_tensors="pt", truncation=True, padding=True)

    t0 = time.perf_counter()
    if is_onnx:
        # ONNXRuntime trabaja en CPU
        out_ids = model.generate(
            **inputs,
            max_length=512,
            num_beams=int(beams),
            do_sample=False
        )
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_length=512,
                num_beams=int(beams),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
    t1 = time.perf_counter()

    texto = tokenizer.decode(
        out_ids[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )

    salida = normalizar_salida(texto)
    return salida, f"{(t1 - t0):.4f} s"


demo = gr.Interface(
    fn=generar,
    inputs=[
        gr.Textbox(lines=8, label="🧠 Instrucciones en lenguaje natural (múltiples líneas)"),
        gr.Radio(
            choices=["ONNX INT8 (rápido, CPU)", "Full FP32 (PyTorch, CPU/GPU)"],
            value="ONNX INT8 (rápido, CPU)",
            label="⚙️ Modelo"
        ),
        gr.Slider(1, 6, value=4, step=1, label="🔎 Beam search (más beams = más precisión, más lento)")
    ],
    outputs=[
        gr.Textbox(lines=8, label="🖥️ Ensamblado CODE-2"),
        gr.Textbox(label="⏱️ Tiempo de inferencia")
    ],
    title="🟢 Ensamblador CODE-2 (Local)",
    description="Traduce lenguaje natural a ensamblado CODE-2 manteniendo saltos de línea. Selector ONNX INT8 vs FP32.",
    examples=[
        ["Suma r1 y r2 y guarda en r3\nGuarda r3 en la dirección 0345", "ONNX INT8 (rápido, CPU)", 4],
        ["Carga el valor de la dirección 0342 en r1\nGuarda el contenido de r1 en la dirección 0456", "ONNX INT8 (rápido, CPU)", 4],
        ["Guarda el contenido de rF en la dirección 0002", "Full FP32 (PyTorch, CPU/GPU)", 4],
    ],
)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",   # local
        server_port=7860,
        inbrowser=True,            # abre el navegador solo
        share=False                # pon True si quieres link público temporal
    )
