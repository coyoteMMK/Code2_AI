import os
from pathlib import Path
import shutil

from transformers import T5Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

from onnxruntime.quantization import quantize_dynamic, QuantType
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../3.train_final/results_Full_optimizado"
OUT_ONNX_FP32 = "./t5_onnx_fp32"
OUT_ONNX_INT8 = "./t5_onnx_int8_dynamic"

Path(OUT_ONNX_FP32).mkdir(parents=True, exist_ok=True)
Path(OUT_ONNX_INT8).mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("EXPORT TRANSFORMERS -> ONNX Y QUANTIZAR INT8")
print("="*80)
print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Modelo: {MODEL_PATH}")
print("[INFO] Exportando Transformers -> ONNX (FP32) con Optimum...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

# Compatible con versiones donde from_transformers no existe:
ort_model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_PATH, export=True)
ort_model.save_pretrained(OUT_ONNX_FP32)
tokenizer.save_pretrained(OUT_ONNX_FP32)

print(f"[OK] ONNX FP32 guardado en: {OUT_ONNX_FP32}")

# ONNX que suelen generarse
onnx_files = [
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
]

print("[INFO] Cuantizando ONNX con PTQ dinámico INT8 (quantize_dynamic)...")
for fname in onnx_files:
    src = os.path.join(OUT_ONNX_FP32, fname)
    if not os.path.exists(src):
        print(f"[INFO] No existe {fname}, se omite.")
        continue

    dst = os.path.join(OUT_ONNX_INT8, fname)
    quantize_dynamic(
        model_input=src,
        model_output=dst,
        weight_type=QuantType.QInt8
    )
    print(f"[OK] Cuantizado: {fname}")

print("[INFO] Copiando config/tokenizer al directorio INT8...")
for fname in os.listdir(OUT_ONNX_FP32):
    if fname.endswith(".onnx"):
        continue
    src = os.path.join(OUT_ONNX_FP32, fname)
    dst = os.path.join(OUT_ONNX_INT8, fname)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print(f"\n[OK] ONNX INT8 dinámico guardado en: {OUT_ONNX_INT8}")
