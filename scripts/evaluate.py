# scripts/evaluate.py
import argparse
import json
import time
import re
import numpy as np
from tqdm import tqdm

import torch
import evaluate
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from optimum.onnxruntime import ORTModelForSeq2SeqLM


# ===== Normalización (para Exact Match realista) =====
def normalizar(texto: str) -> str:
    # convierte "\n" literal a salto real
    texto = texto.replace("\\n", "\n")

    # quita tokens especiales si aparecen
    texto = re.sub(r"<pad>|</s>|<s>", "", texto)

    # limpia espacios antes/después de saltos
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n[ \t]+", "\n", texto)

    # quita sangría al principio de cada línea
    lineas = [ln.lstrip() for ln in texto.splitlines()]

    # colapsa espacios múltiples por línea
    lineas = [re.sub(r" +", " ", ln).strip() for ln in lineas]

    # elimina líneas vacías
    return "\n".join([ln for ln in lineas if ln != ""]).strip()


def load_valid(valid_path: str, n_ejemplos: int):
    with open(valid_path, encoding="utf-8") as f:
        data = json.load(f)
    if n_ejemplos > 0:
        data = data[:n_ejemplos]
    return data


def evaluar_fp32(model_dir: str, valid_path: str, n: int, beams: int, max_len: int):
    print(f"\n🧠 Evaluando PyTorch FP32: {model_dir}")

    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    data = load_valid(valid_path, n)

    preds, refs = [], []
    tiempos = []
    em = 0
    tokens_out = []

    for ex in tqdm(data, desc="fp32"):
        entrada = ex["input"]
        ref = normalizar(ex["output"])

        inputs = tokenizer(entrada, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_length=max_len,
                num_beams=beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        t1 = time.perf_counter()

        gen = tokenizer.decode(out_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        gen = normalizar(gen)

        preds.append(gen)
        refs.append(ref)
        tiempos.append(t1 - t0)
        tokens_out.append(len(gen.split()))

        if gen == ref:
            em += 1

    return {
        "modelo": "torch_fp32",
        "exact_match": em / len(refs),
        "bleu": bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"],
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
        "tiempo_medio": float(np.mean(tiempos)),
        "tokens_medios": float(np.mean(tokens_out)),
    }


def evaluar_onnx_int8(model_dir: str, valid_path: str, n: int, beams: int, max_len: int):
    print(f"\n⚙️ Evaluando ONNX INT8 dinámico: {model_dir}")

    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = ORTModelForSeq2SeqLM.from_pretrained(model_dir)  # CPU

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    data = load_valid(valid_path, n)

    preds, refs = [], []
    tiempos = []
    em = 0
    tokens_out = []

    for ex in tqdm(data, desc="onnx_int8"):
        entrada = ex["input"]
        ref = normalizar(ex["output"])

        inputs = tokenizer(entrada, return_tensors="pt", truncation=True, padding=True)

        t0 = time.perf_counter()
        out_ids = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=beams,
            do_sample=False
        )
        t1 = time.perf_counter()

        gen = tokenizer.decode(out_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        gen = normalizar(gen)

        preds.append(gen)
        refs.append(ref)
        tiempos.append(t1 - t0)
        tokens_out.append(len(gen.split()))

        if gen == ref:
            em += 1

    return {
        "modelo": "onnx_int8_dynamic",
        "exact_match": em / len(refs),
        "bleu": bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"],
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
        "tiempo_medio": float(np.mean(tiempos)),
        "tokens_medios": float(np.mean(tokens_out)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32_dir", required=True, help="Carpeta del modelo PyTorch FP32")
    ap.add_argument("--onnx_int8_dir", required=True, help="Carpeta del modelo ONNX INT8 dinámico")
    ap.add_argument("--valid", required=True, help="Ruta a valid.json")
    ap.add_argument("--n", type=int, default=2000, help="Número de ejemplos (0 = todos)")
    ap.add_argument("--beams", type=int, default=4, help="num_beams para generate()")
    ap.add_argument("--max_len", type=int, default=512, help="max_length para generate()")
    ap.add_argument("--out", default="resultados_fp32_vs_onnx_int8.csv", help="CSV de salida")
    args = ap.parse_args()

    res_fp32 = evaluar_fp32(args.fp32_dir, args.valid, args.n, args.beams, args.max_len)
    res_int8 = evaluar_onnx_int8(args.onnx_int8_dir, args.valid, args.n, args.beams, args.max_len)

    df = pd.DataFrame([res_fp32, res_int8])
    df.to_csv(args.out, index=False)

    print("\n📊 Resultados comparativos:")
    print(df)
    print(f"\n✅ Guardado: {args.out}")


if __name__ == "__main__":
    main()
