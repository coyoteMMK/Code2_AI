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


# ======================
# Normalización
# ======================
def normalizar(texto: str) -> str:
    texto = texto.replace("\\n", "\n")
    texto = re.sub(r"<pad>|</s>|<s>", "", texto)
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n[ \t]+", "\n", texto)
    lineas = [ln.lstrip() for ln in texto.splitlines()]
    lineas = [re.sub(r" +", " ", ln).strip() for ln in lineas]
    return "\n".join([ln for ln in lineas if ln]).strip()


def cargar_valid(valid_path, n):
    with open(valid_path, encoding="utf-8") as f:
        data = json.load(f)
    return data if n == 0 else data[:n]


# ======================
# Evaluación FP32
# ======================
def evaluar_fp32(model_dir, valid_path, n, beams, max_len):
    print(f"\n🧠 Evaluando PyTorch FP32 → {model_dir}")

    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    data = cargar_valid(valid_path, n)
    preds, refs, tiempos, tokens = [], [], [], []
    em = 0

    for ex in tqdm(data, desc="FP32"):
        inputs = tokenizer(ex["input"], return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        ref = normalizar(ex["output"])

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

        gen = normalizar(tokenizer.decode(out_ids[0], skip_special_tokens=False))
        preds.append(gen)
        refs.append(ref)
        tiempos.append(t1 - t0)
        tokens.append(len(gen.split()))
        em += int(gen == ref)

    return {
        "modelo": "fp32",
        "exact_match": em / len(refs),
        "bleu": bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"],
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
        "tiempo_medio": float(np.mean(tiempos)),
        "tokens_medios": float(np.mean(tokens)),
    }


# ======================
# Evaluación ONNX INT8
# ======================
def evaluar_onnx_int8(model_dir, valid_path, n, beams, max_len):
    print(f"\n⚙️ Evaluando ONNX INT8 dinámico → {model_dir}")

    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = ORTModelForSeq2SeqLM.from_pretrained(model_dir)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    data = cargar_valid(valid_path, n)
    preds, refs, tiempos, tokens = [], [], [], []
    em = 0

    for ex in tqdm(data, desc="ONNX INT8"):
        inputs = tokenizer(ex["input"], return_tensors="pt", truncation=True, padding=True)
        ref = normalizar(ex["output"])

        t0 = time.perf_counter()
        out_ids = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=beams,
            do_sample=False
        )
        t1 = time.perf_counter()

        gen = normalizar(tokenizer.decode(out_ids[0], skip_special_tokens=False))
        preds.append(gen)
        refs.append(ref)
        tiempos.append(t1 - t0)
        tokens.append(len(gen.split()))
        em += int(gen == ref)

    return {
        "modelo": "onnx_int8_dynamic",
        "exact_match": em / len(refs),
        "bleu": bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"],
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
        "tiempo_medio": float(np.mean(tiempos)),
        "tokens_medios": float(np.mean(tokens)),
    }


# ======================
# Main
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", default="models/full_fp32/")
    parser.add_argument("--onnx", default="models/onnx_int8_dynamic/")
    parser.add_argument("--valid", default="data/valid.json")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--out", default="resultados_fp32_vs_onnx_int8.csv")
    args = parser.parse_args()

    r_fp32 = evaluar_fp32(args.fp32, args.valid, args.n, args.beams, args.max_len)
    r_onnx = evaluar_onnx_int8(args.onnx, args.valid, args.n, args.beams, args.max_len)

    df = pd.DataFrame([r_fp32, r_onnx])
    df.to_csv(args.out, index=False)

    print("\n📊 Resultados finales:")
    print(df)
    print(f"\n✅ CSV guardado en: {args.out}")


if __name__ == "__main__":
    main()
