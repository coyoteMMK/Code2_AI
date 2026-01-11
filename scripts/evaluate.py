# scripts/evaluate.py
import argparse
import time
import json
import re
import numpy as np
from tqdm import tqdm

import evaluate
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from optimum.onnxruntime import ORTModelForSeq2SeqLM

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

def normalizar(texto: str) -> str:
    texto = texto.replace("\\n", "\n")
    texto = re.sub(r"<pad>|</s>|<s>", "", texto)
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n[ \t]+", "\n", texto)
    texto = re.sub(r" +", " ", texto)
    # quita sangría por línea
    lineas = [ln.lstrip() for ln in texto.splitlines()]
    return "\n".join([ln.strip() for ln in lineas if ln.strip() != ""]).strip()

def cargar_valid(path, n=None):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if n is not None:
        data = data[:n]
    return data

def evaluar_pytorch(model_dir, valid_path, n, device, beams):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    data = cargar_valid(valid_path, n=n)
    preds, refs, tiempos = [], [], []
    em = 0

    for ex in tqdm(data, desc="pytorch"):
        entrada = ex["input"]
        ref = normalizar(ex["output"])

        inputs = tokenizer(entrada, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=512,
                num_beams=beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        t1 = time.perf_counter()

        gen = tokenizer.decode(out[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        gen = normalizar(gen)

        preds.append(gen)
        refs.append(ref)
        tiempos.append(t1 - t0)
        if gen == ref:
            em += 1

    return {
        "modelo": "pytorch_fp32",
        "exact_match": em / len(refs),
        "bleu": bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"],
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
        "tiempo_medio": float(np.mean(tiempos)),
        "tokens_medios": float(np.mean([len(p.split()) for p in preds])),
    }

def evaluar_onnx(model_dir, valid_path, n, beams):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = ORTModelForSeq2SeqLM.from_pretrained(model_dir)

    data = cargar_valid(valid_path, n=n)
    preds, refs, tiempos = [], [], []
    em = 0

    for ex in tqdm(data, desc="onnx"):
        entrada = ex["input"]
        ref = normalizar(ex["output"])

        inputs = tokenizer(entrada, return_tensors="pt", truncation=True, padding=True)

        t0 = time.perf_counter()
        out = model.generate(**inputs, max_length=512, num_beams=beams, do_sample=False)
        t1 = time.perf_counter()

        gen = tokenizer.decode(out[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        gen = normalizar(gen)

        preds.append(gen)
        refs.append(ref)
        tiempos.append(t1 - t0)
        if gen == ref:
            em += 1

    return {
        "modelo": "onnx",
        "exact_match": em / len(refs),
        "bleu": bleu.compute(predictions=preds, references=[[r] for r in refs])["bleu"],
        "rougeL": rouge.compute(predictions=preds, references=refs)["rougeL"],
        "tiempo_medio": float(np.mean(tiempos)),
        "tokens_medios": float(np.mean([len(p.split()) for p in preds])),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", default="data/valid.json")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--beams", type=int, default=4)

    ap.add_argument("--pytorch_model", default="models/full_fp32")
    ap.add_argument("--onnx_model", default=None, help="Si se pasa, evalúa ONNX también")

    ap.add_argument("--out_csv", default="results_eval.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    resultados = []
    resultados.append(evaluar_pytorch(args.pytorch_model, args.valid, args.n, device=device, beams=args.beams))

    if args.onnx_model:
        resultados.append(evaluar_onnx(args.onnx_model, args.valid, args.n, beams=args.beams))

    import pandas as pd
    df = pd.DataFrame(resultados)
    df.to_csv(args.out_csv, index=False)

    print("\n📊 Resultados:")
    print(df)
    print(f"\n✅ Guardado: {args.out_csv}")

if __name__ == "__main__":
    main()
