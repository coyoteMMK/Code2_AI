# scripts/showcase.py
import argparse
import time
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from optimum.onnxruntime import ORTModelForSeq2SeqLM

def normalizar(texto: str) -> str:
    texto = texto.replace("\\n", "\n")
    texto = re.sub(r"<pad>|</s>|<s>", "", texto)
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n[ \t]+", "\n", texto)
    texto = re.sub(r" +", " ", texto)
    lineas = [ln.lstrip() for ln in texto.splitlines()]
    return "\n".join([ln.strip() for ln in lineas if ln.strip() != ""]).strip()

def cargar_modelo(mode, model_dir):
    if mode == "pytorch":
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
        model.eval()
        return tokenizer, model, device, False
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        model = ORTModelForSeq2SeqLM.from_pretrained(model_dir)
        return tokenizer, model, "cpu", True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pytorch", "onnx"], default="pytorch")
    ap.add_argument("--model_dir", default="models/full_fp32")
    ap.add_argument("--beams", type=int, default=4)
    args = ap.parse_args()

    tokenizer, model, device, is_onnx = cargar_modelo(args.mode, args.model_dir)

    print("\n🟢 CODE-2 - MODO TERMINAL")
    print("ENTER dos veces para ejecutar. Escribe 'salir' para terminar.\n")

    while True:
        print("🧠 Instrucción NL > (multilínea):")
        lines = []
        while True:
            s = input()
            if s.strip().lower() == "salir":
                print("🚪 Saliendo.")
                return
            if s == "":
                break
            lines.append(s)

        entrada = "\n".join(lines).strip()
        if not entrada:
            print("⚠️ Entrada vacía.\n")
            continue

        inputs = tokenizer(entrada, return_tensors="pt", truncation=True, padding=True)

        t0 = time.perf_counter()
        if is_onnx:
            out = model.generate(**inputs, max_length=512, num_beams=args.beams, do_sample=False)
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=args.beams,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        t1 = time.perf_counter()

        raw = tokenizer.decode(out[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        print("\n--- Resultado ---")
        print(normalizar(raw))
        print(f"\n⏱️ Tiempo: {(t1 - t0):.4f} s\n")

if __name__ == "__main__":
    main()
