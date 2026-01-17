# code2_cli_multi.py
import os
import re
import time

def limpiar_salida(texto: str) -> str:
    # Convierte '\n' literal a salto real
    texto = texto.replace("\\n", "\n")

    # Quita tokens especiales típicos si aparecen en texto
    texto = re.sub(r"<pad>|</s>|<s>", "", texto)

    # Quita espacios antes/después del salto de línea
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n[ \t]+", "\n", texto)

    # Normaliza espacios múltiples
    texto = re.sub(r" +", " ", texto)

    # Quita indentación al principio de cada línea
    lineas = texto.splitlines()
    lineas = [ln.lstrip() for ln in lineas]

    return "\n".join(lineas).strip()


def leer_entrada_multilinea() -> str:
    print("\n🧠 Instrucción NL > (multilínea: ENTER para nueva línea, ENTER en blanco para ejecutar)")
    lineas = []
    while True:
        try:
            linea = input()
        except EOFError:
            return ""  # por si se cierra stdin
        if linea.strip().lower() == "salir":
            return "__SALIR__"
        if linea == "":
            break
        lineas.append(linea)
    return "\n".join(lineas).strip()


def elegir_modelo(base_dir: str):
    print("🟢 CODE-2 CLI - Selección de modelo")
    print("1) Full fine-tuning (PyTorch)     -> results_FullFineTuning_Final_50k")
    print("2) ONNX INT8 dynamic (Optimum)    -> t5_onnx_int8_dynamic")
    opcion = input("Elige [1/2]: ").strip()

    if opcion not in {"1", "2"}:
        print("⚠️ Opción no válida. Usando 1 por defecto.")
        opcion = "1"

    torch_path = os.path.join(base_dir, "../models/full_fp32")
    onnx_path  = os.path.join(base_dir, "../models/onnx_int8_dynamic")

    if opcion == "1":
        # PyTorch
        import torch
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        if not os.path.isdir(torch_path):
            raise FileNotFoundError(f"No encuentro la carpeta del modelo PyTorch: {torch_path}")

        tokenizer = T5Tokenizer.from_pretrained(torch_path)

        # Si por lo que sea no está guardado el token especial, lo añadimos (no pasa nada si ya está)
        try:
            tokenizer.add_special_tokens({"additional_special_tokens": ["\n"]})
        except Exception:
            pass

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = T5ForConditionalGeneration.from_pretrained(torch_path).to(device)
        model.eval()

        def generar(texto_entrada: str):
            inputs = tokenizer(texto_entrada, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,        # precisión > creatividad
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            t1 = time.perf_counter()

            bruto = tokenizer.decode(out_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return limpiar_salida(bruto), (t1 - t0)

        return "PyTorch FP32 (Full FT)", generar

    else:
        # ONNX INT8 dynamic
        from transformers import T5Tokenizer
        from optimum.onnxruntime import ORTModelForSeq2SeqLM

        if not os.path.isdir(onnx_path):
            raise FileNotFoundError(f"No encuentro la carpeta del modelo ONNX INT8: {onnx_path}")

        tokenizer = T5Tokenizer.from_pretrained(onnx_path)

        try:
            tokenizer.add_special_tokens({"additional_special_tokens": ["\n"]})
        except Exception:
            pass

        model = ORTModelForSeq2SeqLM.from_pretrained(onnx_path)

        def generar(texto_entrada: str):
            inputs = tokenizer(texto_entrada, return_tensors="pt", truncation=True, padding=True)

            t0 = time.perf_counter()
            out_ids = model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                do_sample=False,
            )
            t1 = time.perf_counter()

            bruto = tokenizer.decode(out_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return limpiar_salida(bruto), (t1 - t0)

        return "ONNX INT8 dynamic (CPU)", generar


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nombre_modelo, generar = elegir_modelo(base_dir)

    print(f"\n✅ Modelo seleccionado: {nombre_modelo}")
    print("Escribe 'salir' en cualquier línea para terminar.")

    while True:
        entrada = leer_entrada_multilinea()
        if entrada == "__SALIR__":
            print("🚪 Saliendo.")
            break
        if not entrada:
            print("⚠️ Entrada vacía. Intenta de nuevo.")
            continue

        salida, dt = generar(entrada)
        print("\n🖥️ CODE-2 ensamblado:")
        print(salida)
        print(f"\n⏱️ Tiempo de inferencia: {dt:.4f} s")


if __name__ == "__main__":
    main()
