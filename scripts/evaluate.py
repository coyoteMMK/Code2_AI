"""
Evaluador interactivo del modelo con controles de temperatura, top_p y num_beams.
Soporta múltiples modelos (Full FP32 y ONNX INT8).
"""
import json
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
except Exception:
    ORTModelForSeq2SeqLM = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

# Configuración
MODELS = {
    "Full FP32": "../models/full_fp32",
    "ONNX INT8": "../models/onnx_int8_dynamic",
}

VALID_DATASET_PATH = "../datasource/test.json"
SAMPLE_SIZE = 100
BATCH_SIZE = 32
TOKENIZER_PATH = "../models/full_fp32"  # Usar el mismo tokenizer para ambos modelos

def _normalize_for_exact_match(text):
    text = text.replace("\n", " SALTO ")
    text = text.replace(" SALTO ", " SALTO ")
    text = " ".join(text.split())
    return text.strip().lower()


def load_model_for_evaluation(model_name: str, model_path: str, device: str = "cuda"):
    """
    Carga los modelos apropiados (PyTorch o ONNX) según el modelo seleccionado.
    Retorna: (modelo, device_usado)
    """
    is_onnx = "ONNX" in model_name
    
    if is_onnx:
        if ORTModelForSeq2SeqLM is None:
            raise RuntimeError(
                "ONNX no disponible. Instala: pip install optimum[onnxruntime] onnxruntime"
            )
        
        # ONNX siempre en CPU para evitar problemas
        print(f"\n[INFO] Cargando modelo ONNX en CPU...")
        model = ORTModelForSeq2SeqLM.from_pretrained(
            model_path,
            provider="CPUExecutionProvider",
            encoder_file_name="encoder_model.onnx",
            decoder_file_name="decoder_model.onnx",
            decoder_with_past_file_name="decoder_with_past_model.onnx",
        )
        return model, "cpu"
    
    # PyTorch (Full FP32)
    print(f"\n[INFO] Cargando modelo PyTorch...")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    model.to(device)
    return model, device

def calculate_bleu(predictions, references):
    try:
        smoothing = SmoothingFunction().method1
        refs = [[ref.split()] for ref in references]
        preds = [pred.split() for pred in predictions]
        bleu = corpus_bleu(refs, preds, smoothing_function=smoothing)
        return round(bleu, 4)
    except Exception:
        return None

def calculate_rouge_l(predictions, references):
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores.append(score["rougeL"].fmeasure)
        return round(float(np.mean(scores)), 4)
    except Exception:
        return None

def calculate_exact_match(predictions, references):
    if not predictions:
        return 0.0
    exact_matches = sum(
        1 for p, r in zip(predictions, references)
        if _normalize_for_exact_match(p) == _normalize_for_exact_match(r)
    )
    return round(exact_matches / len(predictions), 4)

def calculate_optimal_max_length(dataset_raw, tokenizer):
    max_input = 0
    max_output = 0
    for example in dataset_raw:
        input_text = example["input"].replace("\n", " SALTO ")
        output_text = example["output"].replace("\n", " SALTO ")
        input_tokens = tokenizer.encode(input_text, add_special_tokens=True)
        output_tokens = tokenizer.encode(output_text, add_special_tokens=True)
        max_input = max(max_input, len(input_tokens))
        max_output = max(max_output, len(output_tokens))
    return min(512, max_input + 10), min(512, max_output + 10)

def generate_predictions_batched(model, tokenizer, hf_dataset, device, max_gen_len, batch_size, gen_kwargs, model_type="pytorch"):
    model.eval()
    preds = []
    dl = DataLoader(hf_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dl:
            if model_type == "onnx":
                # ONNX no necesita .to(device)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
            else:
                # PyTorch
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_gen_len,
                **gen_kwargs,
            )
            preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return preds

# Cargar tokenizer (igual para todos)
print("\nCargando tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, legacy=False)

# Cargar dataset
print("Cargando dataset...")
dataset_raw = load_dataset("json", data_files={"validation": VALID_DATASET_PATH})
val_raw = dataset_raw["validation"].select(range(min(SAMPLE_SIZE, len(dataset_raw["validation"]))))

max_input_len, max_output_len = calculate_optimal_max_length(val_raw, tokenizer)

def preprocess_function(examples):
    inputs = [text.replace("\n", " SALTO ") for text in examples["input"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        padding="max_length",
        truncation=True,
    )
    return model_inputs

val_tok = val_raw.map(preprocess_function, batched=True, remove_columns=["task", "input", "output"])
val_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

references = [ex["output"].replace("\n", " SALTO ") for ex in val_raw]

print("\n" + "=" * 80)
print("SELECCIONA UN MODELO PARA EVALUAR")
print("=" * 80)

model_options = list(MODELS.items())
for i, (name, path) in enumerate(model_options, 1):
    print(f"  {i}. {name}")
    if not os.path.exists(path):
        print(f"     [ADVERTENCIA] Ruta no encontrada: {path}")

while True:
    try:
        selection = input("\nSelecciona un modelo (número): ").strip()
        model_idx = int(selection) - 1
        if 0 <= model_idx < len(model_options):
            selected_model_name, selected_model_path = model_options[model_idx]
            break
        else:
            print(f"[ERROR] Selecciona un número entre 1 y {len(model_options)}")
    except ValueError:
        print("[ERROR] Entrada inválida. Ingresa un número.")

# Verificar que el modelo existe
if not os.path.exists(selected_model_path):
    print(f"[ERROR] El modelo no existe en: {selected_model_path}")
    exit(1)

# Cargar modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "onnx" if "ONNX" in selected_model_name else "pytorch"

try:
    model, device_used = load_model_for_evaluation(selected_model_name, selected_model_path, device)
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    exit(1)

print("\n" + "=" * 80)
print("EVALUADOR INTERACTIVO - MODELO T5")
print("=" * 80)
print(f"[OK] Modelo cargado: {selected_model_name}")
print(f"[OK] Ruta: {selected_model_path}")
print(f"[OK] Dataset: {SAMPLE_SIZE} ejemplos de validation")
print(f"[OK] Device: {device_used}")
print(f"[CONFIG] Max input length: {max_input_len}")
print(f"[CONFIG] Max output length: {max_output_len}")
print(f"[CONFIG] Tipo de modelo: {model_type}")

# Valores por defecto
default_num_beams = 1
default_temperature = 0.2
default_top_p = 0.6

history = []

while True:
    print("\n" + "=" * 80)
    print("PARÁMETROS DE GENERACIÓN")
    print("=" * 80)
    
    try:
        num_beams = input(f"num_beams [{default_num_beams}]: ").strip()
        num_beams = int(num_beams) if num_beams else default_num_beams
        default_num_beams = num_beams
        
        temperature = input(f"temperature [{default_temperature}]: ").strip()
        temperature = float(temperature) if temperature else default_temperature
        default_temperature = temperature
        
        top_p = input(f"top_p [{default_top_p}]: ").strip()
        top_p = float(top_p) if top_p else default_top_p
        default_top_p = top_p
    except ValueError:
        print("[ERROR] Entrada inválida. Por favor ingresa números válidos.")
        continue
    
    print(f"\n[INFO] Generando predicciones...")
    print(f"   num_beams={num_beams}, temperature={temperature}, top_p={top_p}")
    
    gen_kwargs = {
        "num_beams": num_beams,
        "early_stopping": True,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    preds = generate_predictions_batched(
        model, tokenizer, val_tok, device_used,
        max_gen_len=max_output_len,
        batch_size=BATCH_SIZE,
        gen_kwargs=gen_kwargs,
        model_type=model_type,
    )
    
    bleu = calculate_bleu(preds, references)
    rouge_l = calculate_rouge_l(preds, references)
    em = calculate_exact_match(preds, references)
    
    print("\n" + "=" * 80)
    print("RESULTADOS")
    print("=" * 80)
    print(f"[RESULTADO] Exact Match: {em} ({int(em * len(preds))}/{len(preds)})")
    print(f"[METRICA] BLEU: {bleu}")
    print(f"[METRICA] ROUGE-L: {rouge_l}")
    
    history.append({
        "num_beams": num_beams,
        "temperature": temperature,
        "top_p": top_p,
        "em": em,
        "bleu": bleu,
        "rouge_l": rouge_l,
    })
    
    # Submenu después de generar
    while True:
        print("\n" + "=" * 80)
        print("QUE HACER")
        print("=" * 80)
        print("  (c) Continuar probando con otros parámetros")
        print("  (a) Ver algunos ejemplos")
        print("  (h) Ver historial completo")
        print("  (g) Guardar historial a JSON")
        print("  (s) Salir")
        
        cmd = input("\n>>> ").strip().lower()
        
        if cmd == "c":
            break  # Sale del submenu y pide nuevos parámetros
        
        elif cmd == "a":
            # Visor interactivo de ejemplos
            example_idx = 0
            while True:
                print("\n" + "=" * 80)
                print(f"EJEMPLO {example_idx + 1}/{len(preds)}")
                print("=" * 80)
                
                # Mostrar Input con SALTO como salto de línea real
                input_display = val_raw[example_idx]['input'].replace("\n", "\n")
                print(f"\nInput:")
                print(input_display)
                
                # Mostrar Expected con SALTO como salto de línea real
                expected_display = references[example_idx].replace(" SALTO ", "\n")
                print(f"\nExpected:")
                print(expected_display)
                
                # Mostrar Predicted con SALTO como salto de línea real
                predicted_display = preds[example_idx].replace(" SALTO ", "\n")
                print(f"\nPredicted:")
                print(predicted_display)
                
                match = _normalize_for_exact_match(preds[example_idx]) == _normalize_for_exact_match(references[example_idx])
                print(f"\nMatch: {'SI' if match else 'NO'}")
                
                print("\n" + "-" * 80)
                print("Opciones: (a)nterior  (s)iguiente  (ir) ir a #  (volver) volver al menu")
                nav_cmd = input(">>> ").strip().lower()
                
                if nav_cmd == "a":
                    if example_idx > 0:
                        example_idx -= 1
                    else:
                        print("[INFO] Ya estás en el primer ejemplo")
                elif nav_cmd == "s":
                    if example_idx < len(preds) - 1:
                        example_idx += 1
                    else:
                        print("[INFO] Ya estás en el último ejemplo")
                elif nav_cmd.startswith("ir"):
                    try:
                        parts = nav_cmd.split()
                        if len(parts) >= 2:
                            num = int(parts[1]) - 1
                            if 0 <= num < len(preds):
                                example_idx = num
                            else:
                                print(f"[ERROR] Ingresa un número entre 1 y {len(preds)}")
                    except:
                        print("[ERROR] Uso: ir <numero>")
                elif nav_cmd == "volver":
                    break
                else:
                    print("[ERROR] Comando no reconocido")
        
        elif cmd == "h":
            print("\n" + "=" * 80)
            print("HISTORIAL DE EJECUCIONES")
            print("=" * 80)
            for i, result in enumerate(history, 1):
                print(f"\n{i}. num_beams={result['num_beams']}, temp={result['temperature']}, top_p={result['top_p']}")
                print(f"   EM={result['em']}, BLEU={result['bleu']}, ROUGE-L={result['rouge_l']}")
        
        elif cmd == "g":
            output_file = "eval_interactive_history.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "model_path": MODEL_PATH,
                    "total_examples": len(preds),
                    "max_input_len": max_input_len,
                    "max_output_len": max_output_len,
                    "history": history,
                }, f, indent=2)
            print(f"[OK] Guardado en: {output_file}")
        
        elif cmd == "s":
            print("\n[INFO] Saliendo...")
            exit()
        
        else:
            print("[ERROR] Comando no reconocido. Intenta de nuevo.")
