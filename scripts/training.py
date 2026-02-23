import time
import json
import torch
import pandas as pd
import os
import subprocess
import numpy as np
from transformers import (
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    T5Tokenizer
)
from torch.utils.data import DataLoader
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

# Configuración
MODEL_NAME = "t5-small"
TRAIN_DATASET_PATH = "../datasource/train.json"
VALID_DATASET_PATH = "../datasource/valid.json"
SAVE_DIR = "./results_Full_optimizado"  # Directorio para guardar el modelo
OUTPUT_FILE = "resultados_full_optimizado.csv"
RESULTS_TXT = "resultados_full_optimizado.txt"
RESULTS_JSON = "resultados_full_optimizado.json"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4



def get_used_gpu_memory():
    """Obtiene la memoria GPU usada en MB"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return int(out.decode("utf-8").split("\n")[0])
    except Exception:
        return None


def calculate_bleu(predictions, references):
    """Calcula BLEU score"""
    try:
        smoothing = SmoothingFunction().method1
        refs = [[ref.split()] for ref in references]
        preds = [pred.split() for pred in predictions]
        bleu = corpus_bleu(refs, preds, smoothing_function=smoothing)
        return round(bleu, 4)
    except Exception as e:
        print(f"Error calculando BLEU: {e}")
        return None


def calculate_rouge_l(predictions, references):
    """Calcula ROUGE-L score"""
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = []
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores.append(score['rougeL'].fmeasure)
        return round(np.mean(scores), 4)
    except Exception as e:
        print(f"Error calculando ROUGE-L: {e}")
        return None


def _normalize_for_exact_match(text):
    text = text.replace("\n", " SALTO ")
    text = text.replace(" SALTO ", " SALTO ")
    # Normalize whitespace and case to reduce false mismatches
    text = " ".join(text.split())
    return text.strip().lower()


def calculate_exact_match(predictions, references):
    """Calcula Exact Match score (normalizado)"""
    exact_matches = sum(
        1
        for p, r in zip(predictions, references)
        if _normalize_for_exact_match(p) == _normalize_for_exact_match(r)
    )
    return round(exact_matches / len(predictions), 4)


from torch.utils.data import DataLoader

def generate_predictions_batched(model, tokenizer, hf_dataset, device,
                                 max_gen_len=128, batch_size=16, num_beams=2, collate_fn=None):
    model.eval()
    preds = []

    dl = DataLoader(hf_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for i, batch in enumerate(dl, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_gen_len,
                num_beams=num_beams,
                early_stopping=True
            )

            preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

            if i % 25 == 0:
                print(f"  🔮 Generación: batch {i}/{len(dl)}")

    return preds

# Crear directorio de guardado si no existe
os.makedirs(SAVE_DIR, exist_ok=True)

# Cargar tokenizer
print("🧠 Cargando tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

# Agregar token para saltos de línea (como token normal, no especial)
if "SALTO" not in tokenizer.get_vocab():
    tokenizer.add_tokens(["SALTO"])
    print("✅ Token 'SALTO' agregado al tokenizer")

# Cargar dataset
print("📦 Cargando dataset...")
dataset_raw = load_dataset("json", data_files={"train": TRAIN_DATASET_PATH, "validation": VALID_DATASET_PATH})

# Preprocesamiento del dataset (padding dinámico)
def preprocess_function(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    
    # Reemplazar saltos de línea con token SALTO
    inputs = [text.replace("\n", " SALTO ") for text in inputs]
    outputs = [text.replace("\n", " SALTO ") for text in outputs]
    
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        padding=False,
        truncation=True
    )

    labels = tokenizer(
        outputs,
        max_length=512,
        padding=False,
        truncation=True
    )

    # 🔥 MUY IMPORTANTE: convertir PAD a -100 para que no afecte la loss
    labels_ids = labels["input_ids"]
    labels_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in labels_ids
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs


dataset = dataset_raw.map(preprocess_function, batched=True, remove_columns=["task", "input", "output"])

# Configurar modelo para Full Fine-Tuning
print("🧠 Cargando modelo...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

device = "cuda" 
model.to(device)

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    weight_decay=WEIGHT_DECAY,

    # ✅ Warmup real (5%)
    warmup_ratio=0.03,


    gradient_accumulation_steps=1,
    gradient_checkpointing=True,

)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Iniciar entrenamiento
print("🚀 Iniciando entrenamiento...")
start_time = time.time()
result = trainer.train()
end_time = time.time()

# Evaluación final
print("🧪 Evaluando modelo...")
eval_result = trainer.evaluate()
validation_loss = eval_result.get("eval_loss", "N/A")

samples_per_second = result.metrics["train_samples_per_second"]

# Obtener training loss final
train_loss = next(
    (log["loss"] for log in reversed(trainer.state.log_history) if "loss" in log),
    None,
)

# Calcular loss gap
loss_gap = (
    float(train_loss) - validation_loss
    if (train_loss is not None and validation_loss != "N/A")
    else "N/A"
)

# Obtener memoria GPU usada
gpu_used = get_used_gpu_memory()

# Calcular tiempo total
total_time = round((end_time - start_time) / 60, 2)

print("🔮 Generando predicciones (usando 100 muestras para cálculo de métricas)...")

# Generar predicciones y calcular métricas (solo en 100 muestras como en optuna_full)
val_subset = dataset["validation"].select(range(min(100, len(dataset["validation"]))))
val_subset.set_format(type="torch", columns=["input_ids", "attention_mask"])

predictions = generate_predictions_batched(
    model, tokenizer, val_subset, device,
    max_gen_len=512,
    batch_size=16,
    num_beams=2,
    collate_fn=data_collator
)

val_subset_raw = dataset_raw["validation"].select(range(min(100, len(dataset_raw["validation"]))))
references = [ex["output"].replace("\n", " SALTO ") for ex in val_subset_raw]


# Calcular métricas
bleu_score = calculate_bleu(predictions, references)
rouge_l_score = calculate_rouge_l(predictions, references)
exact_match_score = calculate_exact_match(predictions, references)

# Guardar modelo y tokenizador
print("💾 Guardando modelo y tokenizador...")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Guardar métricas en CSV
results = {
    "Método": "Full Fine-Tuning",
    "Training Loss Final": train_loss,
    "Validation Loss Final": validation_loss,
    "Loss Gap": loss_gap,
    "BLEU": bleu_score,
    "ROUGE-L": rouge_l_score,
    "Exact Match": exact_match_score,
    "Tiempo Total (min)": total_time,
    "Samples por Segundo": samples_per_second,
    "GPU usada (MB)": gpu_used,
    "Max Length": 512,
    "Padding": "Dynamic",
    "Weight Decay": training_args.weight_decay,
    "Gradient Accumulation Steps": training_args.gradient_accumulation_steps,
}
df = pd.DataFrame([results])
df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not os.path.exists(OUTPUT_FILE))

# Guardar resumen en JSON
results_json = {
    "method": "Full Fine-Tuning",
    "metrics": {
        "train_loss": train_loss,
        "eval_loss": validation_loss,
        "loss_gap": loss_gap if loss_gap != "N/A" else None,
        "bleu": bleu_score,
        "rouge_l": rouge_l_score,
        "exact_match": exact_match_score,
        "samples_per_second": samples_per_second,
        "gpu_used_mb": gpu_used,
        "total_time_min": total_time,
    },
    "config": {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": training_args.weight_decay,
        "warmup_ratio": training_args.warmup_ratio,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "max_length": 512,
        "padding": "dynamic",
        "fp16": torch.cuda.is_available(),
        "device": device,
    },
}

with open(RESULTS_JSON, "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2)

# Guardar métricas en archivo TXT
with open(RESULTS_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("RESULTADOS DE ENTRENAMIENTO - FULL FINE-TUNING\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("📊 PÉRDIDAS:\n")
    f.write(f"  Training Loss: {train_loss:.6f}\n" if train_loss else "  Training Loss: N/A\n")
    f.write(f"  Evaluation Loss: {validation_loss:.6f}\n" if validation_loss != "N/A" else "  Evaluation Loss: N/A\n")
    if loss_gap != "N/A":
        f.write(f"  Loss Gap: {loss_gap:.6f}\n\n")
    else:
        f.write(f"  Loss Gap: N/A\n\n")
    
    f.write("📈 MÉTRICAS DE GENERACIÓN:\n")
    f.write(f"  BLEU: {bleu_score}\n" if bleu_score else "  BLEU: N/A\n")
    f.write(f"  ROUGE-L: {rouge_l_score}\n" if rouge_l_score else "  ROUGE-L: N/A\n")
    f.write(f"  Exact Match: {exact_match_score}\n\n" if exact_match_score else "  Exact Match: N/A\n\n")
    
    f.write("💻 RECURSOS:\n")
    f.write(f"  GPU usada: {gpu_used} MB\n" if gpu_used else "  GPU usada: N/A\n")
    f.write(f"  Duración total: {total_time} minutos\n")
    f.write(f"  Samples por segundo: {samples_per_second:.2f}\n\n")
    
    f.write("⚙️  CONFIGURACIÓN:\n")
    f.write(f"  Batch Size: {BATCH_SIZE}\n")
    f.write(f"  Epochs: {EPOCHS}\n")
    f.write(f"  Learning Rate: {LEARNING_RATE}\n")
    f.write(f"  Weight Decay: {training_args.weight_decay}\n")
    f.write(f"  Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}\n")
    f.write(f"  Max Length: 512\n")
    f.write(f"  Padding: Dynamic\n")
    f.write(f"  FP16 Enabled: {torch.cuda.is_available()}\n")
    f.write(f"  Device: {device}\n\n")
    
    f.write("=" * 60 + "\n")

print("\n✅ Modelo y tokenizador guardados en:", SAVE_DIR)
print(f"✅ Resultados guardados en: {RESULTS_TXT}")
print("\n=== RESUMEN DE RESULTADOS ===")
print(f"📉 Training Loss: {train_loss:.6f}" if train_loss else "📉 Training Loss: N/A")
print(f"📉 Evaluation Loss: {validation_loss:.6f}" if validation_loss != "N/A" else "📉 Evaluation Loss: N/A")
if loss_gap != "N/A":
    print(f"📊 Loss Gap: {loss_gap:.6f}")
print(f"🎯 BLEU: {bleu_score}" if bleu_score else "🎯 BLEU: N/A")
print(f"🎯 ROUGE-L: {rouge_l_score}" if rouge_l_score else "🎯 ROUGE-L: N/A")
print(f"🎯 Exact Match: {exact_match_score}" if exact_match_score else "🎯 Exact Match: N/A")
print(f"⚙️  Weight Decay: {training_args.weight_decay}")
print(f"⚙️  Gradient Accumulation Steps: {training_args.gradient_accumulation_steps}")
print(f"💻 GPU usada: {gpu_used} MB" if gpu_used else "💻 GPU usada: N/A")
print(f"⏱️  Duración: {total_time} minutos")

# Verificación: Recargar modelo y tokenizador
print("\n🔍 Verificando que el modelo guardado se puede cargar correctamente...")

try:
    loaded_model = T5ForConditionalGeneration.from_pretrained(SAVE_DIR)
    loaded_tokenizer = T5Tokenizer.from_pretrained(SAVE_DIR)
    print("✅ Modelo y tokenizador recargados exitosamente.")
except Exception as e:
    print("❌ Error al recargar el modelo/tokenizador:", e)
