# scripts/training.py
import argparse
import time
import subprocess
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from datasets import load_dataset


def get_used_gpu_memory():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return int(out.decode("utf-8").split("\n")[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()

    # 🔹 CLAVE: modelo inicial (HF o ruta local)
    ap.add_argument(
        "--model_path",
        default="t5-small",
        help="Modelo base: nombre HF (t5-small) o ruta local entrenada"
    )

    ap.add_argument("--train_json", default="data/train.json")
    ap.add_argument("--valid_json", default="data/valid.json")
    ap.add_argument("--save_dir", default="models/full_fp32")
    ap.add_argument("--seed", type=int, default=42)

    # hiperparámetros finales
    ap.add_argument("--lr", type=float, default=0.0004994714149356016)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--weight_decay", type=float, default=0.0061472284302228454)
    ap.add_argument("--epochs", type=int, default=3)

    ap.add_argument("--max_input_len", type=int, default=402)
    ap.add_argument("--max_output_len", type=int, default=511)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument(
        "--preserve_newlines",
        action="store_true",
        help="Preserva saltos de linea reemplazandolos por SALTO antes de tokenizar",
    )

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("🧠 Cargando tokenizer desde:", args.model_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path, legacy=False)

    if args.preserve_newlines and "SALTO" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["SALTO"]})

    print("🧠 Cargando modelo desde:", args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(device)
    model.resize_token_embeddings(len(tokenizer))

    print("📦 Cargando dataset...")
    dataset_raw = load_dataset(
        "json",
        data_files={"train": args.train_json, "validation": args.valid_json},
    )

    def preprocess(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        if args.preserve_newlines:
            inputs = [text.replace("\n", " SALTO ") for text in inputs]
            outputs = [text.replace("\n", " SALTO ") for text in outputs]

        x = tokenizer(
            inputs,
            max_length=args.max_input_len,
            truncation=True,
            padding="max_length",
        )
        y = tokenizer(
            outputs,
            max_length=args.max_output_len,
            truncation=True,
            padding="max_length",
        )
        x["labels"] = y["input_ids"]
        return x

    dataset = dataset_raw.map(
        preprocess,
        batched=True,
        remove_columns=["task", "input", "output"],
    )

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        fp16=args.fp16 and (device == "cuda"),
        seed=args.seed,
        gradient_accumulation_steps=args.grad_accum,
        report_to=[],
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Entrenando modelo...")
    start = time.time()
    trainer.train()
    eval_result = trainer.evaluate()
    end = time.time()

    gpu_used = get_used_gpu_memory()
    train_loss = next(
        (log["loss"] for log in reversed(trainer.state.log_history) if "loss" in log),
        None,
    )
    loss_gap = (
        float(train_loss) - eval_result["eval_loss"]
        if train_loss is not None
        else None
    )

    print("\n=== RESULTADOS ENTRENAMIENTO ===")
    print(f"📉 Eval Loss: {eval_result['eval_loss']:.6f}")
    if train_loss is not None:
        print(f"🧪 Train Loss: {train_loss:.6f}")
    if loss_gap is not None:
        print(f"📊 Loss Gap: {loss_gap:.6f}")
    print(f"💻 GPU usada: {gpu_used} MB" if gpu_used else "💻 GPU usada: N/A")
    print(f"⏱️ Duración: {round((end - start)/60, 2)} min")

    print("\n💾 Guardando modelo y tokenizer...")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"✅ Guardado en: {args.save_dir}")


if __name__ == "__main__":
    main()
