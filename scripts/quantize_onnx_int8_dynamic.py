# scripts/quantize_onnx_int8_dynamic.py
import argparse
import os
import shutil

from transformers import T5Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# onnxruntime quantization
from onnxruntime.quantization import quantize_dynamic, QuantType


def find_onnx_files(folder: str):
    """Devuelve lista de ficheros .onnx dentro de una carpeta (no recursivo)."""
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".onnx")]


def copy_tokenizer_and_config(src_dir: str, dst_dir: str):
    """
    Copia config/tokenizer al directorio destino para que ORTModelForSeq2SeqLM
    pueda cargarse como carpeta HuggingFace.
    """
    os.makedirs(dst_dir, exist_ok=True)

    # Archivos típicos
    candidates = [
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "generation_config.json",
    ]

    for name in candidates:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))

    # También copiamos added_tokens.json si existe
    extra = os.path.join(src_dir, "added_tokens.json")
    if os.path.exists(extra):
        shutil.copy2(extra, os.path.join(dst_dir, "added_tokens.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        required=True,
        help="Carpeta del modelo PyTorch (e.g. ../5.Train_final/results_FullFineTuning_Final_50k)"
    )
    ap.add_argument(
        "--out_onnx_fp32",
        default="./t5_onnx_fp32",
        help="Salida ONNX FP32 (carpeta)"
    )
    ap.add_argument(
        "--out_onnx_int8",
        default="./t5_onnx_int8_dynamic",
        help="Salida ONNX INT8 dinámico (carpeta)"
    )
    ap.add_argument(
        "--opset",
        type=int,
        default=14,
        help="Opset ONNX (14 suele funcionar bien con T5)"
    )
    ap.add_argument(
        "--per_channel",
        action="store_true",
        help="Activa cuantización per-channel (a veces mejora precisión, a veces no cambia)."
    )
    args = ap.parse_args()

    os.makedirs(args.out_onnx_fp32, exist_ok=True)
    os.makedirs(args.out_onnx_int8, exist_ok=True)

    # 1) Exportar a ONNX FP32
    print("🧠 Exportando Transformers -> ONNX (FP32) con Optimum...")
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(
        args.model_dir,
        export=True,
        opset=args.opset
    )
    ort_model.save_pretrained(args.out_onnx_fp32)

    # Guardar tokenizer en carpeta ONNX FP32
    print("🔧 Guardando tokenizer/config...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.out_onnx_fp32)

    print(f"✅ ONNX FP32 guardado en: {args.out_onnx_fp32}")

    # 2) Cuantizar dinámico INT8
    print("⚙️ Cuantizando ONNX con PTQ dinámico INT8 (quantize_dynamic)...")

    onnx_files = find_onnx_files(args.out_onnx_fp32)
    if not onnx_files:
        raise FileNotFoundError(
            f"No se encontraron .onnx en {args.out_onnx_fp32}. "
            "Revisa que el export haya generado archivos ONNX."
        )

    # Cuantizamos cada .onnx (en seq2seq suele haber encoder/decoder/decoder_with_past)
    for fp32_path in onnx_files:
        fname = os.path.basename(fp32_path)
        int8_path = os.path.join(args.out_onnx_int8, fname)

        print(f"  • {fname} -> INT8")
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,     # INT8 pesos
            per_channel=args.per_channel,    # opcional
            reduce_range=False               # normalmente False
        )

    # Copiar tokenizer/config para cargar carpeta INT8 como modelo HF
    copy_tokenizer_and_config(args.out_onnx_fp32, args.out_onnx_int8)

    print("\n✅ Cuantización ONNX INT8 dinámico completada.")
    print(f"📁 ONNX FP32: {args.out_onnx_fp32}")
    print(f"📁 ONNX INT8 dinámico: {args.out_onnx_int8}")
    print("\n🧪 Prueba rápida de carga:")
    print("   from optimum.onnxruntime import ORTModelForSeq2SeqLM")
    print(f"   m = ORTModelForSeq2SeqLM.from_pretrained('{args.out_onnx_int8}')")


if __name__ == "__main__":
    main()
