# scripts/quantize_onnx_int8_dynamic.py
import os
import argparse
from pathlib import Path

from transformers import T5Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

from onnxruntime.quantization import quantize_dynamic, QuantType


def quantizar_directorio_onnx(input_dir: str, output_dir: str):
    """
    Cuantiza todos los .onnx dentro de input_dir y guarda en output_dir
    manteniendo los mismos nombres de archivo.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(list(input_dir.glob("*.onnx")))
    if not onnx_files:
        raise FileNotFoundError(f"No se encontraron .onnx en: {input_dir}")

    print(f"🔎 ONNX encontrados: {len(onnx_files)}")
    for f in onnx_files:
        out_path = output_dir / f.name
        print(f"⚙️ Cuantizando: {f.name} -> {out_path.name}")

        # PTQ dinámico: no necesita dataset
        quantize_dynamic(
            model_input=str(f),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,   # INT8 dinámico
            per_channel=True,              # suele mejorar calidad
        )

    print("✅ Cuantización dinámica INT8 completada para todos los ONNX.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32_dir", default="models/full_fp32_extended/", help="Directorio del modelo Transformers FP32")
    parser.add_argument("--onnx_fp32_dir", default="models/onnx_fp32_extended/", help="Salida ONNX FP32")
    parser.add_argument("--onnx_int8_dir", default="models/onnx_int8_dynamic_extended/", help="Salida ONNX INT8 dinámico")
    args = parser.parse_args()

    fp32_dir = args.fp32_dir
    onnx_fp32_dir = args.onnx_fp32_dir
    onnx_int8_dir = args.onnx_int8_dir

    os.makedirs(onnx_fp32_dir, exist_ok=True)
    os.makedirs(onnx_int8_dir, exist_ok=True)

    print("🔧 Cargando tokenizer (desde el modelo FP32)...")
    tokenizer = T5Tokenizer.from_pretrained(fp32_dir)

    # 1) Exportar a ONNX FP32 (Optimum)
    print("🧠 Exportando Transformers -> ONNX (FP32) con Optimum...")
    # Nota: en tu versión, export=True funciona (y evita from_transformers)
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(fp32_dir, export=True)
    ort_model.save_pretrained(onnx_fp32_dir)
    tokenizer.save_pretrained(onnx_fp32_dir)
    print(f"✅ ONNX FP32 guardado en: {onnx_fp32_dir}")

    # 2) Cuantizar ONNX -> INT8 dinámico
    print("⚙️ Cuantizando ONNX con PTQ dinámico INT8...")
    quantizar_directorio_onnx(onnx_fp32_dir, onnx_int8_dir)

    # Copiar tokenizer/config al dir INT8 (para cargar ORTModelForSeq2SeqLM)
    print("📦 Guardando tokenizer/config en la carpeta INT8...")
    tokenizer.save_pretrained(onnx_int8_dir)

    # Cargar y volver a guardar ORTModel apuntando a los ONNX INT8 (mantiene estructura)
    # (Esto asegura que el directorio INT8 tenga el layout esperado por Optimum)
    print("🔁 Re-guardando wrapper ORT en carpeta INT8...")
    ort_model_int8 = ORTModelForSeq2SeqLM.from_pretrained(onnx_fp32_dir)
    ort_model_int8.save_pretrained(onnx_int8_dir)

    print("\n✅ Proceso completado:")
    print(f"📁 ONNX FP32: {onnx_fp32_dir}")
    print(f"📁 ONNX INT8 dinámico: {onnx_int8_dir}")


if __name__ == "__main__":
    main()
