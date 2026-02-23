
import torch
import transformers
import sys
import importlib.metadata

def get_package_version(package_name):
    """Obtiene la versión de un paquete instalado"""
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "No disponible"

print("=" * 80)
print("VERIFICACIÓN DEL ENTORNO - CODE2_AI")
print("=" * 80)

# 1. Verificar PyTorch
print("\n[1] PyTorch")
print(f"   Versión: {torch.__version__}")
try:
    cpu_works = torch.tensor([1.0, 2.0, 3.0]).sum().item() == 6.0
    print(f"   ✅ CPU: Funcionando")
except Exception as e:
    print(f"   ❌ CPU: Error - {e}")
    sys.exit(1)

# 2. Verificar CUDA
print("\n[2] CUDA")
cuda_available = torch.cuda.is_available()
print(f"   CUDA disponible: {cuda_available}")

if cuda_available:
    print(f"   Versión CUDA: {torch.version.cuda}")
    print(f"   cuDNN disponible: {torch.backends.cudnn.enabled}")
    print(f"   Número de GPUs: {torch.cuda.device_count()}")
    
    try:
        # Test en GPU
        gpu_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        gpu_result = gpu_tensor.sum().item()
        
        if gpu_result == 6.0:
            print(f"   ✅ GPU: Funcionando")
            
            # Mostrar info de GPU
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"      GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            print(f"   ❌ GPU: Cálculo incorrecto")
            sys.exit(1)
    except Exception as e:
        print(f"   ❌ GPU: Error - {e}")
        sys.exit(1)
else:
    print(f"   ⚠️  CUDA no disponible (usará CPU)")

# 3. Verificar Transformers
print("\n[3] Transformers (Hugging Face)")
print(f"   Versión: {transformers.__version__}")
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    print(f"   ✅ T5 Tokenizer y Modelo: Importables")
except Exception as e:
    print(f"   ❌ T5: Error - {e}")
    sys.exit(1)

# 4. Verificar Optimum
print("\n[4] Optimum")
optimum_version = get_package_version("optimum")
print(f"   Versión: {optimum_version}")
try:
    import optimum
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    print(f"   ✅ ORTModelForSeq2SeqLM: Importable")
except Exception as e:
    print(f"   ❌ Optimum ONNX: Error - {e}")
    sys.exit(1)

# 5. Verificar ONNX Runtime
print("\n[5] ONNX Runtime")
onnxruntime_version = get_package_version("onnxruntime")
print(f"   Versión: {onnxruntime_version}")
try:
    import onnxruntime
    available_providers = onnxruntime.get_available_providers()
    print(f"   Providers disponibles: {', '.join(available_providers)}")
    print(f"   ✅ ONNX Runtime: Funcionando")
except Exception as e:
    print(f"   ❌ ONNX Runtime: Error - {e}")
    sys.exit(1)

# 6. Verificar Datasets
print("\n[6] Datasets (Hugging Face)")
try:
    from datasets import load_dataset
    datasets_version = get_package_version("datasets")
    print(f"   Versión: {datasets_version}")
    print(f"   ✅ load_dataset: Importable")
except Exception as e:
    print(f"   ❌ Datasets: Error - {e}")
    sys.exit(1)

# 7. Verificar métricas
print("\n[7] Métricas (ROUGE, BLEU)")
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import corpus_bleu
    rouge_version = get_package_version("rouge-score")
    nltk_version = get_package_version("nltk")
    print(f"   rouge-score: {rouge_version}")
    print(f"   nltk: {nltk_version}")
    print(f"   ✅ ROUGE y BLEU: Importables")
except Exception as e:
    print(f"   ❌ Métricas: Error - {e}")
    sys.exit(1)

# 8. Verificar pandas y numpy
print("\n[8] Utilidades (pandas, numpy)")
try:
    import pandas as pd
    import numpy as np
    pandas_version = pd.__version__
    numpy_version = np.__version__
    print(f"   ✅ pandas: {pandas_version}")
    print(f"   ✅ numpy: {numpy_version}")
except Exception as e:
    print(f"   ❌ Utilidades: Error - {e}")
    sys.exit(1)

# 9. Verificar psutil
print("\n[9] Monitoreo de sistema (psutil)")
try:
    import psutil
    psutil_version = get_package_version("psutil")
    print(f"   Versión: {psutil_version}")
    print(f"   ✅ psutil: Importable")
except Exception as e:
    print(f"   ❌ psutil: Error - {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ RESULTADO: Entorno estable y listo para usar")
print("=" * 80)

if cuda_available:
    print("\n💡 Nota: CUDA está disponible. El entrenamiento usará GPU para mayor velocidad.")
else:
    print("\n💡 Nota: CUDA no está disponible. El entrenamiento usará CPU (más lento).")
