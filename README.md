# CODE2_AI — Traductor NL → CODE-2 (T5 fine-tuning + ONNX INT8)

Este repositorio contiene el pipeline completo para entrenar y evaluar un modelo tipo T5 que traduce **lenguaje natural (NL)** a **ensamblador CODE-2**.

Incluye:
- Entrenamiento **Full Fine-Tuning** (PyTorch FP32)
- Exportación y cuantización **ONNX INT8 dinámico**
- Evaluación con métricas (Exact Match, BLEU, ROUGE-L, tiempo de inferencia)
- Demo local (Gradio) para probar el modelo

---

## 📁 Estructura del repositorio

```
Code2_AI/
├─ datasource/
│  ├─ train.json          # Dataset de entrenamiento
│  ├─ valid.json          # Dataset de validación
│  └─ test.json           # Dataset de prueba
│
├─ models/
│  ├─ full_fp32/          # Modelo PyTorch pre-entrenado (opcional)
│  └─ onnx_int8_dynamic/  # Modelo ONNX cuantizado (opcional)
│
├─ scripts/
│  ├─ training.py                    # Entrenamiento Full Fine-Tuning
│  ├─ evaluate.py                    # Evaluación interactiva
│  ├─ quantize_onnx_int8_dynamic.py  # Cuantización ONNX
│  ├─ code2_cli_multi.py             # Demo CLI interactiva
│  ├─ data_generation.py             # Generador de datasets
│  └─ test_env.py                    # Test de dependencias
│
├─ site/                  # Aplicación Next.js (frontend web - opcional)
│
├─ README.md
├─ requirements.txt
└─ requirements-torch.txt  # Dependencias PyTorch con CUDA
```

---

## ✅ Instalación

```bash
git clone https://github.com/coyoteMMK/Code2_AI.git
cd Code2_AI
```

Crear y activar entorno virtual (recomendado):

```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Windows CMD:
.venv\Scripts\activate.bat

# Linux/Mac:
source .venv/bin/activate
```

Instalación de dependencias:

```bash
# 1. Instalar PyTorch con soporte CUDA 12.1
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121

# 2. Instalar el resto de dependencias
pip install -r requirements.txt

# 3. Verificar instalación
cd scripts
python test_env.py
```

---

## 📝 Generación de Datos

Si necesitas generar un nuevo dataset o aumentar el existente:

```bash
cd scripts
python data_generation.py --n_total 36000 --out_dir ../datasource
```

Parámetros disponibles:
- `--n_total`: Número total de ejemplos (default: 36000)
- `--out_dir`: Directorio de salida (default: ../datasource)
- `--max_instr`: Máximo de instrucciones por ejemplo (default: 11)
- `--train_ratio`: Proporción para entrenamiento (default: 0.70)
- `--valid_ratio`: Proporción para validación (default: 0.20)
- `--test_ratio`: Proporción para prueba (default: 0.10)
- `--preserve_newlines`: Preservar saltos de línea como token SALTO

---

## 🏋️ Entrenamiento (Full Fine-Tuning)

Entrena un modelo T5-small con Full Fine-Tuning. Los parámetros están configurados directamente en el archivo.

```bash
cd scripts
python training.py
```

Este script:
- Carga `t5-small` desde Hugging Face
- Entrena con los datos de `datasource/train.json` y `datasource/valid.json`
- Guarda el modelo en `scripts/results_Full_optimizado/`
- Genera métricas en CSV, TXT y JSON

> Para modificar parámetros (batch size, learning rate, epochs, etc.), edita las variables de configuración al inicio de [training.py](scripts/training.py)

---

## ⚙️ Cuantización ONNX INT8 (PTQ dinámico)

Exporta el modelo entrenado a ONNX y aplica cuantización INT8 dinámica.

```bash
cd scripts
python quantize_onnx_int8_dynamic.py
```

Este script:
- Carga el modelo desde `scripts/results_Full_optimizado/`
- Exporta a ONNX FP32 en `scripts/t5_onnx_fp32/`
- Cuantiza a INT8 dinámico en `scripts/t5_onnx_int8_dynamic/`

> **uador interactivo que permite probar diferentes configuraciones (num_beams, temperature, top_p).

```bash
cd scripts
python evaluate.py
```

Este script:
- Carga el modelo desde `scripts/results_Full_optimizado/`
- Evalúa con datos de `datasource/test.json`
- Permite ajustar parámetros de generación interactivamente
- Ver ejemplos de predicciones
- Guardar historial de evaluaciones

Métricas calculadas:

* Exact Match (con normalización de espacios y saltos de línea)
* BLEU
* ROUGE-L

> **Nota**: Por defecto usa 100 ejemplos del dataset de test para evaluación rápida. Modifica `SAMPLE_SIZE` en el archivo para cambiar esto

* Exact Match (con normalización de espacios y saltos de línea)
* BLEU
* ROUGE-L
* Tiempo medio de inferencia
* Longitud media en tokens

---
CLI Interactiva

Interfaz de línea de comandos con menú interactivo para traducir lenguaje natural a código ensamblador CODE-2.

```bash
cd scripts
python code2_cli_multi.py
```

Características:
- Menú de selección de modelo (PyTorch FP32 u ONNX INT8)
- Entrada multilínea de instrucciones
- Visualización de código generado con tiempo de inferencia
- Soporte para TOKEN SALTO (saltos de línea en el ensamblador)

> **Nota**: Asegúrate de haber entrenado el modelo primero con `training.py` y opcionalmente cuantizado con `quantize_onnx_int8_dynamic.python scripts/showcase.py
```

---

## 🧠 Formato esperado

### Entrada (NL)

* Texto multilínea (una instrucción por línea)
* Se permite variación de mayúsculas/minúsculas

Ejemplo:

```text
Suma r1 y r2 y guarda en r3
Guarda r3 en la dirección 0345
```

### Salida (CODE-2)

Ejemplo:

```text
ADDS r3,r1,r2
ST [rD+H'45'],r3 ; rD = 0300
```

---

## 📌 Notas de compatibilidad

* `transformers` requiere `huggingface-hub<1.0`.
  Si al instalar aparece `huggingface-hub==1.x`, desinstala y vuelve a instalar:

```bash
pip uninstall -y huggingface-hub
pip install "huggingface-hub<1.0"
```

---

## 📜 Licencia

Placeholder (pendiente de definir).

```
