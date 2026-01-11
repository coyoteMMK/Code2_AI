# CODE2_AI — Traductor NL → CODE-2 (T5 fine-tuning + ONNX INT8)

Este repositorio contiene el pipeline completo para entrenar y evaluar un modelo tipo T5 que traduce **lenguaje natural (NL)** a **ensamblador CODE-2**.

Incluye:
- Entrenamiento **Full Fine-Tuning** (PyTorch FP32)
- Exportación y cuantización **ONNX INT8 dinámico**
- Evaluación con métricas (Exact Match, BLEU, ROUGE-L, tiempo de inferencia)
- Demo local (Gradio) para probar el modelo

---

## 📁 Estructura del repositorio


CODE2_AI/
├─ datasource/
│  ├─ train.json
│  ├─ valid.json
│  └─ test.json
│
├─ models/
│  ├─ full_fp32/           # modelo PyTorch (fine-tuning completo)
│  └─ onnx_int8_dynamic/   # modelo ONNX cuantizado INT8 dinámico
│
├─ scripts/
│  ├─ training.py
│  ├─ evaluate.py
│  ├─ quantize_onnx_int8_dynamic.py
│  ├─ showcase.py
│  ├─ data-generation.py
│  └─ test_env.py
│
├─ README.md
└─ requirements.txt

---

## ✅ Instalación

Se recomienda entorno virtual:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
````

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Entrenamiento (Full Fine-Tuning)

Entrena un modelo y lo guarda en `models/full_fp32/`.

```bash
python scripts/training.py \
  --model_name t5-small \
  --train_json datasource/train.json \
  --valid_json datasource/valid.json \
  --save_dir models/full_fp32
```

> Para continuar entrenando desde un modelo ya afinado (no desde `t5-small`), usa:

```bash
python scripts/training.py \
  --model_name models/full_fp32 \
  --train_json datasource/train.json \
  --valid_json datasource/valid.json \
  --save_dir models/full_fp32
```

---

## ⚙️ Cuantización ONNX INT8 (PTQ dinámico)

Exporta a ONNX y aplica cuantización INT8 dinámica. Guarda el resultado en `models/onnx_int8_dynamic/`.

```bash
python scripts/quantize_onnx_int8_dynamic.py \
  --model_dir models/full_fp32 \
  --out_dir models/onnx_int8_dynamic
```

---

## 📊 Evaluación

Evalúa **dos modelos**:

* `models/full_fp32/` (PyTorch FP32)
* `models/onnx_int8_dynamic/` (ONNX INT8 dinámico)

```bash
python scripts/evaluate.py \
  --fp32_dir models/full_fp32 \
  --onnx_dir models/onnx_int8_dynamic \
  --valid_json datasource/valid.json \
  --n 2000
```

Métricas calculadas:

* Exact Match (con normalización de espacios y saltos de línea)
* BLEU
* ROUGE-L
* Tiempo medio de inferencia
* Longitud media en tokens

---

## 🧪 Demo local (Gradio)

Lanza una app local para introducir instrucciones multilínea y elegir modelo (FP32 u ONNX INT8).

```bash
python scripts/showcase.py
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

---

Si quieres, en el siguiente paso te dejo **el contenido recomendado de cada script** (`evaluate.py`, `quantize_onnx_int8_dynamic.py`, `showcase.py`) con rutas ya alineadas a tu repo (`datasource/` y `models/`).
```
