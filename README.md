# CODE-2 Translator (T5) — NL → Ensamblado CODE-2

Este repositorio contiene un sistema completo para **traducir instrucciones en lenguaje natural** a **ensamblado CODE-2** mediante un modelo **T5 (Seq2Seq)** entrenado con un **dataset sintético** generado automáticamente.

Incluye:
- Generación de dataset (NL → CODE-2) en JSON
- Entrenamiento (Full fine-tuning)
- Evaluación con métricas (Exact Match, BLEU, ROUGE-L y tiempo de inferencia)
- Cuantización (PTQ dinámico INT8 en PyTorch y ONNX INT8 dinámico)
- Despliegue local (CLI y Gradio)
- Despliegue en Hugging Face Spaces (Gradio)

---

## ✨ Características principales

- Traducción de múltiples líneas: la entrada puede contener varias instrucciones separadas por saltos de línea.
- Manejo de errores: el generador incorpora ejemplos con registros/direcciones inválidas para mejorar robustez.
- Normalización de salida: elimina tokens especiales, espacios sobrantes y preserva los saltos de línea reales.
- Optimización para inferencia: opción ONNX INT8 dinámico con mejoras reales de tiempo en CPU.

---

## 📁 Estructura del repositorio

```text
CODE2-T5/
├── data/
│   ├── train.json
│   ├── valid.json
│   └── README.md
├── models/
│   ├── full_fp32/
│   └── onnx_int8_dynamic/
├── scripts/
│   ├── data_generation.py
│   ├── training.py
│   ├── evaluate.py
│   ├── export_onnx_quantize.py
│   └── compare_models.py
├── deployment/
│   ├── cli.py
│   └── app_gradio.py
├── requirements.txt
└── README.md

---

## ⚙️ Instalación

Recomendado: entorno virtual.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

> Nota: si se usa `transformers`, es importante mantener `huggingface-hub < 1.0` para compatibilidad.

---

## 🚀 Uso rápido (CLI)

Ejecuta el traductor desde consola y elige el modelo (FP32 u ONNX INT8):

```bash
python deployment/cli.py
```

Ejemplo de entrada:

```text
Suma r1 y r2 y guarda en r3
Guarda r3 en la dirección 0345
```

Salida esperada:

```text
ADDS r3,r1,r2
ST [rD+H'45'],r3 ; rD = 0300
```

---

## 🌐 Demo web (Gradio)

### Local

```bash
python deployment/app_gradio.py
```

La interfaz permite:

* Cambiar entre modelo **FP32** y **ONNX INT8**
* Ajustar `num_beams` (beam search)
* Medir tiempo de inferencia en cada ejecución

### Hugging Face Spaces

Este mismo archivo es compatible con Spaces (modo CPU).
Solo es necesario subirlo como `app.py` junto con `requirements.txt` y los directorios de modelo.

---

## 🧪 Evaluación (métricas)

Evalúa un modelo usando `valid.json`:

```bash
python scripts/evaluate.py --model_path models/full_fp32 --valid_path data/valid.json
```

Métricas calculadas:

* Exact Match (con normalización de saltos y tokens especiales)
* BLEU
* ROUGE-L
* Tiempo medio de inferencia
* Tokens medios generados

---

## 🧠 Entrenamiento (Full fine-tuning)

Entrena T5 con el dataset JSON generado:

```bash
python scripts/training.py --train data/train.json --valid data/valid.json --save_dir models/full_fp32
```

El entrenamiento:

* Tokeniza entrada/salida con padding y truncation
* Ajusta `max_length` en función del dataset
* Guarda modelo y tokenizer al final

---

## 🧰 Generación de datos (dataset sintético)

Genera un dataset NL → CODE-2 en formato JSON:

```bash
python scripts/data_generation.py
```

El generador:

* Produce instrucciones válidas y casos con error (≈10%)
* Evita duplicados
* Construye bloques multilínea (hasta 11 instrucciones)
* Mantiene saltos de línea como `\n` reales

---

## 🧮 Cuantización (PTQ INT8)

### 1) PTQ dinámico INT8 (PyTorch)

Se aplica `dynamic quantization` sobre capas `nn.Linear` en CPU.

```bash
python scripts/ptq_dynamic_int8.py --model_path models/full_fp32 --out_dir models/ptq_int8_dynamic
```

### 2) Exportación a ONNX + cuantización INT8 dinámica

```bash
python scripts/export_onnx_quantize.py --model_path models/full_fp32 --out_dir models/onnx_int8_dynamic
```

---

## ➕ Cómo añadir nuevas instrucciones (extender CODE-2)

1. Edita `scripts/data_generation.py`:

   * Añade una plantilla NL nueva
   * Añade el formato CODE-2 correspondiente
2. Genera dataset:

   ```bash
   python scripts/data_generation.py
   ```
3. Entrena o continúa entrenamiento:

   ```bash
   python scripts/training.py --train data/train.json --valid data/valid.json --save_dir models/full_fp32
   ```

---

## 📌 Reproducibilidad

* Se fija `seed` en entrenamiento/evaluación.
* Se mantiene un pipeline completo: generación → entrenamiento → evaluación → despliegue.
* Los modelos se guardan junto con el tokenizer utilizado.

---

## 📄 Licencia

Uso académico / TFG.
Si se publica abiertamente, se recomienda añadir una licencia (MIT o Apache-2.0).

```

---

### Siguiente paso
Para seguir “paso a paso” como quieres:

1) dime el **nombre exacto del repo** (ej: `code2-t5`)  
2) dime si vas a subir **pesos** al repo o solo enlaces a Hugging Face

y te preparo el `requirements.txt` compatible (evitando el error de `huggingface-hub==1.2.4`) y los textos cortos del `data/README.md`.
```
