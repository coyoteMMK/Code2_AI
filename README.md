# CODE2_AI — Traductor NL → CODE-2 (T5 fine-tuning + ONNX INT8)

Un **traductor automático** de lenguaje natural a código ensamblador **CODE-2** basado en el modelo T5 (Text-to-Text Transfer Transformer) de Google.

## 🎯 Descripción general

Este repositorio contiene un **pipeline completo** para:
1. **Entrenar** un modelo T5-small con Fine-Tuning completo (PyTorch FP32)
2. **Optimizar** el modelo mediante cuantización INT8 y exportación a ONNX
3. **Evaluar** con métricas estándar (Exact Match, BLEU, ROUGE-L)
4. **Usar** el modelo a través de una CLI interactiva intuitiva

### 🚀 Características principales

- ✅ **Full Fine-Tuning**: Entrenamiento completo del modelo con datos específicos
- ✅ **Cuantización INT8**: Reduce tamaño (76% menor) sin pérdida significativa de precisión
- ✅ **ONNX Runtime**: Inferencia ultra-optimizada y multiplataforma
- ✅ **CLI Interactiva**: Interfaz fácil de usar con selección de modelos
- ✅ **Evaluación completa**: Métricas BLEU, ROUGE-L, Exact Match
- ✅ **GPU & CPU**: Soporte automático para CUDA 12.1 o CPU
- ✅ **Multi-plataforma**: Windows, Linux, macOS

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

## 💻 Requisitos del sistema

### Requisitos mínimos

| Componente | Requisito | Notas |
|-----------|-----------|-------|
| **Python** | ≥ 3.8 | Probado con 3.10, 3.11, 3.12 |
| **RAM** | ≥ 8 GB | Recomendado ≥ 16 GB para entrenamiento |
| **Navegador** | Cualquiera | Para interfaz web (opcional) |

### Requisitos recomendados para entrenamiento

| Componente | Mínimo | Recomendado |
|-----------|--------|------------|
| **GPU NVIDIA** | No requerida | RTX 2060 Super o superior |
| **VRAM** | - | ≥ 4 GB |
| **Espacio disco** | 10 GB | 20 GB |
| **OS** | Windows/Linux/Mac | Windows 10+ o Ubuntu 20.04+ |

### GPU Compatibles (CUDA 12.1)

- ✅ NVIDIA GeForce RTX 2060 / 2070 / 2080 / 2090
- ✅ NVIDIA GeForce RTX 3000 series
- ✅ NVIDIA GeForce RTX 4000 series
- ✅ NVIDIA A100, RTX 6000, etc.
- ✅ Cualquier GPU NVIDIA con Compute Capability ≥ 6.1

**Sin GPU?** No hay problema, el modelo puede entrenarse en CPU (será más lento, ~5-10x).

---

## ✅ Instalación

### Paso 1: Clonar el repositorio

```bash
# Clonar desde GitHub
git clone https://github.com/coyoteMMK/Code2_AI.git
cd Code2_AI
```

**Alternativas:**

Si prefieres clonar una rama específica:
```bash
git clone -b main https://github.com/coyoteMMK/Code2_AI.git
```

Si ya tienes el repositorio y necesitas actualizar:
```bash
cd Code2_AI
git pull origin main
```

### Paso 2: Crear y activar entorno virtual

Se recomienda usar un entorno virtual para evitar conflictos de dependencias:

```bash
python -m venv .venv
```

**Activar en Windows:**

```powershell
# PowerShell:
.venv\Scripts\Activate.ps1

# CMD:
.venv\Scripts\activate.bat
```

**Activar en Linux/Mac:**

```bash
source .venv/bin/activate
```

Verifica que el entorno está activo (debe aparecer `(.venv)` en tu terminal):
```bash
(.venv) $ 
```

### Paso 3: Instalar dependencias

**Opción A: Instalación estándar (RECOMENDADO para NVIDIA GPU)**

```bash
# 1. Instalar PyTorch con soporte CUDA 12.1
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121

# 2. Instalar el resto de dependencias
pip install -r requirements.txt

# 3. Verificar que todo está correcto
cd scripts
python test_env.py
```

**Opción B: Instalación solo CPU**

Si no tienes GPU o quieres usar solo CPU:

```bash
# Instalar PyTorch sin CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto de dependencias
pip install -r requirements.txt

# Verificar instalación
cd scripts
python test_env.py
```

**Opción C: Instalación en macOS (M1/M2 chips)**

```bash
# Para chips Apple Silicon:
pip install torch torchvision torchaudio

# Instalar dependencias
pip install -r requirements.txt
```

### Paso 4: Verificar la instalación

El script `test_env.py` verifica que todo está configurado correctamente:

```bash
cd scripts
python test_env.py
```

**Salida esperada:**
```
================================================================================
VERIFICACIÓN DEL ENTORNO - CODE2_AI
================================================================================

[1] PyTorch
   Versión: 2.5.1+cu121
   ✅ CPU: Funcionando

[2] CUDA
   CUDA disponible: True
   Versión CUDA: 12.1
   cuDNN disponible: True
   Número de GPUs: 1
   ✅ GPU: Funcionando
      GPU 0: NVIDIA GeForce RTX 2060 (6.00 GB)

[3] Transformers (Hugging Face)
   Versión: 4.57.6
   ✅ T5 Tokenizer y Modelo: Importables

...

✅ RESULTADO: Entorno estable y listo para usar
```

---

## 🎬 Inicio rápido (Quick Start)

Si ya tienes todo instalado y solo quieres probar el modelo:

```bash
# 1. Activar entorno
.venv\Scripts\Activate.ps1  # Windows PowerShell
# o
source .venv/bin/activate   # Linux/Mac

# 2. Ir a scripts
cd scripts

# 3. Lanzar CLI interactiva
python code2_cli_multi.py
```

Luego:
1. Elige el modelo: **Full FP32** (más preciso, usa GPU) o **ONNX INT8** (más rápido, usa CPU)
2. Escribe una instrucción en lenguaje natural (ej: "Suma r1 y r2 y guarda en r3")
3. Presiona ENTER dos veces
4. ¡Recibe código ensamblador CODE-2 generado!

---

---

## 📦 Dependencias Completas

### requirements-torch.txt
Dependencias de PyTorch optimizadas para CUDA 12.1:

```
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
```

**Instalación especial**:
```bash
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
```

### requirements.txt
Dependencias principales del proyecto:

| Librería | Propósito |
|----------|-----------|
| `transformers>=4.30` | Modelos y tokenizers de Hugging Face (T5) |
| `datasets>=2.10` | Carga y procesamiento de datasets |
| `optimum[onnxruntime]>=1.12` | Exportación y optimización a ONNX |
| `onnxruntime>=1.15` | Ejecución de modelos ONNX (CPUExecutionProvider) |
| `onnxruntime-gpu>=1.15` | *Opcional*: Soporte GPU para ONNX |
| `rouge-score>=0.1.2` | Cálculo de métrica ROUGE-L |
| `nltk>=3.8` | Cálculo de métrica BLEU |
| `pandas>=1.3` | Manipulación de datos y resultados |
| `numpy>=1.21` | Computación numérica |
| `psutil>=5.9` | Monitoreo de recursos del sistema |
| `huggingface-hub<1.0` | Descarga de modelos de Hugging Face |

**Verificación de instalación**:
```bash
# Este script verifica que todas las dependencias estén correctamente instaladas
cd scripts
python test_env.py
```

El script `test_env.py` verifica:
- ✅ Disponibilidad de PyTorch y CUDA
- ✅ Disponibilidad de transformers y datasets
- ✅ Disponibilidad de ONNX Runtime (CPU y opcionalmente GPU)
- ✅ Bibliotecas de métricas (ROUGE, BLEU)
- ✅ Otras dependencias auxiliares
- ✅ Acceso a Hugging Face Hub

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

## 🔄 Workflow típico

Sigue estos pasos para entrenar y usar el modelo:

### 1️⃣ Verificar dependencias
```bash
cd scripts
python test_env.py
```

### 2️⃣ (Opcional) Generar datos personalizados
```bash
cd scripts
python data_generation.py --n_total 50000 --out_dir ../datasource
```

### 3️⃣ Entrenar el modelo
```bash
cd scripts
python training.py
# Guarda en: ../models/full_fp32/
```

### 4️⃣ (Opcional) Cuantizar a ONNX INT8
```bash
cd scripts
python quantize_onnx_int8_dynamic.py
# Genera: ../models/onnx_int8_dynamic/
```

### 5️⃣ Evaluar el modelo
```bash
cd scripts
python evaluate.py
# Elige modelo y ajusta parámetros interactivamente
```

### 6️⃣ Usar en CLI
```bash
cd scripts
python code2_cli_multi.py
# Traduce texto natural a CODE-2
```

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
- Guarda el modelo en `models/full_fp32/`
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
- Carga el modelo desde `models/full_fp32/`
- Exporta a ONNX FP32 en `models/onnx_fp32/`
- Cuantiza a INT8 dinámico en `models/onnx_int8_dynamic/`

> **Nota**: Requiere que `optimum[onnxruntime]` esté instalado. Ver sección [Dependencias](#-dependencias-completas).

---

## 🔍 Evaluación Interactiva

Evaluador interactivo que permite probar diferentes configuraciones (num_beams, temperature, top_p).

```bash
cd scripts
python evaluate.py
```

Este script:
- Permite seleccionar entre modelo PyTorch FP32 o ONNX INT8
- Evalúa con datos de `datasource/test.json`
- Permite ajustar parámetros de generación interactivamente
- Ver ejemplos de predicciones
- Guardar historial de evaluaciones

Métricas calculadas:

* **Exact Match**: Coincidencia exacta (con normalización de espacios y saltos de línea)
* **BLEU**: Bilingual Evaluation Understudy
* **ROUGE-L**: Recall-Oriented Understudy for Gisting Evaluation

> **Nota**: Por defecto usa 100 ejemplos del dataset de test para evaluación rápida. Modifica `SAMPLE_SIZE` en el archivo para cambiar esto

---

## 💻 CLI Interactiva

Interfaz de línea de comandos con menú interactivo para traducir lenguaje natural a código ensamblador CODE-2.

```bash
cd scripts
python code2_cli_multi.py
```

Características:
- **Menú de selección de modelo**: Elige entre PyTorch FP32 u ONNX INT8
- **Entrada multilínea**: Introduce instrucciones en lenguaje natural con múltiples líneas
- **Visualización de código**: Muestra el código generado con tiempo de inferencia
- **Soporte TOKEN SALTO**: Preserva saltos de línea en el ensamblador
- **Interfaz interactiva**: Navegación con flechas y ENTER

> **Nota**: Asegúrate de haber entrenado el modelo primero con `training.py` y opcionalmente cuantizado con `quantize_onnx_int8_dynamic.py`

---

## 🧠 Formato esperado

### Entrada (NL)

* **Texto multilínea**: Una instrucción o descripción por línea
* **Flexibility**: Se permite variación de mayúsculas/minúsculas
* **Tokens especiales**: El modelo maneja automáticamente saltos de línea como `SALTO`

Ejemplo:

```text
Suma r1 y r2 y guarda en r3
Guarda r3 en la dirección 0345
```

### Salida (CODE-2)

El modelo genera código ensamblador CODE-2 optimizado:

```text
ADDS r3,r1,r2
ST [rD+H'45'],r3 ; rD = 0300
```

> **Nota**: Los saltos de línea en la entrada se normalizan con el token `SALTO` internamente

---

## 🚀 Características principales

### Modelos soportados

| Modelo | Device | Ventajas | Tamaño |
|--------|--------|----------|--------|
| **Full FP32** (PyTorch) | GPU (CUDA) | Máxima precisión | ~242 MB |
| **ONNX INT8** | CPU | Inferencia rápida, menor consumo de recursos | ~63 MB |

### Métricas de calidad

- **BLEU (Bilingual Evaluation Understudy)**: Similitud léxica coin-grams
- **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)**: Evaluación semántica
- **Exact Match**: Coincidencia perfecta (normalizada)

### Optimizaciones aplicadas

- ✅ **Quantization INT8**: Reduce tamaño y aumenta velocidad sin pérdida significativa de precisión
- ✅ **ONNX Runtime**: Motor de inferencia ultra-optimizado
- ✅ **Token SALTO**: Manejo eficiente de saltos de línea
- ✅ **Batch Processing**: Procesa múltiples muestras simultáneamente
- ✅ **Dynamic Padding**: Ajusta el padding según la longitud de entrada

---

## ⚡ Benchmark (referencia)

Tiempo de inferencia en ejemplos típicos (100 muestras, batch_size=32):

| Modelo | Device | Tiempo promedio | Velocidad |
|--------|--------|-----------------|-----------|
| Full FP32 | GPU (CUDA) | ~0.42s | ~238 muestras/s |
| ONNX INT8 | CPU | ~0.85s | ~118 muestras/s |

> Valores aproximados. Varían según hardware y configuración.

---

## 📌 Notas de compatibilidad

* `transformers` requiere `huggingface-hub<1.0`.
  Si al instalar aparece `huggingface-hub==1.x`, desinstala y vuelve a instalar:

```bash
pip uninstall -y huggingface-hub
pip install "huggingface-hub<1.0"
```

---

## 🔧 Troubleshooting

### ❌ Error: "No module named 'optimum'"
```bash
pip install optimum[onnxruntime]
```

### ❌ Error: "CUDA out of memory"

**Solución:**
- Reduce `BATCH_SIZE` en `training.py` (línea ~20)
- Usa el modelo ONNX INT8 que corre en CPU
- Verifica tu versión de CUDA: `python -c "import torch; print(torch.cuda.get_device_properties(0))"`

### ❌ Error: "Cannot import name 'ORTModelForSeq2SeqLM'"
```bash
pip install --upgrade optimum onnxruntime
```

### ❌ Error: "AttributeError: module 'optimum' has no attribute '__version__'"

El script `test_env.py` ya está corregido. Si lo ves, ejecuta:
```bash
git pull origin main
```

### ❌ Error: "RuntimeError: expected scalar type Half but found Float"

En algunos scripts, asegúrate que el dtype es correcto:
```bash
# En training.py, línea 114, verifica:
fp16=torch.cuda.is_available()  # Debe ser automático
```

### ⚠️ CUDA detectada pero no funciona

Verifica tu instalación:
```bash
cd scripts
python test_env.py
```

Si dice `CUDA disponible: False`, reinstala PyTorch:
```bash
pip uninstall torch
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
```

### 💡 Test rápido del entorno
```bash
cd scripts
python test_env.py
```

Debe mostrar estado ✅ en todas las secciones.

---

## 📚 Bibliografía

- **T5 Model**: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- **ONNX Runtime**: [ONNX Runtime Optimization Guide](https://onnxruntime.ai/)
- **Quantization**: [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic Only Inference](https://arxiv.org/abs/1806.08342)
- **CODE-2 Assembly**: [CODE-2 ISA Documentation](https://www.example.com)

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📧 Contacto & Soporte

- **Issues**: [Reportar problema](https://github.com/coyoteMMK/Code2_AI/issues)
- **Discussions**: [Preguntas y discusiones](https://github.com/coyoteMMK/Code2_AI/discussions)
- **Email**: contacto@example.com

---

## 📜 Licencia

Este proyecto está licenciado bajo la **MIT License** - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- Equipo de [Hugging Face](https://huggingface.co/) por `transformers` y `datasets`
- Microsoft por [ONNX Runtime](https://onnxruntime.ai/)
- Comunidad de Machine Learning en código abierto
