# Class Translate AI 🎙️

Herramienta de transcripción y evaluación de audio y video a texto utilizando el modelo **Whisper** de OpenAI, optimizada para **Apple Silicon (MLX)** y **NVIDIA CUDA (Windows/Linux)**.

---

## 🚀 Características

- **Aceleración Nativa Multiplataforma**:
  - **macOS (Apple Silicon M1/M2/M3/M4)**: Soporte nativo con `mlx-whisper` (GPU + Neural Engine vía Metal) sin cuellos de botella de PyTorch MPS.
  - **Windows 11 / Linux (NVIDIA CUDA)**: Soporte para `faster-whisper` (CTranslate2) y PyTorch CUDA con sincronización de hardware.
- **Suite de Evaluación y Benchmarking (`evals_whisper.py`)**:
  - Medición rigurosa de calidad: **WER**, **Word Accuracy ($1 - \text{WER}$)**, **CER**, y **Tasa de Alucinación** en tramos de silencio.
  - Medición de eficiencia de hardware: **RTF (Real-Time Factor)**, **Throughput ($X\times$ tiempo real)** y **Consumo de Memoria Pico**.
  - Modo comparador (`--compare`) para generar reportes Markdown cruzando resultados entre Windows 11 y macOS.
- **Transcripción Robusta sin Cortes**: Parámetros anti-cascada (`condition_on_previous_text=False`, umbrales de voz adaptados) para evitar silenciamientos en clases largas.
- **Interfaz Web (Streamlit)**: Subida directa de archivos largos con soporte de división automática.
- **Procesamiento de Archivos Grandes**: Extracción y compresión eficiente a WAV 16kHz mono mediante FFmpeg.

---

## 📦 Instalación de Dependencias

### 1. Requisitos Generales
Asegúrate de tener instalado **FFmpeg** en tu sistema:
- **macOS**: `brew install ffmpeg`
- **Windows**: `winget install Gyan.FFmpeg` o descargar desde [ffmpeg.org](https://ffmpeg.org/) y agregar al PATH.

### 2. Entorno Python

```bash
# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate        # En macOS / Linux
# .venv\Scripts\activate       # En Windows
```

### 3. Instalar según tu Plataforma

#### 🍎 En macOS (Apple Silicon M1/M2/M3/M4):
```bash
pip install -r requirements.txt
pip install mlx mlx-whisper
```

#### 🪟 En Windows 11 (NVIDIA CUDA):
```bash
pip install -r requirements.txt
pip install faster-whisper pynvml torch --index-url https://download.pytorch.org/whl/cu121
```

> [!IMPORTANT]
> **Nota para Windows 11 (CTranslate2 / faster-whisper):**
> Asegúrate de que las librerías `cublas64_12.dll` y `cudnn64_9.dll` estén disponibles en el PATH del sistema o en la carpeta del entorno virtual. Esto evita que el motor haga fallback a CPU en silencio.

---

## 📊 Suite de Evaluación (`evals_whisper.py`)

El script evalúa el modelo `whisper-large-v3-turbo` con control estricto de variables (decodificación greedy determinista `temperature=0.0`, `beam_size=1`, normalización canónica con `jiwer` y sincronización de timers).

### 1. Crear dataset de prueba (si no tienes uno)
```bash
python evals_whisper.py --create-samples
```
Crea audios de prueba en `eval_dataset/` incluyendo muestras de voz y muestras de silencio/ruido para medir alucinaciones.

### 2. Ejecutar la evaluación en tu máquina actual
```bash
# 5 corridas por muestra para promedios estadísticos y desviación estándar
python evals_whisper.py --dataset_dir eval_dataset --runs 5 --output_json resultados_mi_equipo.json
```

### 3. Comparar resultados de Windows (CUDA) vs Mac (MLX)
Una vez ejecutada la evaluación en ambas computadoras, coloca ambos archivos `.json` en una misma carpeta y ejecuta:
```bash
python evals_whisper.py --compare eval_results_mlx_macos.json eval_results_cuda_windows.json
```
Esto imprimirá y guardará una tabla comparativa en Markdown con las diferencias en **Word Accuracy, WER, RTF, Speedup y Memoria**.

---

## 🎙️ Scripts de Transcripción

### 1. Transcripción por Lotes (Descargas -> Documentos)
```bash
python Obtener_Audios_WSP.py
```
Lee los videos ubicados en `~/Downloads/clases`, extrae el audio a WAV 16kHz y genera los `.txt` en `~/Documents/audios_clases`.

### 2. Interfaz Web (Streamlit)
```bash
streamlit run app.py
```
Accede a `http://localhost:8501` (Contraseña por defecto: `admin_123`).

### 3. Generación de Subtítulos (.SRT)
```bash
python generate_sub.py
```

### 4. Unificación de Audios MP3
```bash
python Unificar_Audios.py
```

---

## 📁 Estructura del Proyecto

| Archivo / Carpeta | Descripción |
|---|---|
| `evals_whisper.py` | Framework de evaluación y benchmarking objetivo (CUDA vs. MLX) |
| `eval_dataset/` | Conjunto de audios y Ground Truth (`manifest.json`) |
| `Obtener_Audios_WSP.py` | Extracción de audios de descargas y transcripción por lotes |
| `backend_transcripcion.py` | Lógica central de transcripción con soporte nativo MLX/CUDA |
| `app.py` | Aplicación Web interactiva en Streamlit |
| `generate_sub.py` | Generador de subtítulos en formato `.srt` |
| `Unificar_Audios.py` | Concatenación de archivos de audio MP3 |
| `requirements.txt` | Dependencias del proyecto y de evaluación |
