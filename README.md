# Class Translate AI

Herramienta de transcripción de audio y video a texto utilizando el modelo Whisper de OpenAI.

## Características

- **Interfaz web** con Streamlit para subir y transcribir archivos de audio/video
- **Soporte para archivos largos** mediante división automática en chunks de 10 minutos
- **Aceleración por GPU** (CUDA) cuando está disponible
- **Extracción de audio** desde archivos de video en múltiples formatos
- **Unificación de audios** MP3 desde una carpeta

## Requisitos

- Python 3.8+
- PyTorch
- FFmpeg
- Modelo Whisper (`openai-whisper`)
- Streamlit
- pydub

## Instalación

```bash
pip install torch whisper streamlit pydub
```

Asegúrate de tener FFmpeg instalado en tu sistema.

## Uso

### Interfaz web

```bash
streamlit run app.py
```

Accede a `http://localhost:8501` y sube archivos de audio/video para transcribir.

### Script de extracción de audios

```bash
python Obtener_Audios_WSP.py
```

Procesa archivos de video desde la carpeta de descargas y genera transcripciones.

### Unificación de audios

```bash
python Unificar_Audios.py
```

Combina múltiples archivos MP3 en uno solo.

## Estructura

| Archivo | Descripción |
|---------|-------------|
| `app.py` | Aplicación web con Streamlit |
| `backend_transcripcion.py` | Lógica de transcripción con Whisper |
| `Obtener_Audios_WSP.py` | Extracción y transcripción de videos |
| `Unificar_Audios.py` | Unificación de archivos MP3 |

## Configuración

- **Contraseña de acceso**: `admin_123` (cambiar en producción)
- **Idioma de transcripción**: Español (`language="es"`)
- **Modelo Whisper**: `turbo`
- **Chunk size**: 10 minutos por segmento
