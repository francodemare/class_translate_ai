import os
import platform
import subprocess

# ==============================================================================
# Detección del Motor de Transcripción y Aceleración de Hardware
# ==============================================================================
USE_MLX = False
try:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import mlx_whisper
        USE_MLX = True
        print("🚀 Usando Apple Silicon MLX (mlx-whisper: GPU + Neural Engine optimizado)")
except ImportError:
    pass

if not USE_MLX:
    import torch
    import whisper
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 Usando aceleración NVIDIA CUDA")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Usando aceleración PyTorch MPS (Apple Silicon)")
    else:
        device = "cpu"
        print("⚠️ Usando CPU")
        
    print(f"Cargando modelo Whisper 'turbo' en {device}...")
    model = whisper.load_model("turbo", device=device)

# ==============================================================================
# Rutas del Sistema (macOS, Linux y Windows)
# ==============================================================================
HOME = os.path.expanduser("~")
ruta_carpeta = os.path.join(HOME, "Documents", "audios_clases")
ruta_descagar = os.path.join(HOME, "Downloads", "clases")

os.makedirs(ruta_carpeta, exist_ok=True)
os.makedirs(ruta_descagar, exist_ok=True)


def extraer_audio(video_path, audio_output_path):
    """
    Extrae el audio de un archivo de video o audio y lo convierte a WAV 16kHz mono.
    Usa ffmpeg directamente con stream copy/conversión ligera de bajo consumo de RAM.
    """
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            audio_output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return audio_output_path
    except Exception as e:
        print(f"❌ Error extrayendo audio de {video_path}: {e}")
        return None


def procesar_videos_en_descargas():
    """Procesa todos los videos en la carpeta de descargas y extrae el audio a la carpeta destino."""
    extensiones_validas = ('.mp4', '.wmv', '.avi', '.mov', '.mkv', '.m4a', '.mp3', '.aac', '.flac', '.ogg')
    archivos = [f for f in os.listdir(ruta_descagar) if f.lower().endswith(extensiones_validas)]
    
    if not archivos:
        print(f"No se encontraron archivos multimedia en: {ruta_descagar}")
        return

    print(f"Encontrados {len(archivos)} archivo(s) para procesar en {ruta_descagar}")
    for nombre_archivo in archivos:
        video_path = os.path.join(ruta_descagar, nombre_archivo)
        base_name = os.path.splitext(nombre_archivo)[0]
        audio_output_path = os.path.join(ruta_carpeta, f"{base_name}.wav")
        
        if os.path.exists(audio_output_path):
            print(f"Audio ya existe, saltando extracción: {audio_output_path}")
            continue

        print(f"Extrayendo audio: {nombre_archivo} -> {audio_output_path}")
        resultado = extraer_audio(video_path, audio_output_path)
        if resultado:
            print(f"✅ Audio listo: {resultado}")


def transcribir_archivo(audio_file):
    """
    Transcribe un archivo de audio utilizando mlx-whisper (en Mac Apple Silicon)
    o whisper estándar (en CUDA/CPU) con parámetros anti-cortes de audio.
    """
    # Parámetros optimizados para evitar saltos de texto, silenciamiento y alucinaciones
    if USE_MLX:
        # mlx-whisper usa modelos nativos optimizados para Apple Silicon
        model_repo = "mlx-community/whisper-large-v3-turbo"
        resultado = mlx_whisper.transcribe(
            audio_file,
            path_or_hf_repo=model_repo,
            language="es",
            condition_on_previous_text=False,  # CRÍTICO: Evita bucles de corte y silencio
            no_speech_threshold=0.75,          # Evita descartar pausas del profesor o voz lejana
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            initial_prompt="Transcripción continua y detallada de clase o audio en español."
        )
        return resultado.get("text", "").strip()
    else:
        # PyTorch OpenAI Whisper (para Windows CUDA o CPU)
        audio = whisper.load_audio(audio_file)
        options = dict(
            language="es",
            beam_size=5,
            best_of=5,
            fp16=(device == "cuda"),
            condition_on_previous_text=False,  # CRÍTICO: Evita cortes en cadena
            no_speech_threshold=0.75,          # Evita descartar voz lejana o pausas
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            initial_prompt="Transcripción continua y detallada de clase o audio en español."
        )
        resultado = model.transcribe(audio, **options)
        
        # Limpieza de memoria si aplica
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
            
        return resultado.get("text", "").strip()


def transcribir_audios():
    """Transcribe todos los archivos WAV en la carpeta destino."""
    audios = [f for f in os.listdir(ruta_carpeta) if f.lower().endswith('.wav')]
    if not audios:
        print(f"No se encontraron audios WAV para transcribir en: {ruta_carpeta}")
        return

    print(f"\nIniciando transcripción de {len(audios)} archivo(s)...")
    for nombre_archivo in audios:
        audio_file = os.path.join(ruta_carpeta, nombre_archivo)
        base_name, _ = os.path.splitext(nombre_archivo)
        ruta_txt = os.path.join(ruta_carpeta, f"{base_name}.txt")

        # Evitar re-transcribir si ya existe el .txt
        if os.path.exists(ruta_txt):
            print(f"Saltando {nombre_archivo} (el archivo {base_name}.txt ya existe)")
            continue

        print(f"\n🎙️ Transcribiendo: {nombre_archivo}...")
        try:
            texto = transcribir_archivo(audio_file)
            
            with open(ruta_txt, 'w', encoding='utf-8') as f:
                f.write(texto)
                
            print(f"✅ Transcripción completada: {ruta_txt}")

        except Exception as e:
            print(f"❌ Error transcribiendo {nombre_archivo}: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Iniciando procesamiento de audios y transcripción con MLX/Whisper")
    print(f"Carpeta de entrada (Videos/Audios): {ruta_descagar}")
    print(f"Carpeta de salida (Audios/TXT):     {ruta_carpeta}")
    print("=" * 60)

    # 1. Extraer audios de videos
    procesar_videos_en_descargas()

    # 2. Transcribir audios a texto
    transcribir_audios()
