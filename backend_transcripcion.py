import math
import os
import platform
import re
import subprocess

# ==============================================================================
# Detección del Motor de Transcripción y Acelerador de Hardware
# ==============================================================================
USE_MLX = False
try:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import mlx_whisper
        USE_MLX = True
        print("🚀 backend_transcripcion: Usando Apple Silicon MLX (GPU + Neural Engine)")
except ImportError:
    pass

if not USE_MLX:
    import torch
    import whisper
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 backend_transcripcion: Usando aceleración NVIDIA CUDA")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🚀 backend_transcripcion: Usando aceleración Apple Silicon MPS")
    else:
        device = "cpu"
        print("⚠️ backend_transcripcion: Usando CPU")

    print(f"Cargando modelo Whisper 'turbo' en {device}...")
    model = whisper.load_model("turbo", device=device)


def obtener_duracion_video(video_path):
    """Obtiene la duración en segundos."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        try:
            cmd = ['ffmpeg', '-i', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.PIPE)
            duration_pattern = r"Duration: (\d{2}):(\d{2}):(\d{2}).(\d{2})"
            matches = re.search(duration_pattern, result.stderr)
            if matches:
                hours, minutes, seconds, centiseconds = map(int, matches.groups())
                return hours * 3600 + minutes * 60 + seconds + centiseconds / 100
        except Exception as e:
            print(f"Error obteniendo duración: {e}")
    return 0


def dividir_por_tiempo(file_path, duracion_chunk_min=10):
    """Divide el archivo en segmentos de X minutos en formato WAV 16k mono."""
    duracion_total = obtener_duracion_video(file_path)
    if duracion_total == 0:
        return [file_path]

    chunk_seconds = duracion_chunk_min * 60
    num_partes = math.ceil(duracion_total / chunk_seconds)
    
    if num_partes <= 1:
        # Extraer a WAV 16kHz mono directo
        base_name, _ = os.path.splitext(file_path)
        output_name = f"{base_name}_temp16k.wav"
        cmd = [
            'ffmpeg', '-y', '-i', file_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_name
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return [output_name]
        except Exception:
            return [file_path]

    print(f"Dividiendo audio de {duracion_total:.2f}s en {num_partes} partes de {duracion_chunk_min} min...")
    
    base_name, _ = os.path.splitext(file_path)
    chunk_paths = []
    
    for i in range(num_partes):
        start_time = i * chunk_seconds
        output_name = f"{base_name}_part{i:03d}.wav"
        
        cmd = [
            'ffmpeg', '-y', '-i', file_path, 
            '-ss', str(start_time), 
            '-t', str(chunk_seconds),
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 
            output_name
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_paths.append(output_name)
        except subprocess.CalledProcessError as e:
            print(f"Error extrayendo parte {i}: {e}")
            
    return chunk_paths


def transcribir_segmento(audio_path):
    """Transcribe un segmento de audio usando MLX o PyTorch con parámetros anti-cortes."""
    if USE_MLX:
        res = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            language="es",
            condition_on_previous_text=False,
            no_speech_threshold=0.75,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            initial_prompt="Transcripción continua y detallada de clase o audio en español."
        )
        return res.get("text", "").strip()
    else:
        options = dict(
            language="es",
            beam_size=5,
            best_of=5,
            fp16=(device == "cuda"),
            condition_on_previous_text=False,
            no_speech_threshold=0.75,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            initial_prompt="Transcripción continua y detallada de clase o audio en español."
        )
        audio = whisper.load_audio(audio_path)
        res = model.transcribe(audio, **options)
        
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
            
        return res.get("text", "").strip()


def procesar_entrada_con_callback(file_path, progress_callback=None):
    """
    Procesa el archivo dividiéndolo y reportando progreso a la UI.
    progress_callback: función que acepta (int porcentaje, str mensaje)
    """
    files_to_cleanup = []
    texto_completo = ""
    
    try:
        if progress_callback:
            progress_callback(5, "Analizando duración y preparando audio...")
        
        partes = dividir_por_tiempo(file_path, duracion_chunk_min=10)
        
        if partes != [file_path]:
            files_to_cleanup.extend(partes)
            
        total_partes = len(partes)
        
        for idx, parte in enumerate(partes):
            if progress_callback: 
                progreso = 10 + int((idx / total_partes) * 80)
                progress_callback(progreso, f"Transcribiendo parte {idx+1} de {total_partes}...")
            
            texto_parte = transcribir_segmento(parte)
            texto_completo += texto_parte + "\n\n"

    except Exception as e:
        texto_completo += f"\n[ERROR: {str(e)}]"
        print(f"Error crítico: {e}")
    
    finally:
        if progress_callback:
            progress_callback(95, "Finalizando limpieza...")
        for f in files_to_cleanup:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    return texto_completo.strip()