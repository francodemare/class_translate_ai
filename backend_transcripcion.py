import whisper
import torch
import os
import subprocess
import re
import math

# --- Configuración Inicial ---
device = "cuda" if torch.cuda.is_available() else "cpu"
options = dict(language="es", beam_size=5, best_of=5)

print(f"Cargando modelo Whisper en {device}...")
model = whisper.load_model("turbo", device=device)

def obtener_duracion_video(video_path):
    """Obtiene la duración en segundos."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        # Fallback a ffmpeg si ffprobe falla
        try:
            cmd = ['ffmpeg', '-i', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.PIPE)
            duration_pattern = r"Duration: (\d{2}):(\d{2}):(\d{2}).(\d{2})"
            matches = re.search(duration_pattern, result.stderr)
            if matches:
                hours, minutes, seconds, centiseconds = map(int, matches.groups())
                return hours * 3600 + minutes * 60 + seconds + centiseconds/100
        except Exception as e:
            print(f"Error obteniendo duración: {e}")
    return 0

def dividir_por_tiempo(file_path, duracion_chunk_min=10):
    """Divide el archivo en segmentos de X minutos."""
    duracion_total = obtener_duracion_video(file_path)
    if duracion_total == 0:
        return [file_path] # Si falla la detección, intenta procesar entero

    chunk_seconds = duracion_chunk_min * 60
    num_partes = math.ceil(duracion_total / chunk_seconds)
    
    # Si es más corto que el chunk, devolver original
    if num_partes <= 1:
        return [file_path]

    print(f"Dividiendo audio de {duracion_total}s en {num_partes} partes de {duracion_chunk_min} min...")
    
    base_name, _ = os.path.splitext(file_path)
    chunk_paths = []
    
    for i in range(num_partes):
        start_time = i * chunk_seconds
        output_name = f"{base_name}_part{i:03d}.wav"
        
        # Extraer segmento y convertir a WAV ligero (16k mono)
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

def procesar_entrada_con_callback(file_path, progress_callback=None):
    """
    Procesa el archivo dividiéndolo y reportando progreso a la UI.
    progress_callback: función que acepta (int porcentaje, str mensaje)
    """
    files_to_cleanup = []
    texto_completo = ""
    
    try:
        if progress_callback: progress_callback(5, "Analizando duración y dividiendo...")
        
        # 1. Dividir en chunks de 10 minutos
        partes = dividir_por_tiempo(file_path, duracion_chunk_min=10)
        
        # Marcar para limpieza si se generaron archivos nuevos
        if partes != [file_path]:
            files_to_cleanup.extend(partes)
            
        total_partes = len(partes)
        
        # 2. Procesar cada parte
        for idx, parte in enumerate(partes):
            if progress_callback: 
                progreso = 10 + int((idx / total_partes) * 80)
                progress_callback(progreso, f"Transcribiendo parte {idx+1} de {total_partes}...")
            
            # Transcribir
            audio = whisper.load_audio(parte)
            result = model.transcribe(audio, **options)
            texto_completo += result["text"] + "\n\n"
            
            # Limpiar memoria de CUDA por seguridad entre chunks
            if device == "cuda":
                torch.cuda.empty_cache()

    except Exception as e:
        texto_completo += f"\n[ERROR: {str(e)}]"
        print(f"Error crítico: {e}")
    
    finally:
        if progress_callback: progress_callback(95, "Finalizando limpieza...")
        # 3. Limpieza
        for f in files_to_cleanup:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass

    return texto_completo