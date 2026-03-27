import whisper
import torch
import os
import subprocess
from pydub import AudioSegment

# Configuración de la GPU
device = "cuda"
options = dict(language="es", beam_size=5, best_of=5)

# Obtenemos la ruta del documentos
ruta_carpeta = "C:\\Users\\frand\\Documents\\audios_clases"
ruta_descagar = "C:\\Users\\frand\\Downloads\\clases" 

# Transcripción
model = whisper.load_model("turbo")

def dividir_audio(audio_segment, nombre_base, extension):
    """Divide un AudioSegment en dos partes iguales y guarda cada parte."""
    duracion = len(audio_segment)
    mitad = duracion // 2
    
    parte1 = audio_segment[:mitad]
    parte2 = audio_segment[mitad:]
    
    nombre_parte1 = f"{nombre_base}_parte1{extension}"
    nombre_parte2 = f"{nombre_base}_parte2{extension}"
    
    return [(parte1, nombre_parte1), (parte2, nombre_parte2)]

def obtener_duracion_video(video_path):
    """Obtiene la duración del video en segundos usando ffmpeg."""
    import subprocess
    import json
    import re
    
    try:
        # Primero intentamos obtener la duración directamente con ffprobe
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duracion = float(result.stdout.strip())
        return duracion
        
    except subprocess.CalledProcessError:
        try:
            # Si ffprobe falla, intentamos con ffmpeg
            cmd = ['ffmpeg', '-i', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.PIPE)
            
            # Buscar la duración en la salida de error de ffmpeg
            duration_pattern = r"Duration: (\d{2}):(\d{2}):(\d{2}).(\d{2})"
            matches = re.search(duration_pattern, result.stderr)
            
            if matches:
                hours = int(matches.group(1))
                minutes = int(matches.group(2))
                seconds = int(matches.group(3))
                centiseconds = int(matches.group(4))
                
                total_seconds = hours * 3600 + minutes * 60 + seconds + centiseconds/100
                return total_seconds
                
        except Exception as e:
            print(f"Error obteniendo duración del video con ffmpeg: {e}")
            
    return 0

def dividir_archivo(file_path, chunk_size=2*1024*1024*1024):
    """Divide un video en partes iguales usando ffmpeg."""
    file_size = os.path.getsize(file_path)
    if file_size <= chunk_size:
        return [file_path]

    base_name, ext = os.path.splitext(file_path)
    duracion_total = obtener_duracion_video(file_path)
    
    if duracion_total == 0:
        print(f"No se pudo obtener la duración del video: {file_path}")
        return [file_path]
    
    print(f"Duración total del video: {duracion_total} segundos")
    
    # Dividir en dos partes iguales
    mitad_duracion = duracion_total / 2
    chunk_paths = []
    
    for i in range(2):
        inicio = i * mitad_duracion
        chunk_path = f"{base_name}_parte{i}{ext}"
        duracion = mitad_duracion if i == 0 else (duracion_total - inicio)
        
        print(f"Extrayendo parte {i+1} desde {inicio} hasta {inicio + duracion} segundos")
        
        # Usar ffmpeg para dividir el video directamente a audio
        output_audio = chunk_path.replace(ext, '.wav')
        cmd = [
            'ffmpeg', '-y',
            '-i', file_path,
            '-ss', str(inicio),
            '-t', str(duracion),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Formato WAV
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Mono
            output_audio
        ]
        
        try:
            print(f"Ejecutando comando: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            chunk_paths.append(output_audio)
            print(f"Parte {i+1} extraída exitosamente: {output_audio}")
        except subprocess.CalledProcessError as e:
            print(f"Error dividiendo video: {e}")
            print(f"Error output: {e.output}")
            return [file_path]
    
    return chunk_paths

def extraer_audio(video_path, audio_path):
    """ Extrae el audio de un archivo de video y lo guarda como archivo de audio.
    Si el archivo es mayor a 4GB, lo divide en partes antes de procesarlo. """
    try:
        # Verificar tamaño del archivo
        tamano_archivo = os.path.getsize(video_path)
        TRES_GB = 3 * 1024 * 1024 * 1024  # 3GB en bytes
        
        archivos_resultantes = []
        if tamano_archivo > TRES_GB:
            print(f"Archivo {video_path} es mayor a 4GB, dividiendo en partes...")
            partes_video = dividir_archivo(video_path)
            
            for idx, parte_video in enumerate(partes_video):
                nombre_base, extension = os.path.splitext(audio_path)
                audio_output = f"{nombre_base}_parte{idx}{extension}"
                
                # Procesar cada parte del video
                video = AudioSegment.from_file(parte_video)
                video.export(audio_output, format='wav', parameters=["-ac", "1", "-ar", "16000"])
                archivos_resultantes.append(audio_output)
                
                # Limpiar archivo temporal si fue creado
                if parte_video != video_path:
                    os.remove(parte_video)
                    
                print(f"Parte {idx+1} guardada en: {audio_output}")
        else:
            video = AudioSegment.from_file(video_path)
            video.export(audio_path, format='wav', parameters=["-ac", "1", "-ar", "16000"])
            archivos_resultantes.append(audio_path)
            
        return archivos_resultantes
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        return []

def procesar_videos_en_descargas():
    """ Procesa todos los archivos de video en la carpeta de descargas y extrae el audio. """
    for nombre_archivo in os.listdir(ruta_descagar):
        if nombre_archivo.lower().endswith(('.mp4', '.wmv', '.avi', '.mov', '.mkv', '.m4a', 'mp3')):
            video_path = os.path.join(ruta_descagar, nombre_archivo)
            audio_output_path = os.path.join(ruta_carpeta, f"{os.path.splitext(nombre_archivo)[0]}.wav")
            archivos_audio = extraer_audio(video_path, audio_output_path)
            for archivo in archivos_audio:
                print(f"Audio extraído y guardado en: {archivo}")

# Ejemplo de uso
procesar_videos_en_descargas()

# Mueve el modelo a la GPU
if torch.cuda.is_available():
    # Configura el dispositivo en 'cuda'
    model = model.to(device)

for nombre_archivo in os.listdir(ruta_carpeta):
    # Unimos la ruta del escritorio con el nombre del archivo
    if nombre_archivo.lower().endswith(('.wav')):
        audio_file = os.path.join(ruta_carpeta, nombre_archivo)
        audio = whisper.load_audio(audio_file)
        result = model.transcribe(audio, **options)
    
        # Obtener la extensión del archivo
        aux, extension = os.path.splitext(nombre_archivo)
        ruta_completa = os.path.join(ruta_carpeta, nombre_archivo.replace(extension,".txt"))
        with open(ruta_completa, 'w', encoding='utf-8') as file:
            file.write(result["text"])
            
        print("Audio procesado: "+ruta_completa)

