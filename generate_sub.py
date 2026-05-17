import whisper

def format_timestamp(seconds: float):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def write_srt(result, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

import os
from pydub import AudioSegment

print("Cargando modelo Whisper...")
model = whisper.load_model("turbo")

# Reemplaza 'video.mp4' por el nombre real de tu archivo de video
video_path = "video/IMG_0738.MOV"

print(f"Extrayendo audio de {video_path}...")
base_name, _ = os.path.splitext(video_path)
audio_output_path = f"{base_name}.wav"
video = AudioSegment.from_file(video_path)
video.export(audio_output_path, format='wav', parameters=["-ac", "1", "-ar", "16000"])
print(f"Audio extraído y guardado en {audio_output_path}")

print(f"Transcribiendo {audio_output_path}...")
result = model.transcribe(audio_output_path)

output_srt = "subtitulos.srt"
write_srt(result, output_srt)
print(f"Subtítulos generados con éxito y guardados en {output_srt}")