import os
import platform
import subprocess

# Detección de MLX en Apple Silicon vs PyTorch en CUDA/CPU
USE_MLX = False
try:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import mlx_whisper
        USE_MLX = True
        print("🚀 generate_sub: Usando Apple Silicon MLX (GPU + Neural Engine)")
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
        print("🚀 Usando aceleración Apple Silicon MPS")
    else:
        device = "cpu"
        print("⚠️ Usando CPU")
    print(f"Cargando modelo Whisper 'turbo' en {device}...")
    model = whisper.load_model("turbo", device=device)


def format_timestamp(seconds: float):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


# Ruta del archivo de video
video_path = "video/video_russia.mp4"

if os.path.exists(video_path):
    print(f"Extrayendo audio de {video_path} con ffmpeg...")
    base_name, _ = os.path.splitext(video_path)
    audio_output_path = f"{base_name}.wav"
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        audio_output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Audio extraído y guardado en {audio_output_path}")

    print("Transcribiendo audio...")
    if USE_MLX:
        result = mlx_whisper.transcribe(
            audio_output_path,
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            language="es",
            condition_on_previous_text=False,
            no_speech_threshold=0.75,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4
        )
    else:
        options = dict(
            language="es",
            fp16=(device == "cuda"),
            condition_on_previous_text=False,
            no_speech_threshold=0.75,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4
        )
        result = model.transcribe(audio_output_path, **options)

    output_srt = "subtitulos.srt"
    write_srt(result.get("segments", []), output_srt)
    print(f"✅ Subtítulos generados con éxito y guardados en {output_srt}")
else:
    print(f"Archivo no encontrado: {video_path}")