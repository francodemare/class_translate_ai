import os
from pydub import AudioSegment


def unificar_audios(ruta_carpeta, archivo_salida):
    """Combina todos los archivos MP3 de una carpeta en un único archivo MP3."""
    if not os.path.exists(ruta_carpeta):
        print(f"Carpeta no encontrada: {ruta_carpeta}")
        return

    archivos_mp3 = sorted([f for f in os.listdir(ruta_carpeta) if f.lower().endswith(".mp3")])
    if not archivos_mp3:
        print(f"No se encontraron archivos MP3 en: {ruta_carpeta}")
        return

    print(f"Unificando {len(archivos_mp3)} archivos MP3...")
    audio_unificado = AudioSegment.empty()

    for nombre_archivo in archivos_mp3:
        audio_file = os.path.join(ruta_carpeta, nombre_archivo)
        print(f"Agregando: {nombre_archivo}")
        audio = AudioSegment.from_mp3(audio_file)
        audio_unificado += audio

    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(os.path.abspath(archivo_salida)), exist_ok=True)
    audio_unificado.export(archivo_salida, format="mp3")
    print(f"✅ Audio unificado guardado exitosamente en: {archivo_salida}")


if __name__ == "__main__":
    HOME = os.path.expanduser("~")
    ruta_carpeta = os.path.join(HOME, "Desktop", "audio")
    archivo_salida = os.path.join(HOME, "Desktop", "audio_unificado.mp3")
    unificar_audios(ruta_carpeta, archivo_salida)