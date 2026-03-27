from pydub import AudioSegment
import os

def unificar_audios(ruta_carpeta, archivo_salida):
    # Crear un objeto AudioSegment vacío
    audio_unificado = AudioSegment.empty()
    
    # Iterar sobre los archivos en la carpeta
    for nombre_archivo in os.listdir(ruta_carpeta):
        if nombre_archivo.endswith(".mp3"):
            # Unir la ruta del archivo
            audio_file = os.path.join(ruta_carpeta, nombre_archivo)
            # Cargar el archivo de audio
            audio = AudioSegment.from_mp3(audio_file)
            # Añadir el audio al audio unificado
            audio_unificado += audio
    
    # Exportar el audio unificado a un archivo MP3
    audio_unificado.export(archivo_salida, format="mp3")
    print(f"Audio unificado guardado en: {archivo_salida}")

# Ejemplo de uso
ruta_carpeta = "C:\\Users\\frand\\OneDrive\\Escritorio\\audio"
archivo_salida = "C:\\Users\\frand\\OneDrive\\Escritorio\\audio_unificado.mp3"
unificar_audios(ruta_carpeta, archivo_salida)