import streamlit as st
import os
import backend_transcripcion as backend # Importamos tu script de arriba

# --- BLOQUE DE SEGURIDAD ---
def check_password():
    """Retorna True si el usuario ingresó la contraseña correcta."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # Input de contraseña
    pwd = st.text_input("Ingresa la clave de acceso:", type="password")
    
    if st.button("Entrar"):
        if pwd == "admin_123":  # <--- CAMBIA ESTO
            st.session_state["password_correct"] = True
            st.rerun()  # Recarga la página para mostrar el contenido
        else:
            st.error("Contraseña incorrecta")
    return False

if not check_password():
    st.stop()  # Detiene la ejecución aquí si no hay login

# Configuración de la página
st.set_page_config(page_title="Transcriptor Pro", layout="wide")
st.title("🎙️ Transcriptor Local (Soporte Archivos Largos)")

TEMP_DIR = "temp_uploads"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

# Aumentar visualmente el límite en el mensaje (Streamlit soporta 200MB por defecto)
archivo_subido = st.file_uploader("Sube audio/video", type=None, help="Límite default: 200MB")

if archivo_subido is not None:
    if st.button("Comenzar Transcripción"):
        ruta_archivo = os.path.join(TEMP_DIR, archivo_subido.name)
        
        # Guardado del archivo
        with open(ruta_archivo, "wb") as f:
            f.write(archivo_subido.getbuffer())
        
        # Contenedores para feedback visual
        barra_progreso = st.progress(0)
        status_text = st.empty()
        
        # Función callback para actualizar la barra desde el backend
        def actualizar_progreso(porcentaje, mensaje):
            barra_progreso.progress(porcentaje)
            status_text.text(mensaje)

        try:
            # Llamada con callback
            resultado = backend.procesar_entrada_con_callback(ruta_archivo, actualizar_progreso)
            
            barra_progreso.progress(100)
            status_text.success("¡Completado!")
            
            st.text_area("Resultado:", value=resultado, height=400)
            st.download_button("Descargar .txt", resultado, file_name=f"{archivo_subido.name}.txt")
            
        except Exception as e:
            st.error(f"Error en la app: {e}")
        
        finally:
            if os.path.exists(ruta_archivo):
                os.remove(ruta_archivo)