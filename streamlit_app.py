import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import wget
import zipfile
import os
import contextlib

# Mostrar la versión de TensorFlow
st.write("## Clasificación de Neumonía en Rayos X Pediátricos")
st.write("Examinar imágenes de radiografías y identificar señales de neumonía.")

# Descargar y descomprimir el modelo si no existe
def download_and_extract_model():
    model_url = 'https://dl.dropboxusercontent.com/s/rq0m4fazavr3hitf9tfx0/best_model.zip?rlkey=vl3svfzhwi55bju9crcp91nqd&st=h15bg8d7&'
    zip_path = 'best_model.zip'
    extract_folder = 'extracted_files'

    # Descargar el archivo zip si no existe
    if not os.path.exists(zip_path):
        try:
            wget.download(model_url, zip_path)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")
            return None

    # Descomprimir el archivo
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    return os.path.join(extract_folder, 'best_model_local.keras')  # Asegúrate de que este sea el nombre correcto

modelo_path = download_and_extract_model()

# Verificar si el archivo del modelo existe
if modelo_path is None or not os.path.exists(modelo_path):
    st.error("No se encontró el archivo del modelo")
else:
    st.success("Archivo del modelo encontrado")

# Cargar el modelo entrenado directamente
try:
    model = load_model(modelo_path)  # Cargar el modelo sin especificar la arquitectura
    st.success("Modelo cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Sube una radiografía.", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None and model is not None:
    # Mostrar la imagen subida
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=(224,224))  # Asegúrate de que el tamaño coincida con lo que tu modelo espera
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            prediction = model.predict(img_array)

    # Mostrar resultados
    if prediction[0][0] > 0.5:
        st.success("El modelo predice que la imagen *no* muestra *signos de neumonía*.")
    else:
        st.error("El modelo predice que la imagen muestra **signos de neumonía**.")
        
    # Recomendación
    st.write("## Sugerencia:")
    st.write("Consulta a un profesional de la salud para obtener un diagnóstico más preciso.")