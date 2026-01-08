import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Configuración de la página (Estética)
st.set_page_config(page_title="Clasificador de Textos Clásicos", page_icon="📜", layout="centered")

# Estilo CSS personalizado para mejorar la ergonomía visual
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #2e4053; color: white; }
    .result-box { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 2. Cargar el modelo (Usamos cache para que no se recargue cada vez que el usuario haga clic)
@st.cache_resource
def cargar_modelo():
    ruta = "./modelo_final"
    tokenizer = AutoTokenizer.from_pretrained(ruta)
    model = AutoModelForSequenceClassification.from_pretrained(ruta)
    return tokenizer, model

tokenizer, model = cargar_modelo()
id2label = model.config.id2label

# 3. Interfaz de Usuario
st.title("📜 Clasificación Semántica de Textos Clásicos")
st.markdown("Esta herramienta utiliza Inteligencia Artificial para categorizar fragmentos literarios en tres ejes: **Areté**, **Política** o **Divinidad**.")

st.divider()

# Cuadro de entrada de texto
texto_usuario = st.text_area("Pegue aquí el fragmento del texto clásico:", placeholder="Ej: 'Canta, oh musa, la cólera de Aquiles...'")

if st.button("Analizar Fragmento"):
    if texto_usuario.strip() == "":
        st.warning("Por favor, ingrese un texto para analizar.")
    else:
        # --- PROCESO DE INFERENCIA ---
        with st.spinner("El modelo está analizando los matices semánticos..."):
            # Tokenización
            inputs = tokenizer(texto_usuario, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Predicción
            with torch.no_grad():
                outputs = model(**inputs)
                probabilidades = F.softmax(outputs.logits, dim=1)
                confianza, prediccion_id = torch.max(probabilidades, dim=1)
            
            categoria = id2label[prediccion_id.item()]
            porcentaje = confianza.item() * 100

        # --- PRESENTACIÓN DE RESULTADOS ---
        st.subheader("Resultado del Análisis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Categoría Predicha:**")
            st.info(f"### {categoria}")
        
        with col2:
            st.markdown(f"**Nivel de Confianza:**")
            st.metric(label="Probabilidad", value=f"{porcentaje:.1f}%")

        # Visualización de todas las probabilidades para interpretabilidad
        st.write("---")
        st.write("**Distribución de probabilidad por categoría:**")
        for i, prob in enumerate(probabilidades[0]):
            nombre_cat = id2label[i]
            valor_prob = prob.item()
            st.write(f"{nombre_cat}")
            st.progress(valor_prob)

st.sidebar.title("Sobre el Proyecto")
st.sidebar.info("Sistema basado en BERT para investigadores de Humanidades. Desarrollado como parte de la clasificación de textos clásicos.")