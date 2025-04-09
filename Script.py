import streamlit as st
import cohere
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key)

# Configuración general
st.set_page_config(page_title="Chatbot - Oficinas TI", layout="centered")

# Estilos CSS (inspirado en WhatsApp)
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #ece5dd;
        color: #303030;
    }
    .message-bubble {
        border-radius: 16px;
        padding: 12px 18px;
        margin-bottom: 10px;
        max-width: 80%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-size: 14px;
    }
    .user-message {
        background-color: #dcf8c6;
        align-self: flex-end;
    }
    .bot-message {
        background-color: #ffffff;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .header-icon {
        font-size: 1.5rem;
        color: #075e54;
    }
    </style>
""", unsafe_allow_html=True)

# Cabecera estilo WhatsApp
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='margin-bottom: 0.2rem; color: #075e54;'>
            <span class="header-icon">💻</span> Oficinas de Tecnologías de Información
        </h1>
        <p style='color: #6c757d;'>Asistente inteligente potenciado por Cohere y RAG</p>
    </div>
""", unsafe_allow_html=True)

# Funciones
def leer_pdf(pdf_file):
    texto = ""
    with fitz.open(pdf_file) as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto

def dividir_en_chunks(texto, tamaño=300):
    palabras = texto.split()
    return [" ".join(palabras[i:i+tamaño]) for i in range(0, len(palabras), tamaño)]

def encontrar_chunk_mas_relevante(chunks, pregunta):
    vectorizer = TfidfVectorizer().fit([pregunta] + chunks)
    vectores = vectorizer.transform([pregunta] + chunks)
    similitudes = cosine_similarity(vectores[0:1], vectores[1:]).flatten()
    indice = similitudes.argmax()
    return chunks[indice]

# Subida de archivo
archivo = st.file_uploader("📄 Subir documento PDF", type=["pdf"])
documento_texto = ""
chunks = []

if archivo is not None:
    with open("temp.pdf", "wb") as f:
        f.write(archivo.read())
    documento_texto = leer_pdf("temp.pdf")
    chunks = dividir_en_chunks(documento_texto)
    st.success("✅ Documento cargado correctamente.")

# Estado de la conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes estilo WhatsApp
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    clase = "user-message" if msg["role"] == "user" else "bot-message"
    st.markdown(
        f"<div class='message-bubble {clase}'>{msg['content']}</div>",
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# Entrada del usuario
prompt = st.chat_input("Haz una pregunta sobre el documento...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    chat_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages]
    )

    contexto = ""
    if chunks:
        contexto = encontrar_chunk_mas_relevante(chunks, prompt)

    prompt_final = f"""Usa la siguiente información del documento para responder la pregunta del usuario:

{contexto}

Conversación:
{chat_history}

Respuesta:"""

    # Mostrar el GIF mientras se procesa la respuesta
    st.markdown(f"""<div class='running-man'><img src="https://i.gifer.com/embedded/download/B0J6.gif" width="50" height="50"></div>""", unsafe_allow_html=True)

    # Mostrar el spinner mientras se genera la respuesta
    with st.spinner("💬 Pensando..."):
        response = co.generate(
            model="command-xlarge",
            prompt=prompt_final,
            max_tokens=200,
            temperature=0.7,
        )
        reply = response.generations[0].text.strip()

    # Mostrar la respuesta del bot
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
