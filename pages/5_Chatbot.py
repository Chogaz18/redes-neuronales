import streamlit as st
def generate_answer(prompt):
    return f"Respuesta basada en ECG:\n\n{prompt}"

# ================= CONFIG =================
st.set_page_config(page_title="Chatbot ECG", layout="wide")

# ================= ESTILOS =================
st.markdown("""
<style>

.stApp {
    background-color: #f5f7fa;
}

.main-container {
    max-width: 900px;
    margin: auto;
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.chat-header {
    position: sticky;
    top: 0;
    z-index: 999;
    background: linear-gradient(135deg, #003d82 0%, #0056b3 100%);
    padding: 20px;
    color: white;
}

.chat-body {
    height: 520px;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
}

.msg-user {
    text-align: right;
    margin: 10px 0;
}

.msg-user div {
    background: #003d82;
    color: white;
    padding: 10px;
    border-radius: 12px;
    display: inline-block;
}

.msg-bot {
    text-align: left;
    margin: 10px 0;
}

.msg-bot div {
    background: #e9eef6;
    padding: 10px;
    border-radius: 12px;
    display: inline-block;
}

</style>
""", unsafe_allow_html=True)

# ================= ESTADO =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= LEER DATOS ECG =================
hr = st.session_state.get("hr", None)
rr = st.session_state.get("rr_mean", None)
n_rpeaks = st.session_state.get("n_rpeaks", None)
record_id = st.session_state.get("record_id", "No seleccionado")

# ================= HEADER =================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown(f"""
<div class="chat-header">
    <h2>Chatbot ECG</h2>
    <p>Registro: {record_id}</p>
    <p>HR: {f"{hr:.2f} bpm" if hr else "Sin datos"}</p>
</div>
""", unsafe_allow_html=True)

# ================= CHAT =================
st.markdown('<div class="chat-body" id="chat-box">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='msg-user'><div>{msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='msg-bot'><div>{msg['content']}</div></div>", unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

# ================= AUTOSCROLL =================
st.markdown("""
<script>
setTimeout(() => {
    const chatBox = document.getElementById("chat-box");
    if (chatBox) {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}, 200);
</script>
""", unsafe_allow_html=True)

# ================= INPUT =================
query = st.chat_input("Pregunta sobre ECG...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    try:
        docs = retrieve(query)
        context = "\n".join([doc.page_content for doc in docs]) if docs else ""

        # ================= DATOS ECG REALES =================
        ecg_context = f"""
Datos del paciente:
- Registro: {record_id}
- Frecuencia cardíaca: {hr:.2f} bpm
- Intervalo RR promedio: {rr:.3f} s
- Número de picos R: {n_rpeaks}
""" if hr else "No hay datos ECG cargados."

        # ================= PROMPT =================
        prompt = f"""
Eres un asistente experto en electrocardiografía.

Contexto:
{context}

Datos ECG en tiempo real:
{ecg_context}

Pregunta:
{query}

Responde de forma clara, clínica y breve.
"""

        response = generate_answer(prompt)

    except Exception as e:
        response = f"Error: {str(e)}"

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()