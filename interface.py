import streamlit as st
import os, json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv

load_dotenv()
ARQUIVO_DB = "banco_de_dados.json"
ARQUIVO_MEMORIA = "memoria_consultas.json" # ARQUIVO DE CACHE

# 1. Funções de Memória (Cache)
def carregar_memoria():
    if not os.path.exists(ARQUIVO_MEMORIA):
        return {}
    with open(ARQUIVO_MEMORIA, "r", encoding="utf-8") as f:
        return json.load(f)

def salvar_na_memoria(pergunta, resposta, fontes, metricas):
    memoria = carregar_memoria()
    memoria[pergunta.lower()] = {
        "resposta": resposta,
        "fontes": list(fontes),
        "total_tokens": metricas["total_tokens"],
        "custo": metricas["total_cost"]
    }
    with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=4)

# 2. Configurações da Página
st.set_page_config(page_title="RAG PDF Expert - Com Memória", layout="wide")

# 3. Sidebar
with st.sidebar:
    st.title("🧠 Memória do Agente")
    memoria_atual = carregar_memoria()
    st.write(f"Perguntas memorizadas: **{len(memoria_atual)}**")
    if st.button("Limpar Memória"):
        if os.path.exists(ARQUIVO_MEMORIA): os.remove(ARQUIVO_MEMORIA)
        st.success("Memória limpa!")
        st.rerun()

st.title("🤖 Agente RAG com Memória Inteligente")

# 4. Inicializa o Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Interação
if prompt := st.chat_input("Pergunte algo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # --- VERIFICAÇÃO NA MEMÓRIA ---
    memoria = carregar_memoria()
    if prompt.lower() in memoria:
        cached = memoria[prompt.lower()]
        output = cached["resposta"]
        is_cached = True
        fontes_consultadas = cached["fontes"]
        total_tokens = cached["total_tokens"]
        custo = cached["custo"]
    else:
        # --- CONSULTA REAL RAG ---
        with st.chat_message("assistant"):
            with st.spinner("IA Pensando (Consulta Original)..."):
                vectorstore = InMemoryVectorStore.load(ARQUIVO_DB, OpenAIEmbeddings())
                resultados = vectorstore.similarity_search(prompt, k=25)
                contexto = "\n\n".join([doc.page_content for doc in resultados])
                
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                cp = ChatPromptTemplate.from_template("Use: {contexto}\n\nQ: {pergunta}")
                chain = cp | llm
                
                with get_openai_callback() as cb:
                    res = chain.invoke({"contexto": contexto, "pergunta": prompt})
                    output = res.content
                    total_tokens = cb.total_tokens
                    custo = cb.total_cost

                # Extrai fontes
                fontes_consultadas = []
                for d in resultados:
                    fontes_consultadas.append(f"📄 {os.path.basename(d.metadata.get('source','?'))} (pág {d.metadata.get('page','?')})")
                
                # Salva nova consulta na memória
                salvar_na_memoria(prompt, output, set(fontes_consultadas), {"total_tokens": total_tokens, "total_cost": custo})
                is_cached = False

    # Exibe a resposta
    with st.chat_message("assistant"):
        if 'is_cached' in locals() and is_cached:
            st.info("⚡ Resposta recuperada da Memória (Custo Zero)")
        st.markdown(output)
        
        with st.sidebar:
            st.write(f"🔹 **Métrica:** {total_tokens} tokens")
            for f in sorted(list(set(fontes_consultadas))): st.write(f)

    st.session_state.messages.append({"role": "assistant", "content": output})
