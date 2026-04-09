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

# 4. Inicializa o Chat e Métricas
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"total": 0, "prompt": 0, "completion": 0, "cost": 0.0}
if "sources" not in st.session_state:
    st.session_state.sources = []

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Sidebar (Atualizada com Métricas)
with st.sidebar:
    st.title("🧠 Memória e Uso")
    memoria_atual = carregar_memoria()
    st.write(f"Perguntas memorizadas: **{len(memoria_atual)}**")
    
    if memoria_atual:
        pergunta_selecionada = st.selectbox(
            "📜 Histórico de Perguntas",
            options=["Selecione para ver..."] + list(memoria_atual.keys()),
            index=0
        )
        if pergunta_selecionada != "Selecione para ver...":
            detalhes = memoria_atual[pergunta_selecionada]
            st.info(f"**Resposta memorizada:**\n\n{detalhes['resposta'][:200]}...")
            if st.button("Usar esta pergunta"):
                st.session_state.prompt_from_history = pergunta_selecionada
                st.rerun()
    
    st.divider()
    st.subheader("📊 Métricas da Última Consulta")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tokens Total", st.session_state.metrics["total"])
        st.metric("Tokens Prompt", st.session_state.metrics["prompt"])
    with col2:
        st.metric("Custo (USD)", f"${st.session_state.metrics['cost']:.4f}")
        st.metric("Tokens Resposta", st.session_state.metrics["completion"])
    
    st.divider()
    st.subheader("📚 Fontes Consultadas")
    if st.session_state.sources:
        for f in st.session_state.sources:
            st.write(f)
    else:
        st.write("Nenhuma fonte consultada ainda.")

    st.divider()
    if st.button("Limpar Memória", use_container_width=True):
        if os.path.exists(ARQUIVO_MEMORIA): os.remove(ARQUIVO_MEMORIA)
        st.session_state.metrics = {"total": 0, "prompt": 0, "completion": 0, "cost": 0.0}
        st.session_state.sources = []
        st.success("Memória limpa!")
        st.rerun()

st.title("🤖 Agente RAG com Memória Inteligente")

# 5. Interação (Chat Input ou Histórico)
prompt = st.chat_input("Pergunte algo...")

# Se veio uma pergunta do dropdown de histórico
if "prompt_from_history" in st.session_state and st.session_state.prompt_from_history:
    prompt = st.session_state.prompt_from_history
    st.session_state.prompt_from_history = None # Limpa para não entrar em loop

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # --- VERIFICAÇÃO NA MEMÓRIA ---
    memoria = carregar_memoria()
    if prompt.lower() in memoria:
        cached = memoria[prompt.lower()]
        output = cached["resposta"]
        is_cached = True
        st.session_state.sources = sorted(list(set(cached["fontes"])))
        
        # Resposta da memória tem custo zero de API
        st.session_state.metrics = {
            "total": 0,
            "prompt": 0,
            "completion": 0,
            "cost": 0.0
        }
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
                    
                    # Atualiza métricas no session_state
                    st.session_state.metrics = {
                        "total": cb.total_tokens,
                        "prompt": cb.prompt_tokens,
                        "completion": cb.completion_tokens,
                        "cost": cb.total_cost
                    }

                # Extrai fontes
                fontes = []
                for d in resultados:
                    fontes.append(f"📄 {os.path.basename(d.metadata.get('source','?'))} (pág {d.metadata.get('page','?')})")
                st.session_state.sources = sorted(list(set(fontes)))
                
                # Salva nova consulta na memória
                salvar_na_memoria(prompt, output, set(st.session_state.sources), {
                    "total_tokens": st.session_state.metrics["total"], 
                    "total_cost": st.session_state.metrics["cost"]
                })
                is_cached = False

    # Exibe a resposta
    with st.chat_message("assistant"):
        if 'is_cached' in locals() and is_cached:
            st.info("⚡ Resposta recuperada da Memória (Custo Zero)")
        st.markdown(output)

    st.session_state.messages.append({"role": "assistant", "content": output})
    st.rerun() # Rerun para atualizar a sidebar com as novas métricas imediatamente

