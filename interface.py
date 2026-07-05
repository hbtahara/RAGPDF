import streamlit as st
import os, json, requests, re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from config import *
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks.manager import get_openai_callback

load_dotenv()
ARQUIVO_MEMORIA = MEMORIA_CACHE

# 1. Funções de Memória (Cache)
def carregar_memoria():
    if not os.path.exists(ARQUIVO_MEMORIA):
        return {}
    with open(ARQUIVO_MEMORIA, "r", encoding="utf-8") as f:
        return json.load(f)

def salvar_na_memoria(pergunta, resposta, fontes, metricas):
    memoria = carregar_memoria()
    chave = pergunta.strip().lower()
    memoria[chave] = {
        "resposta": resposta,
        "fontes": list(fontes),
        "total_tokens": metricas["total_tokens"],
        "custo": metricas["total_cost"]
    }
    with open(ARQUIVO_MEMORIA, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=4)

@st.cache_data(ttl=5)
def verificar_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            modelos = [m["name"] for m in response.json().get("models", [])]
            return True, modelos
        return False, []
    except:
        return False, []

def baixar_modelo_ollama(nome_modelo):
    # Sanitização de segurança: permite apenas caracteres padrão de nome de modelo
    if not re.match(r"^[a-zA-Z0-9\.\:\_\-]+$", nome_modelo):
        st.error("Nome de modelo inválido! Use apenas letras, números, pontos, dois-pontos, hífens e sublinhados.")
        return False
    try:
        url = "http://localhost:11434/api/pull"
        # timeout=(5, None): 5 segundos para conectar, sem limite de tempo para baixar os dados do stream
        response = requests.post(url, json={"name": nome_modelo}, stream=True, timeout=(5, None))
        
        if response.status_code != 200:
            st.error(f"Erro na API do Ollama: Código de status {response.status_code}")
            return False
            
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    status = data.get("status", "")
                    completed = data.get("completed", 0)
                    total = data.get("total", 0)
                    
                    if total > 0:
                        percent = min(1.0, max(0.0, completed / total))
                        progress_bar.progress(percent)
                        status_text.text(f"📥 Download: {status} ({percent:.1%})")
                    else:
                        status_text.text(f"⏳ Status: {status}")
                except Exception:
                    pass
                    
        progress_bar.empty()
        status_text.empty()
        return True
    except Exception as e:
        st.error(f"Erro ao conectar com Ollama ou baixar o modelo: {e}")
        return False

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

# 3. Sidebar (Atualizada com Métricas e Configurações)
with st.sidebar:
    st.title("⚙️ Configuração do LLM")
    provedor = st.selectbox("Provedor", options=["OpenAI", "Ollama (Local)"], index=0)
    
    if provedor == "OpenAI":
        modelo_selecionado = st.selectbox("Modelo OpenAI", options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    else:
        ollama_online, modelos_instalados = verificar_ollama()
        if ollama_online:
            st.markdown("🟢 **Ollama:** Conectado")
            if modelos_instalados:
                modelo_selecionado = st.selectbox("Modelo Ollama", options=modelos_instalados)
            else:
                st.warning("Nenhum modelo baixado no Ollama. Baixe usando `ollama run <modelo>`.")
                modelo_selecionado = st.selectbox("Modelo Ollama", options=["llama3", "mistral"])
        else:
            st.markdown("🔴 **Ollama:** Desconectado")
            st.error("Inicie o Ollama no seu computador para usá-lo.")
            modelo_selecionado = st.selectbox("Modelo Ollama", options=["llama3", "mistral"])
        
    st.session_state.provedor = provedor
    st.session_state.modelo = modelo_selecionado
    
    # NOVO: Gerenciador de Downloads Ollama
    if provedor == "Ollama (Local)":
        st.divider()
        st.subheader("📥 Baixar Novo Modelo")
        
        opcoes_base = [
            "llama3.1",
            "llama3.2",
            "deepseek-r1:8b",
            "deepseek-coder",
            "phi3",
            "gemma2:9b",
            "qwen2.5:7b",
            "mistral",
            "mxbai-embed-large",
            "nomic-embed-text",
            "bge-m3"
        ]
        
        opcoes_modelos = ["Personalizado (Digitar nome)..."]
        instalados = modelos_instalados if 'modelos_instalados' in locals() else []
        
        for mod in opcoes_base:
            if mod in instalados or f"{mod}:latest" in instalados:
                opcoes_modelos.append(f"{mod} ✅")
            else:
                opcoes_modelos.append(mod)

        escolha_download = st.selectbox("Selecione um modelo popular:", options=opcoes_modelos)
        
        if escolha_download == "Personalizado (Digitar nome)...":
            novo_modelo = st.text_input("Digite o nome exato do modelo (veja em ollama.com/library):", placeholder="Ex: llama3:70b")
        else:
            novo_modelo = escolha_download.replace(" ✅", "")
            if "✅" in escolha_download:
                st.info(f"O modelo **{novo_modelo}** já está instalado! Você pode baixá-lo novamente para atualizar.")
            else:
                st.info(f"Modelo selecionado: **{novo_modelo}**")

        if st.button("Baixar Modelo"):
            if novo_modelo:
                with st.spinner(f"Baixando {novo_modelo}... Isso pode demorar alguns minutos."):
                    if baixar_modelo_ollama(novo_modelo):
                        st.success(f"Modelo {novo_modelo} baixado com sucesso!")
                        st.rerun()
                    else:
                        st.error("Erro ao baixar. Verifique o nome do modelo ou sua conexão.")
            else:
                st.warning("Digite ou selecione o nome do modelo.")

    st.divider()

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
    prompt_limpo = prompt.strip().lower()
    if prompt_limpo in memoria:
        cached = memoria[prompt_limpo]
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
        # Exibe imediatamente para o usuário
        with st.chat_message("assistant"):
            st.info("⚡ Resposta recuperada da Memória (Custo Zero)")
            st.markdown(output)
    else:
        # --- CONSULTA REAL RAG ---
        with st.chat_message("assistant"):
            with st.spinner("IA Pensando (Consulta Original)..."):
                try:
                    if st.session_state.provedor == "OpenAI":
                        arquivo_db = DB_OPENAI
                        embeddings = OpenAIEmbeddings()
                    else:
                        arquivo_db = DB_OLLAMA
                        embeddings = OllamaEmbeddings(model=MODEL_OLLAMA_EMBED)
                        
                    vectorstore = InMemoryVectorStore.load(arquivo_db, embeddings)
                except FileNotFoundError:
                    st.error(f"⚠️ Banco de dados não encontrado para o provedor {st.session_state.provedor}. Por favor, rode `python criar_db.py` no terminal para gerar o arquivo `{arquivo_db}` antes de continuar.")
                    st.stop()
                except Exception as e:
                    erro_str = str(e)
                    if "404" in erro_str or "not found" in erro_str or "pull" in erro_str:
                        st.error(f"⚠️ **Modelo do Ollama Não Encontrado:** O modelo de embeddings `{MODEL_OLLAMA_EMBED}` não está instalado no seu Ollama local.")
                        st.info("💡 **Como resolver:** Vá na barra lateral esquerda em **📥 Baixar Novo Modelo**, selecione o modelo e clique em **Baixar Modelo**, ou execute no terminal:\n"
                                f"`ollama pull {MODEL_OLLAMA_EMBED}`")
                    elif "connection" in erro_str.lower() or "connect" in erro_str.lower() or "refused" in erro_str.lower():
                        st.error("⚠️ **Ollama Indisponível:** Não foi possível conectar ao Ollama local. Certifique-se de que o Ollama está em execução em seu computador (geralmente na porta 11434).")
                    else:
                        st.error(f"Erro ao carregar o banco de dados: {e}")
                    st.stop()

                try:
                    resultados = vectorstore.similarity_search(prompt, k=K_RETRIEVAL)
                    contexto = "\n\n".join([doc.page_content for doc in resultados])
                    
                    # Instancia o modelo selecionado na sidebar
                    if st.session_state.provedor == "OpenAI":
                        llm = ChatOpenAI(model=st.session_state.modelo, temperature=0)
                    else:
                        llm = ChatOllama(
                            model=st.session_state.modelo,
                            temperature=0,
                            num_ctx=OLLAMA_CONTEXT_WINDOW,
                            num_predict=2048,  # era 1024 — respostas mais completas
                            num_gpu=99,        # força todos os layers na GPU
                            num_thread=8,      # threads CPU para pré-processamento
                        )

                    # PROMPT COM HISTÓRICO (CHAT MEMORY)
                    cp = ChatPromptTemplate.from_messages([
                        ("system", SYSTEM_PROMPT),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{pergunta}")
                    ])
                    chain = cp | llm
                    
                    # Converte histórico para o formato LangChain
                    history = []
                    for msg in st.session_state.messages[:-1]: # Exclui a pergunta atual que já está no prompt
                        if msg["role"] == "user":
                            history.append(HumanMessage(content=msg["content"]))
                        else:
                            history.append(AIMessage(content=msg["content"]))

                    if st.session_state.provedor == "OpenAI":
                        with get_openai_callback() as cb:
                            res = chain.invoke({"contexto": contexto, "chat_history": history, "pergunta": prompt})
                            output = res.content
                            
                            # Atualiza métricas no session_state
                            st.session_state.metrics = {
                                "total": cb.total_tokens,
                                "prompt": cb.prompt_tokens,
                                "completion": cb.completion_tokens,
                                "cost": cb.total_cost
                            }
                            # Exibe a resposta final (OpenAI costuma ser rápida ou stream opcional)
                            with st.chat_message("assistant"):
                                st.markdown(output)
                    else:
                        # Modelos locais (Ollama) - STREAMING para melhor UX
                        with st.chat_message("assistant"):
                            placeholder = st.empty()
                            full_response = ""
                            # Usamos .stream() para ver a resposta sendo gerada
                            for chunk in chain.stream({"contexto": contexto, "chat_history": history, "pergunta": prompt}):
                                # Em Ollama, o chunk pode vir como string ou objeto com content
                                content = chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
                                full_response += content
                                placeholder.markdown(full_response + "▌")
                            
                            placeholder.markdown(full_response)
                            output = full_response
                        
                        st.session_state.metrics = {
                            "total": 0,
                            "prompt": 0,
                            "completion": 0,
                            "cost": 0.0
                        }
                except Exception as e:
                    erro_str = str(e)
                    if "insufficient_quota" in erro_str or "429" in erro_str or "RateLimitError" in erro_str:
                        st.error("⚠️ **Cota da OpenAI Excedida (RateLimitError):** Você esgotou os créditos da sua chave de API OpenAI. Por isso, a aplicação não consegue realizar a busca ou gerar a resposta com a OpenAI. Para continuar usando sem custos, altere o provedor para **Ollama (Local)** na barra lateral e certifique-se de ter o `banco_de_dados_ollama.json` criado.")
                    elif "404" in erro_str or "not found" in erro_str or "pull" in erro_str:
                        st.error(f"⚠️ **Modelo do Ollama Não Encontrado:** O modelo de embeddings `{MODEL_OLLAMA_EMBED}` ou o modelo de linguagem `{st.session_state.modelo}` não está instalado no seu Ollama local.")
                        st.info("💡 **Como resolver:**\n\n"
                                "1. Use a barra lateral em **📥 Baixar Novo Modelo**, digite ou selecione o modelo e clique em **Baixar**.\n\n"
                                f"2. Ou execute no terminal:\n`ollama pull {MODEL_OLLAMA_EMBED}` para os embeddings e `ollama pull {st.session_state.modelo}` para o chat.")
                    elif "connection" in erro_str.lower() or "connect" in erro_str.lower() or "refused" in erro_str.lower():
                        st.error("⚠️ **Ollama Indisponível:** Não foi possível conectar ao Ollama local. Certifique-se de que o Ollama está em execução em seu computador (geralmente na porta 11434).")
                    else:
                        st.error(f"❌ Erro durante a consulta: {e}")
                    st.stop()

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

    # Adiciona ao histórico e recarrega para atualizar sidebar
    if 'output' in locals():
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.rerun()
