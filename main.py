from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks.manager import get_openai_callback
from dotenv import load_dotenv
from config import *
import os, json, sys

# Evita erros de caractere Unicode no console Windows ao imprimir emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

def carregar_memoria():
    if not os.path.exists(MEMORIA_CACHE):
        return {}
    with open(MEMORIA_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)

def salvar_na_memoria(pergunta, resposta, fontes):
    memoria = carregar_memoria()
    chave = pergunta.strip().lower()
    memoria[chave] = {
        "resposta": resposta,
        "fontes": list(fontes),
        "total_tokens": 0, # CLI simplificado
        "custo": 0.0
    }
    with open(MEMORIA_CACHE, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=4)

def principal():
    print("--- 🤖 BEM-VINDO AO RAG PDF EXPERT (CLI) ---")
    print("1. OpenAI (Nuvem)")
    print("2. Ollama (Local)")
    escolha = input("Escolha o provedor (1 ou 2): ")

    if escolha == "1":
        provedor = "OpenAI"
        arquivo_db = DB_OPENAI
        embeddings = OpenAIEmbeddings()
    else:
        provedor = "Ollama"
        arquivo_db = DB_OLLAMA
        embeddings = OllamaEmbeddings(model=MODEL_OLLAMA_EMBED)

    if not os.path.exists(arquivo_db):
        print(f"❌ Erro: O arquivo {arquivo_db} não foi encontrado. Rode o 'criar_db.py' primeiro!")
        return

    print(f"✅ Carregando base de conhecimento ({provedor})...")
    try:
        vectorstore = InMemoryVectorStore.load(arquivo_db, embeddings)
    except Exception as e:
        erro_str = str(e)
        if "404" in erro_str or "not found" in erro_str or "pull" in erro_str:
            print(f"\n❌ ERRO: O modelo de embeddings '{MODEL_OLLAMA_EMBED}' não foi encontrado localmente!")
            print("Para corrigir, execute no seu terminal:")
            print(f"   ollama pull {MODEL_OLLAMA_EMBED}")
        elif "connection" in erro_str.lower() or "connect" in erro_str.lower() or "refused" in erro_str.lower():
            print("\n❌ ERRO: Não foi possível conectar ao serviço local do Ollama.")
            print("Certifique-se de que o Ollama está rodando e acessível na porta 11434.")
        else:
            print(f"\n❌ Erro inesperado ao carregar base de dados: {e}")
        return
    
    chat_history = []
    
    print("\nDigite 'sair' para encerrar a conversa.\n")

    while True:
        pergunta_usuario = input("\n👤 Você: ")
        if pergunta_usuario.lower() in ["sair", "exit", "quit"]:
            break

        # --- VERIFICAÇÃO NO CACHE ---
        memoria = carregar_memoria()
        pergunta_limpa = pergunta_usuario.strip().lower()
        
        if pergunta_limpa in memoria:
            cached = memoria[pergunta_limpa]
            print(f"\n🤖 {provedor} (Memória): {cached['resposta']}")
            chat_history.append(HumanMessage(content=pergunta_usuario))
            chat_history.append(AIMessage(content=cached['resposta']))
            continue

        # Busca de similaridade e geração de resposta com tratamento de erro
        try:
            resultados = vectorstore.similarity_search(pergunta_usuario, k=K_RETRIEVAL)
            contexto = "\n\n".join([doc.page_content for doc in resultados])

            # Extrai fontes
            fontes_unicas = set()
            for doc in resultados:
                nome_arquivo = os.path.basename(doc.metadata.get('source', 'Desconhecido'))
                pagina = doc.metadata.get('page', 'N/A')
                fontes_unicas.add(f"📚 {nome_arquivo} (pág {pagina})")

            # Configura o LLM e Chain
            if provedor == "OpenAI":
                llm = ChatOpenAI(model=MODEL_OPENAI, temperature=0)
            else:
                llm = ChatOllama(
                    model=MODEL_OLLAMA_CHAT,
                    temperature=0,
                    num_ctx=OLLAMA_CONTEXT_WINDOW,
                    num_predict=2048,  # era 1024 — respostas mais completas
                    num_gpu=99,        # força todos os layers na GPU
                    num_thread=8,      # threads CPU para pré-processamento
                )

            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT.format(contexto=contexto)),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{pergunta}")
            ])
            
            chain = prompt | llm
            
            print(f"🤖 {provedor} Pensando...")

            if provedor == "OpenAI":
                with get_openai_callback() as cb:
                    resposta = chain.invoke({"chat_history": chat_history, "pergunta": pergunta_usuario})
                    resposta_texto = resposta.content
                    print(f"\n--- RESPOSTA ---\n{resposta_texto}")
                    print(f"\n[Custo: ${cb.total_cost:.5f} | Tokens: {cb.total_tokens}]")
            else:
                resposta = chain.invoke({"chat_history": chat_history, "pergunta": pergunta_usuario})
                resposta_texto = resposta.content
                print(f"\n--- RESPOSTA ---\n{resposta_texto}")
        except Exception as e:
            erro_str = str(e)
            if "insufficient_quota" in erro_str or "429" in erro_str or "RateLimitError" in erro_str:
                print("\n❌ ERRO: Cota da OpenAI excedida ou chave inválida.")
            elif "404" in erro_str or "not found" in erro_str or "pull" in erro_str:
                print(f"\n❌ ERRO: Modelo do Ollama não encontrado localmente!")
                print("Certifique-se de que baixou os modelos necessários rodando no seu terminal:")
                print(f"   ollama pull {MODEL_OLLAMA_EMBED}")
                print(f"   ollama pull {MODEL_OLLAMA_CHAT}")
            elif "connection" in erro_str.lower() or "connect" in erro_str.lower() or "refused" in erro_str.lower():
                print("\n❌ ERRO: Não foi possível conectar ao serviço local do Ollama.")
                print("Certifique-se de que o Ollama está rodando e acessível na porta 11434.")
            else:
                print(f"\n❌ Erro durante a consulta: {e}")
            continue

        # Atualiza histórico e cache
        chat_history.append(HumanMessage(content=pergunta_usuario))
        chat_history.append(AIMessage(content=resposta_texto))
        
        salvar_na_memoria(pergunta_usuario, resposta_texto, fontes_unicas)
        
        print("\nFontes:", ", ".join(list(fontes_unicas)[:3]), "...")

    print("\nAté logo!")

if __name__ == "__main__":
    principal()
