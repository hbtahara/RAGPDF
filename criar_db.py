from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from config import *
from dotenv import load_dotenv
import os
import math
import time
import sys

# Evita erros de caractere Unicode no console Windows ao imprimir emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

def criar_db():
    print("Iniciando processo de criação do banco de dados...\n")
    
    print("Qual provedor de Embeddings você deseja usar?")
    print("1. OpenAI (Requer API Key, gera banco_de_dados.json)")
    print("2. Ollama (Local/Gratuito, gera banco_de_dados_ollama.json)")
    escolha = input("Digite 1 ou 2: ")
    
    if escolha == "1":
        provedor = "OpenAI"
        embeddings = OpenAIEmbeddings()
        arquivo_db = DB_OPENAI
    elif escolha == "2":
        provedor = "Ollama"
        embeddings = OllamaEmbeddings(model=MODEL_OLLAMA_EMBED)
        arquivo_db = DB_OLLAMA
    else:
        print("Opção inválida. Saindo...")
        return

    documentos = carrega_documentos()
    
    if not documentos:
        print("Erro: Nenhum documento encontrado na pasta 'src'!")
        return
        
    chunks = divide_em_chuncks(documentos)
    db = vetoriza_chuncks(chunks, embeddings, provedor)
    
    if db is not None:
        salvar_db(db, arquivo_db)
        print(f"--- BANCO DE DADOS ({provedor}) CRIADO E SALVO COM SUCESSO ---")
    else:
        print("--- ERRO: BANCO DE DADOS NÃO FOI CRIADO ---")

def carrega_documentos():
    loader = PyPDFDirectoryLoader(SRC_DIR, glob="*.pdf")
    documentos = loader.load()
    print(f"Documentos carregados: {len(documentos)}")
    return documentos

def divide_em_chuncks(documentos):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Divisão concluída: {len(chunks)} chunks gerados.")
    return chunks

def vetoriza_chuncks(chunks, embeddings, provedor):
    # Obtém o nome do modelo se disponível
    modelo_nome = getattr(embeddings, 'model', getattr(embeddings, 'model_name', 'padrão'))
    print(f"Vetorizando {len(chunks)} chunks com {provedor} ({modelo_nome}) Embeddings...")
    
    total_chunks = len(chunks)
    if total_chunks == 0:
        print("Nenhum chunk para vetorizar.")
        return None
        
    batch_size = max(1, math.ceil(total_chunks / 100))
    proximo_marco = 1
    processed = 0
    db = None
    
    start_time = time.time()
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        
        try:
            if db is None:
                db = InMemoryVectorStore.from_documents(batch, embeddings)
            else:
                db.add_documents(batch)
        except Exception as e:
            erro_str = str(e)
            if "401" in erro_str or "invalid_api_key" in erro_str:
                print("\n❌ ERRO: Sua API Key da OpenAI é inválida ou expirou!")
                print("Verifique o arquivo .env ou use a opção 2 (Ollama).")
                return None
            elif "404" in erro_str or "not found" in erro_str or "pull" in erro_str:
                print(f"\n❌ ERRO: O modelo do Ollama '{modelo_nome}' não foi encontrado localmente!")
                print("Para corrigir, execute no seu terminal:")
                print(f"   ollama pull {modelo_nome}")
                return None
            elif "connection" in erro_str.lower() or "connect" in erro_str.lower() or "refused" in erro_str.lower():
                print("\n❌ ERRO: Não foi possível conectar ao serviço local do Ollama.")
                print("Certifique-se de que o Ollama está rodando e acessível na porta 11434.")
                return None
            else:
                print(f"\n❌ Erro inesperado na vetorização: {e}")
                return None
            
        processed += len(batch)
        percent_atual = (processed / total_chunks) * 100
        
        while proximo_marco <= percent_atual and proximo_marco <= 100:
            if proximo_marco % 10 == 0:
                print(f"[{proximo_marco}%]", end=" ---> " if proximo_marco < 100 else "\n", flush=True)
            else:
                print(f"{proximo_marco}%", end=" -> ", flush=True)
            proximo_marco += 1

    end_time = time.time()
    print(f"\nVetorização concluída em {int(end_time - start_time)} segundos!")
    return db

def salvar_db(db, arquivo_db):
    print(f"Salvando o banco no arquivo: {arquivo_db}")
    db.dump(arquivo_db)

if __name__ == "__main__":
    criar_db()
