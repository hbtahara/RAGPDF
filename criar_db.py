from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from config import *
from dotenv import load_dotenv
import os
import glob
import math
import time
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Garante que o event loop do asyncio funcione corretamente no Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Evita erros de caractere Unicode no console Windows ao imprimir emojis
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

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
        embeddings = OllamaEmbeddings(model=MODEL_OLLAMA_EMBED, base_url=OLLAMA_BASE_URL)
        arquivo_db = DB_OLLAMA
    else:
        print("Opção inválida. Saindo...")
        return

    documentos = carrega_documentos()

    if not documentos:
        print("Erro: Nenhum documento encontrado na pasta 'src'!")
        return

    chunks = divide_em_chuncks(documentos)
    chunks = sanitizar_chunks(chunks)
    db = vetoriza_chuncks(chunks, embeddings, provedor)

    if db is not None:
        salvar_db(db, arquivo_db)
        print(f"--- BANCO DE DADOS ({provedor}) CRIADO E SALVO COM SUCESSO ---")
    else:
        print("--- ERRO: BANCO DE DADOS NÃO FOI CRIADO ---")

def carrega_documentos():
    """Carrega PDFs em paralelo usando ThreadPoolExecutor (aproveita múltiplos cores do Xeon)."""
    pdfs = sorted(glob.glob(os.path.join(SRC_DIR, "*.pdf")))

    if not pdfs:
        return []

    # Filtra arquivos macOS ocultos (._arquivo.pdf)
    pdfs = [p for p in pdfs if not os.path.basename(p).startswith("._")]

    max_workers = min(len(pdfs), 4)  # máximo 4 paralelos (I/O bound)
    print(f"Carregando {len(pdfs)} PDFs em paralelo ({max_workers} workers)...")

    def load_single(path):
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            print(f"  ✓ {os.path.basename(path)} — {len(docs)} páginas")
            return docs
        except Exception as e:
            print(f"  ✗ {os.path.basename(path)}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(load_single, pdfs))

    documentos = [doc for docs in results for doc in docs]
    print(f"\nTotal carregado: {len(documentos)} páginas\n")
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

def sanitizar_chunks(chunks):
    """Remove caracteres surrogate inválidos (ex: \\ud835 de PDFs matemáticos)
    que causam 'surrogates not allowed' ao serializar para UTF-8/JSON."""
    total_sanitizados = 0
    for chunk in chunks:
        texto_original = chunk.page_content
        texto_limpo = texto_original.encode('utf-8', errors='ignore').decode('utf-8')
        if texto_limpo != texto_original:
            chunk.page_content = texto_limpo
            total_sanitizados += 1
    if total_sanitizados:
        print(f"⚠️  {total_sanitizados} chunks sanitizados (caracteres inválidos removidos).")
    return chunks

async def _vetoriza_async(chunks, embeddings, provedor):
    """Vetorização assíncrona com lotes paralelos — aproveita OLLAMA_NUM_PARALLEL."""
    modelo_nome = getattr(embeddings, 'model', getattr(embeddings, 'model_name', 'padrão'))
    total_chunks = len(chunks)
    concurrencia = EMBED_CONCURRENCY  # lotes simultâneos (padrão: 4)

    print(f"Vetorizando {total_chunks} chunks com {provedor} ({modelo_nome})")
    print(f"Modo: {concurrencia} lotes paralelos | batch_size: 64 chunks/lote\n")

    if total_chunks == 0:
        print("Nenhum chunk para vetorizar.")
        return None

    # Batches fixos menores (64) evitam OOM e crashes no runner interno do Ollama
    batch_size = 64
    batches = [chunks[i:i + batch_size] for i in range(0, total_chunks, batch_size)]
    total_batches = len(batches)

    db = InMemoryVectorStore(embedding=embeddings)
    semaphore = asyncio.Semaphore(concurrencia)
    processados = 0
    start_time = time.time()
    lock = asyncio.Lock()

    async def processar_batch(idx, batch):
        nonlocal processados
        async with semaphore:
            try:
                await db.aadd_documents(batch)
            except Exception as e:
                erro_str = str(e)
                if "401" in erro_str or "invalid_api_key" in erro_str:
                    raise RuntimeError("API_KEY_INVALIDA")
                elif "404" in erro_str or "not found" in erro_str or "pull" in erro_str:
                    raise RuntimeError(f"MODELO_NAO_ENCONTRADO:{modelo_nome}")
                elif "connection" in erro_str.lower() or "refused" in erro_str.lower():
                    raise RuntimeError("OLLAMA_OFFLINE")
                else:
                    raise

            async with lock:
                processados += len(batch)
                pct = (processados / total_chunks) * 100
                elapsed = int(time.time() - start_time)
                print(
                    f"  [{pct:5.1f}%] lote {idx + 1:3d}/{total_batches} "
                    f"({len(batch)} chunks) | {elapsed}s",
                    flush=True
                )

    try:
        tasks = [processar_batch(i, batch) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks)
    except RuntimeError as e:
        erro_str = str(e)
        if "API_KEY_INVALIDA" in erro_str:
            print("\n❌ ERRO: Sua API Key da OpenAI é inválida ou expirou!")
            print("Verifique o arquivo .env ou use a opção 2 (Ollama).")
        elif "MODELO_NAO_ENCONTRADO" in erro_str:
            nome = erro_str.split(":")[-1]
            print(f"\n❌ ERRO: O modelo '{nome}' não foi encontrado localmente!")
            print(f"   ollama pull {nome}")
        elif "OLLAMA_OFFLINE" in erro_str:
            print("\n❌ ERRO: Não foi possível conectar ao Ollama (porta 11434).")
            print("Certifique-se de que o Ollama está rodando.")
        else:
            print(f"\n❌ Erro inesperado na vetorização: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Erro inesperado na vetorização: {e}")
        return None

    elapsed_total = int(time.time() - start_time)
    print(f"\nVetorização concluída em {elapsed_total} segundos!")
    return db

def vetoriza_chuncks(chunks, embeddings, provedor):
    """Wrapper síncrono que executa a vetorização assíncrona."""
    return asyncio.run(_vetoriza_async(chunks, embeddings, provedor))

def salvar_db(db, arquivo_db):
    print(f"Salvando o banco no arquivo: {arquivo_db}")
    db.dump(arquivo_db)

if __name__ == "__main__":
    criar_db()
