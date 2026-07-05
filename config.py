import os
from dotenv import load_dotenv

# --- Configurações de Processamento de Documentos ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Configurações de Recuperação (RAG) ---
K_RETRIEVAL = 10

# --- Modelos e Provedores ---
MODEL_OPENAI = "gpt-4o-mini"
MODEL_OLLAMA_EMBED = "mxbai-embed-large"
MODEL_OLLAMA_CHAT = "deepseek-r1:8b" # Modelo principal aproveitando a GPU de 16GB
OLLAMA_CONTEXT_WINDOW = 8192 # Janela de contexto ampliada para o RAG local
#MODEL_OLLAMA_CHAT = "phi"
#MODEL_OLLAMA_CHAT = "gemma3:"


# --- Arquivos de Dados ---
SRC_DIR = "src"
DB_OPENAI = "banco_de_dados.json"
DB_OLLAMA = "banco_de_dados_ollama.json"
MEMORIA_CACHE = "memoria_consultas.json"

# --- Prompt Expert ---
SYSTEM_PROMPT = """
Você é um Engenheiro de IA Especialista. Sua missão é responder perguntas técnicas com base EXCLUSIVAMENTE nos documentos fornecidos.

Regras de Ouro:
1. Se a resposta não estiver no contexto, diga honestamente que não encontrou essa informação.
2. Seja técnico, preciso e direto.
3. Cite o nome do arquivo e a página quando possível.
4. Mantenha o tom profissional e prestativo.

Contexto dos Documentos:
{contexto}
"""
