# Imagem base oficial leve do Python
FROM python:3.11-slim

# Evita que o Python grave arquivos .pyc no disco
ENV PYTHONDONTWRITEBYTECODE=1

# Garante que a saída do console seja exibida em tempo real
ENV PYTHONUNBUFFERED=1

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema que podem ser necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia apenas o requirements.txt primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Instala as dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia todo o restante do código do projeto
COPY . .

# Explicita que a aplicação roda na porta 8501 (Porta padrão do Streamlit)
EXPOSE 8501

# Healthcheck para garantir que o container está saudável e respondendo
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando para iniciar a aplicação Streamlit
CMD ["streamlit", "run", "interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
