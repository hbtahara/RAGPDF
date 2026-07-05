# 🤖 Projeto RAG PDF Chat - AI Expert

Este projeto é um sistema de **RAG (Retrieval-Augmented Generation)** de alta performance, otimizado para **Python 3.13**, focado em análise técnica de documentos PDF com rastreamento detalhado de custos e memória inteligente.

---

## 🛠️ Instalação e Configuração

### 1. Clonar e Preparar Ambiente
Certifique-se de estar na pasta do projeto e instale todas as dependências necessárias:

```bash
pip install -r requirements.txt
```

### 2. Configurar API Key
Crie ou edite o arquivo `.env` na raiz do projeto e adicione sua chave da OpenAI:
```env
OPENAI_API_KEY=sua_chave_aqui
```

⚠️ **Nota para Mac/Python 3.13**: O projeto utiliza `InMemoryVectorStore` para persistência local em JSON, garantindo 100% de compatibilidade sem necessidade de drivers complexos de banco de dados.

---

## 📂 Estrutura do Projeto

*   **`src/`**: Pasta onde devem ser colocados os arquivos PDF para análise.
*   **`criar_db.py`**: Script para processar os PDFs, gerar embeddings e salvar o `banco_de_dados.json`.
*   **`interface.py`**: Dashboard profissional em Streamlit com chat, métricas e histórico.
*   **`main.py`**: Versão simplificada para uso via terminal.
*   **`config.py`**: Central de configurações (Chunk size, modelos, caminhos).
*   **`memoria_consultas.json`**: Cache inteligente que economiza tokens em perguntas repetidas.

---

## 🚀 Como Usar

### Passo 1: Processar Documentos
Sempre que adicionar novos PDFs na pasta `src`, atualize o banco de dados:
```bash
python3 criar_db.py
```

### Passo 2: Iniciar a Interface
Abra o aplicativo no seu navegador:
```bash
streamlit run interface.py
```

---

## 🧠 Funcionalidades de Elite

### 🔹 Painel de Métricas (Sidebar)
Acompanhe em tempo real o consumo da última consulta:
*   **Contagem de Tokens**: Total, Prompt e Resposta.
*   **Custo em USD**: Valor exato gasto na chamada da API.
*   **Custo Zero no Cache**: Ao responder a partir da memória local, os contadores são zerados automaticamente.

### 🔹 Memória de Conversa (Multi-turn)
*   O agente agora entende o contexto das mensagens anteriores, permitindo uma conversa natural e fluida com os documentos.

### 🔹 Rastreamento de Fontes
*   Visualize exatamente de qual PDF e de qual página a informação foi extraída. As fontes são exibidas de forma persistente na barra lateral para cada consulta.

---

## 🔧 Configurações Técnicas
*   **Modelo LLM**: `gpt-4o-mini` ou `Ollama`.
*   **Busca de Contexto**: Recupera até **25 trechos relevantes** (k=25).
*   **Text Splitting**: Chunks de 1500 caracteres com 300 de sobreposição (Configurável via `config.py`).

---

## 👨‍💻 Desenvolvimento
Projeto desenvolvido seguindo princípios de **Clean Code**, utilizando o ecossistema **LangChain** para orquestração de IA.
