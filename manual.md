# 🤖 Projeto RAG PDF Chat - AI Expert

Este projeto é um sistema de **RAG (Retrieval-Augmented Generation)** de alta performance, otimizado para rodar em **Python 3.13** no Mac, focado em análise técnica de grandes volumes de documentos PDF.

---

## 🛠️ Requisitos de Instalação

Antes de começar, instale as bibliotecas necessárias no seu terminal:

```bash
pip install streamlit langchain-openai langchain-community pypdf python-dotenv
```

⚠️ **Nota para Mac/Python 3.13**: Este projeto utiliza o `InMemoryVectorStore` para evitar incompatibilidades com o `chromadb` em versões recentes do Python, garantindo 100% de estabilidade.

---

## 📂 Estrutura de Arquivos

1.  **`.env`**: Contém a sua `OPENAI_API_KEY`.
2.  **`src/`**: Pasta onde você deve colocar todos os seus arquivos PDF.
3.  **`criar_db.py`**: Motor de processamento inicial dos documentos.
4.  **`interface.py`**: A interface gráfica (GUI) profissional para chat.
5.  **`main.py`**: Versão do chat para uso direto via terminal.
6.  **`banco_de_dados.json`**: Onde os vetores dos seus PDFs ficam salvos.
7.  **`memoria_consultas.json`**: Cache de inteligência do agente (Custo Zero).

---

## 🚀 Como Utilizar (Passo a Passo)

### 1. Preparação (Importante!)
Coloque seus arquivos PDF na pasta `src` e garanta que sua chave da OpenAI esteja no arquivo `.env`.

### 2. Criar a Base de Conhecimento
Sempre que adicionar ou alterar PDFs, rode o comando:
```bash
python3 criar_db.py
```
*O sistema dividirá os textos em pedaços de 1500 caracteres (chunks) com 300 de sobreposição para garantir que nenhum detalhe seja perdido.*

### 3. Iniciar o Chat (Interface Gráfica)
Para a melhor experiência, utilize a interface moderna via navegador:
```bash
streamlit run interface.py
```

---

## 🧠 Funcionalidades Inteligentes Implementadas

### 🔹 Sistema de Memória (Cache JSON)
O agente agora possui um arquivo `memoria_consultas.json`. Se você fizer uma pergunta que já foi feita antes:
*   A resposta é **instantânea**.
*   O custo de tokens é **zero**.
*   A consulta não utiliza a API da OpenAI (Offline).

### 🔹 Painel de Métricas e Fontes
Na barra lateral da interface, você pode acompanhar:
*   **Métricas de Uso**: Contador exato de tokens e custo total em dólar de cada pergunta.
*   **Fontes Consultadas**: Lista exata de quais PDFs e quais páginas foram lidas para gerar a resposta.

### 🔹 Busca Profunda (K=25)
Configuramos a busca para ler até **25 trechos simultâneos** por pergunta. Isso permite que a IA "atropele" índices e sumários para encontrar o conteúdo real dos capítulos lá no meio dos PDFs.

---

## 👨‍💻 Suporte Técnico
Desenvolvido com foco em **Clean Code** e **Modularização**, permitindo fácil manutenção e expansão para outros modelos de IA.
