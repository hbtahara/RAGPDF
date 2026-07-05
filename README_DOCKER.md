# Guia de Inicialização com Docker - RAG PDF Expert

Este guia explica como executar o projeto RAG PDF Expert utilizando Docker e Docker Compose de forma simples e segura.

---

## 🛠️ Pré-requisitos
Certifique-se de ter instalado em sua máquina:
1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) ou Docker Engine (Linux).
2. Docker Compose (normalmente incluído no Docker Desktop).
3. **Ollama** instalado na máquina física (caso utilize o provedor local).

---

## 🔒 Configuração de Segurança Importante
Para evitar que suas chaves de API sejam expostas na imagem do Docker:
1. Certifique-se de que o arquivo `.env` **não** seja adicionado ao repositório público do Git. Ele já está configurado no seu `.dockerignore` para nunca ser incorporado à imagem Docker.
2. Para fornecer a chave de API da OpenAI de forma segura ao container, você pode defini-la temporariamente no terminal ou manter o arquivo `.env` local na mesma pasta de onde você executará o `docker-compose`.

---

## 🚀 Passo a Passo para Execução

### Passo 1: Garantir os Arquivos de Persistência
Como o Docker Compose monta volumes pontuais para persistir seus bancos de dados e histórico de memória, o Docker espera que esses arquivos existam. Se eles não existirem, o Docker criará pastas vazias no lugar deles, o que causará erros.

Execute o comando correspondente abaixo na pasta raiz do projeto para criar arquivos vazios caso eles ainda não existam:

**No Windows (PowerShell):**
```powershell
New-Item -ItemType File -Name "banco_de_dados.json", "banco_de_dados_ollama.json", "memoria_consultas.json" -Force
```

**No Linux/macOS:**
```bash
touch banco_de_dados.json banco_de_dados_ollama.json memoria_consultas.json
```

### Passo 2: Executar com Docker Compose
Com os arquivos de dados criados e a sua chave configurada no arquivo `.env` (ou no terminal), execute o comando para compilar e iniciar o container:

```bash
docker-compose up --build
```

O comando irá:
1. Construir a imagem Python com as dependências do projeto.
2. Expor a porta `8501`.
3. Montar a pasta de PDFs `src/` e os arquivos JSON de dados para persistência local.
4. Mapear o DNS `host.docker.internal` para que a aplicação no container acesse o serviço do Ollama rodando no seu computador Host.

### Passo 3: Acessar a Interface
Após a inicialização do container, abra seu navegador de preferência e acesse:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 💡 Comandos Úteis do Docker

* **Parar o container:**
  ```bash
  docker-compose down
  ```

* **Visualizar logs do container em tempo real:**
  ```bash
  docker-compose logs -f
  ```

* **Recriar banco de dados (dentro do container):**
  Se você precisar rodar o script `criar_db.py` para re-processar os PDFs usando as dependências do Docker, você pode rodar:
  ```bash
  docker exec -it ragpdf-app python criar_db.py
  ```
