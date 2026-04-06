from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback # Import para tokens
from dotenv import load_dotenv
import os

load_dotenv()

ARQUIVO_DB = "banco_de_dados.json" 

def principal(): 
    if not os.path.exists(ARQUIVO_DB):
        print(f"Erro: O arquivo {ARQUIVO_DB} não foi encontrado. Rode o 'criar_db.py'!")
        return

    pergunta_usuario = input("Qual a sua dúvida sobre IA? ")

    print("Carregando base de conhecimento...")
    embeddings = OpenAIEmbeddings()
    vectorstore = InMemoryVectorStore.load(ARQUIVO_DB, embeddings)

    # Busca de similaridade
    resultados = vectorstore.similarity_search(pergunta_usuario, k=25)
    contexto = "\n\n".join([doc.page_content for doc in resultados])

    template = """
    Responda a pergunta com base nestas informações:
    {contexto}
    
    Pergunta: {pergunta}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = prompt | llm
    
    print("\nGerando resposta...\n")

    # USANDO O CALLBACK PARA CONTAR TOKENS
    with get_openai_callback() as cb:
        resposta = chain.invoke({
            "contexto": contexto,
            "pergunta": pergunta_usuario
        })

    # Exibe a resposta final
    print("--- RESPOSTA ---")
    print(resposta.content)
    
    print("\n--- MÉTRICAS DE USO (TOKENS) ---")
    print(f"Total de Tokens: {cb.total_tokens}")
    print(f"Tokens de Prompt: {cb.prompt_tokens}")
    print(f"Tokens de Resposta: {cb.completion_tokens}")
    print(f"Custo Total (USD): ${cb.total_cost:.6f}")

    print("\n--- FONTES CONSULTADAS ---")
    fontes_unicas = set()
    for doc in resultados:
        nome_arquivo = os.path.basename(doc.metadata.get('source', 'Desconhecido'))
        pagina = doc.metadata.get('page', 'N/A')
        fontes_unicas.add(f"📚 PDF: {nome_arquivo} | Página: {pagina}")
    
    for fonte in sorted(list(fontes_unicas)):
        print(fonte)
    print("----------------")

if __name__ == "__main__":
    principal()
