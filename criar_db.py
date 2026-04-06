from langchain_community.document_loaders import PyPDFDirectoryLoader
#Carrega os documentos pdf da pasta src
from langchain_text_splitters import RecursiveCharacterTextSplitter
#Divide os documentos em pedacos (chuncks)
from langchain_core.vectorstores import InMemoryVectorStore
#cria o banco de dados em memoria
from langchain_openai import OpenAIEmbeddings
#cria os embeddings
from dotenv import load_dotenv
#carrega as variaveis de ambiente
import os
# Carrega as chaves do arquivo .env
load_dotenv()
# Carrega as chaves do arquivo .env
src = "src"
#pasta onde estão os documentos pdf


ARQUIVO_DB = "banco_de_dados.json" # Nome do arquivo que será salvo
def criar_db():
   print("Iniciando processo de criação do banco de dados...")#imprime uma mensagem de inicio
   documentos = carrega_documentos()#carrega os documentos
   
   if not documentos:#se não houver documentos
       print("Erro: Nenhum documento encontrado na pasta 'src'!")#imprime uma mensagem de erro
       return
   chunks = divide_em_chuncks(documentos)#divide os documentos em pedacos (chuncks)
   db = vetoriza_chuncks(chunks)#vetoriza os pedacos de documentos
   
   # SALVANDO NO DISCO
   salvar_db(db)#salva o banco de dados
   
   print("--- BANCO DE DADOS CRIADO E SALVO COM SUCESSO ---")#imprime uma mensagem de sucesso

   return db #retorna o banco de dados

def carrega_documentos():
    loader = PyPDFDirectoryLoader(src, glob="*.pdf")
    documentos = loader.load()
    print(f"Documentos carregados: {len(documentos)}")
    return documentos

def divide_em_chuncks(documentos):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, #tamanho do pedaco de texto
        chunk_overlap=500, #sobreposicao de pedacos de texto
        length_function=len, #funcao para calcular o tamanho do pedaco de texto
        add_start_index=True #adiciona o indice inicial do pedaco de texto
    )
    chunks = text_splitter.split_documents(documentos)#divide os documentos em pedacos (chuncks)
    print(f"Divisão concluída: {len(chunks)} chunks gerados.") #imprime uma mensagem de conclusao
    return chunks #retorna os pedacos de textos (Chuncks)

def vetoriza_chuncks(chunks): #funcao que vetoriza os pedacos de documentos
    print("Vetorizando chunks com OpenAI Embeddings...")#imprime uma mensagem de inicio
    embeddings = OpenAIEmbeddings()#cria os embeddings
    db = InMemoryVectorStore.from_documents(chunks, embeddings)#cria o banco de dados em memoria
    print("Vetorização concluída!")#imprime uma mensagem de conclusao
    return db #retorna o banco de dados

def salvar_db(db): #funcao que salva o banco de dados
    print(f"Salvando o banco no arquivo: {ARQUIVO_DB}")#imprime uma mensagem de inicio
    # O InMemoryVectorStore permite salvar em um arquivo JSON de forma simples
    db.dump(ARQUIVO_DB)#salva o banco de dados em um arquivo JSON

if __name__ == "__main__": #funcao principal que chama as outras funcoes
    criar_db() #executa a funcao criar_db
