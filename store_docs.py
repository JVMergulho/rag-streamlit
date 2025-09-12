import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_single_document(file_path: str):
    if not Path(file_path).is_file():
        print(f"❌ Erro: O arquivo '{file_path}' não foi encontrado.")
        return []

    try:
        loader = PyMuPDFLoader(file_path)
        doc = loader.load()
        print(f"✅ Carregado com sucesso: {Path(file_path).name}")
        return doc
    except Exception as e:
        print(f"❌ Erro ao carregar o arquivo '{Path(file_path).name}': {e}")
        return []

def load_documents(directory: str = "content") -> list:
    docs = []
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"Buscando arquivos PDF em {directory}...")

    for doc_path in Path(directory).glob("*.pdf"):
        doc = load_single_document(str(doc_path))
        docs.extend(doc)

    print(f"\nTotal de páginas carregadas: {len(docs)}")
    return docs

def split_in_chunks(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"\nDocumentos divididos em {len(chunks)} chunks.")
    return chunks

def store_in_vector_db(documents: list, vector_db_path: str = "faiss_index") -> None:
    print("\nGerando embeddings com OpenAI e criando a base de dados vetorial...")
    
    # divide em chunks
    chunks = split_in_chunks(documents)

    # Cria os embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Cria e salva o índice FAISS a partir dos chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vector_db_path)
    print(f"✅ Base de dados vetorial criada e salva em '{vector_db_path}'.")

    delete_from_directory()

def delete_from_directory(source_dir = os.environ["UPLOAD_DIR"]):
    print(f"Removendo documentos da pasta '{source_dir}'...")
    try:
        shutil.rmtree(source_dir)
        print(f"✅ Pasta '{source_dir}' e todo o seu conteúdo foram removidos com sucesso.")
    except OSError as e:
        print(f"Erro: {e.filename} - {e.strerror}.")

if __name__ == "__main__":
    documents = load_documents(directory="content")
    if documents:
        chunks = split_in_chunks(documents)
        store_in_vector_db(chunks)
    else:
        print("\nNenhum documento encontrado. Verifique se há PDFs na pasta 'content'.")