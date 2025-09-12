from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_vector_db(vector_db_path: str = "faiss_index"):
    print(f"Carregando a base de dados vetorial de '{vector_db_path}'...")
    
    # É essencial usar a mesma classe de embeddings usada para criar o índice
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Carrega o índice FAISS do disco
    vectorstore = FAISS.load_local(
        folder_path=vector_db_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True 
    )
    
    print("✅ Base de dados vetorial carregada com sucesso!")
    return vectorstore

def find_similar_documents(query: str, vectorstore, score_threshold: float = 0.3, k: int = 3):
    print(f"\nBuscando por '{query}'...")
    
    # Realiza a busca por similaridade
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", 
                                     search_kwargs={"score_threshold": score_threshold, "k": k})
    
    results = retriever.invoke(query)

    print("\n--- Resultados Encontrados ---")
    for i, doc in enumerate(results):
        print(f"Documento {i+1}:")
        print(doc.page_content)
        print("-" * 20)

    if results:
        return results
    else:
        return []

if __name__ == "__main__":
    # 1. Carrega a base de dados vetorial
    db = load_vector_db(vector_db_path="faiss_index")
    
    # 2. Define uma pergunta para buscar
    minha_pergunta = "What is attention?"
    
    # 3. Realiza a busca e exibe os resultados
    find_similar_documents(minha_pergunta, db)