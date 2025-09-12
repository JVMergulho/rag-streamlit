from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

import retrieve_results

load_dotenv()

def generate_response(query: str):
    vectorstore = retrieve_results.load_vector_db(vector_db_path="faiss_index")
    related_docs = retrieve_results.find_similar_documents(query, vectorstore)

    if len(related_docs) == 0:
        return "Eu não sei."

    # define o prompt com a estrutura RAG (Retrieval-Augmented Generation)
    prompt_rag = ChatPromptTemplate.from_messages([
        ("system",
        "Você é um assistente que responde a perguntas com base no texto fornecido."
        "Responda SOMENTE com base no contexto fornecido."
        "Se não houver contexto suficiente, simplesmente responda 'Eu não sei.'"),

        ("human", "Pergunta: {input}\n\nContexto:\n{context}")
    ])

    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.0
    )

    document_chain = create_stuff_documents_chain(llm, prompt_rag)

    response = document_chain.invoke({"input": query,
                                        "context": related_docs})
    
    return response

if __name__ == "__main__":
    response = generate_response("Why the transformer uses self-attention?")
    print(response)
