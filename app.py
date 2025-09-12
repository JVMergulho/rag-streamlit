import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Importa as funções dos arquivos separados
import store_docs
import langchain_manager

# Carrega as variáveis de ambiente
load_dotenv()

# Título da aplicação
st.title("📚 Chat com seus Documentos")

# ---
# 1. Área para Upload de PDFs
# ---

st.header("Upload de Documentos")
st.write("Faça o upload de seus arquivos PDF aqui para que eles sejam permanentemente adicionados à base de dados.")
uploaded_files = st.file_uploader(
    "Escolha um ou mais arquivos PDF", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Processar e Armazenar PDFs"):
        with st.spinner("Processando e armazenando documentos..."):
            
            # Garante que o diretório de upload exista
            Path(os.environ["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
            
            # Salva os arquivos no diretório temporário
            for uploaded_file in uploaded_files:
                file_path = os.path.join(os.environ["UPLOAD_DIR"], uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Carrega todos os documentos salvos
            documents = store_docs.load_documents(directory=os.environ["UPLOAD_DIR"])

            if documents:
                store_docs.store_in_vector_db(documents, vector_db_path=os.environ["VECTOR_DB_PATH"])
                st.success("🎉 Documentos processados e base de dados atualizada com sucesso!")
            else:
                st.warning("⚠️ Nenhum documento PDF válido foi carregado.")

# 2. Área para Fazer a Pergunta
# ---

st.header("Faça uma Pergunta")
query = st.text_area(
    "Digite sua pergunta com base nos documentos:",
    placeholder="Ex: Qual a importância do Transformer na atenção do modelo?"
)

# Botão para enviar a pergunta
if st.button("Gerar Resposta"):
    if query and os.path.exists(os.environ["VECTOR_DB_PATH"]):
        with st.spinner("Buscando e gerando a resposta..."):
            # Chama a função do langchain_manager para gerar a resposta
            response = langchain_manager.generate_response(query)
            
            # Exibe a resposta na área de resposta
            st.session_state['response'] = response
            st.success("Resposta gerada!")
    elif not os.path.exists(os.environ["VECTOR_DB_PATH"]):
        st.error("❌ A base de dados vetorial não foi encontrada. Por favor, faça o upload e processe os PDFs primeiro.")
    else:
        st.warning("⚠️ Por favor, digite uma pergunta.")

# 3. Área de Resposta
# ---

st.header("Resposta")
# Exibe a resposta se ela existir na sessão
if 'response' in st.session_state:
    st.markdown(st.session_state['response'])
else:
    st.info("Aguardando uma pergunta...")