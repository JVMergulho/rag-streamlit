# 📚 Document RAG Assistant

Uma aplicação de **Gen AI** end-to-end que permite aos usuários conversar com seus próprios documentos PDF. Utiliza a arquitetura **RAG (Retrieval-Augmented Generation)** para garantir respostas precisas, contextuais e baseadas exclusivamente nos dados fornecidos, reduzindo alucinações.

---

## Funcionalidades

- **Ingestão de Documentos:** Upload e processamento de múltiplos arquivos PDF.
- **Processamento de Linguagem Natural:** Quebra de texto (Chunking) inteligente utilizando `RecursiveCharacterTextSplitter`.
- **Busca Semântica:** Indexação e recuperação de informações utilizando **Vector Database (FAISS)** e **OpenAI Embeddings**.
- **Memória Persistente:** A base vetorial é salva localmente, permitindo consultas futuras sem necessidade de reprocessamento.
- **Interface Interativa:** UI desenvolvida em **Streamlit** com feedback em tempo real.

## Arquitetura Técnica

O projeto segue um pipeline moderno de RAG:

1.  **Load:** Carregamento de PDFs via `PyMuPDFLoader`.
2.  **Split:** Divisão do texto em chunks gerenciáveis para otimizar o contexto do LLM.
3.  **Embed:** Conversão textual para vetores numéricos de alta dimensão usando `OpenAIEmbeddings`.
4.  **Store:** Armazenamento em banco vetorial `FAISS` (Facebook AI Similarity Search).
5.  **Retrieve & Generate:** Ao receber uma pergunta, o sistema busca os chunks mais similares semanticamente e os envia como contexto para o LLM gerar a resposta final.
