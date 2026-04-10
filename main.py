import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader,PyPDFium2Loader,PyPDFLoader
from dotenv import load_dotenv
import tempfile
import time
import warnings
warnings.filterwarnings("ignore")
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile")

prompt=ChatPromptTemplate.from_template("""
Answer the question based on the context provided.
Please provide the answer in a clear and concise manner.
<context>
{context}
</context>
"Question": {input}
""")

st.title("DocuMind AI - RAG-based Document Question Answering System")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

def create_vector_embadding(uploaded_files):
    if "vector" not in st.session_state and uploaded_files:
        st.session_state.embeddings=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
        docs = []
        for uploaded_file in uploaded_files:
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            loaded = False

            #Try PyPDFLoader (fast)
            loader = PyPDFLoader(tmp_path)
            data = loader.load()
            if data:
                docs.extend(data)
                loaded = True

            #Try PyPDFium2Loader
            if not loaded:
                loader = PyPDFium2Loader(tmp_path)
                data = loader.load()
                if data:
                    docs.extend(data)
                    loaded = True

            #Try Unstructured (last fallback)
            if not loaded:
                loader = UnstructuredPDFLoader(tmp_path)
                data = loader.load()
                if data:
                    docs.extend(data)
                    loaded = True

            if not loaded:
                st.error(f"Could not extract text from: {uploaded_file.name}")

        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(docs)
        st.session_state.vector=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


if st.button("Process Documents"):
    if uploaded_files:
        create_vector_embadding(uploaded_files)
        st.success("Documents processed successfully!")
    else:
        st.warning("Please upload at least one PDF")

user_prompt=st.text_input("Ask a question about your uploaded documents")


if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vector.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retriever_chain.invoke({"input":user_prompt})
    print(f"Response time: {time.process_time()-start}")
    st.write(response['answer'])
    #With streamlit expander
    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------")
    