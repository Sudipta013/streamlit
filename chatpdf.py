from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
import os

# Load API keys from Streamlit secrets
open_ai_key = st.secrets["auth_token"]
huggingface_key = st.secrets["huggingface_key"]

def main():
    st.set_page_config(page_title="Chat with Your PDF", page_icon="📄", layout="wide")

    # Custom CSS for improved UI
    st.markdown(
        """
        <style>
            body {
                font-family: 'Nunito', sans-serif;
                background-color: #F8F9FA;
            }
            .main-title {
                text-align: center;
                font-size: 60px;
                font-weight: bold;
                color: #343A40;
                margin-bottom: 10px;
            }
            .description {
                text-align: center;
                font-size: 18px;
                color: #6C757D;
                margin-bottom: 20px;
            }
            .sidebar .sidebar-content {
                background-color: #212529;
                color: white;
            }
            .footer {
                background-color: #FFA500;
                padding: 10px;
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                text-align: center;
                font-weight: bold;
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header and Description
    st.markdown("<div class='main-title'>Chat with Your PDF</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='description'>
        📄 Upload a PDF and ask questions about its content! Whether you need a summary, key insights, or specific details, our AI-powered assistant has you covered.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # File Upload
    st.sidebar.header("Upload Your PDF")
    pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

        # Split text into manageable chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,  # Reduced size for better retrieval
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Embedding Model Selection
        st.sidebar.subheader("Choose Embedding Model")
        embedding_option = st.sidebar.radio("Select Model:", ["OpenAI", "HuggingFace"])
        
        if embedding_option == "OpenAI":
            embeddings = OpenAIEmbeddings(model_name="gpt-3.5-turbo",openai_api_key=open_ai_key)
        else:
            embeddings = HuggingFaceEmbeddings()
        
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Chat Interface
        st.subheader("Chat with Your PDF 🗨️")
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question, k=5)  # Fetching top 5 relevant chunks

            if embedding_option == "OpenAI":
                llm = OpenAI(openai_api_key=open_ai_key)
            else:
                llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512}, huggingfacehub_api_token=huggingface_key,raw_response=True)
            
            chain = load_qa_chain(llm, chain_type="map_reduce")  # Using map_reduce for better handling
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            
            st.write(response)

    # Footer
    st.markdown("<div class='footer'>Powered by The Techie Indians</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
