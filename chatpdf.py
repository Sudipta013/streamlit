from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
import os

headers = {
    "authorization": st.secrets["auth_token"],
   "content-type": "application/json"
}
open_ai_key = st.secrets["auth_token"]
huggingface_key = st.secrets["huggingface_key"]


#using 1 pdf 
def main():
    load_dotenv()
    st.set_page_config(page_title="chatPdf", page_icon="🧊")

    #CSS
    st.header("AskPDF 💬")
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    bg = """
        <style> [data-testid="stAppViewContainer"]
        {
            background: rgb(113,25,192);
            background: linear-gradient(90deg, rgba(113,25,192,0.9903420840992647) 5%, 
            rgba(41,59,181,1) 50%, rgba(143,0,255,1) 95%);
        }
        </style>
        """
    st.markdown(bg, unsafe_allow_html=True)
    

    # upload file
    st.subheader("Welcome please upload a document 😊")
    pdf = st.file_uploader("")
        
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        with st.subheader("Choose your AI model"):
            embedding_option  = st.radio(
            "Choose Model", ["OpenAI", "HuggingFace"])

            # create embeddings
            if embedding_option == "OpenAI":
                embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key)
            elif embedding_option == "HuggingFace":
                #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
                embeddings = HuggingFaceEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        st.subheader("Chat...")
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            if embedding_option == "OpenAI":
                llm = OpenAI(openai_api_key=open_ai_key)
            elif embedding_option == "HuggingFace":
                llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512},huggingfacehub_api_token=huggingface_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()
