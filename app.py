from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os

headers = {
    "authorization": st.secrets["auth_token"],
   "content-type": "application/json"
}

#using 1 pdf 
def main():
    #loading the api key from environment variable
    load_dotenv()
    st.set_page_config(page_title="chatPdf", page_icon="🧊")
    st.header("Ask your PDF 💬")
    
    #st.write("Secret Key", st.secrets["auth_token"]) -- to print the api key 
    #st.write(
    #   "Has environment variables been set:",
    #    os.environ["auth_token"] == st.secrets["auth_token"],)
    
    #Two ways to retrieve your api key 
    #key = st.secrets["auth_token"] -- this uses streamlit secrets 
    apikey = os.getenv("OPENAI_API_KEY")

    # upload file
    st.subheader("Upload a document")
    pdf = st.file_uploader("")
    if pdf is not None:
        st.write(pdf)
        
    if pdf is not None:
        st.subheader("Chat...")
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

        # create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=apikey)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(openai_api_key=apikey)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()
