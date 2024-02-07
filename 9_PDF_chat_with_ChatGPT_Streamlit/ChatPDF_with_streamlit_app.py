#!pip install streamlit
#!pip install altair==4.0

import streamlit as st 
import pickle
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting text into chunks
import os
import threading


os.environ['OPENAI_API_KEY'] = "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"

def main():
    st.header('Chat with your PDF')
    st.sidebar.title('LLM ChatApp using Langchain')
    st.sidebar.markdown('''
    This is a LLM powered chatbot built using :
    - [Streamlit](https://docs.streamlit.io/)  
    - [Langchain](htts://python.langchain.com)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model                
    ''')
    
    # In streamlit, add a link by using [] to hold the text for the link followed by the () which holds the url
    st.sidebar.write('You can write something here to make more information available')

    #Upload PDF file - file path gets loaded to object
    pdf = st.file_uploader("Upload your PDF file", type='pdf')
    #pdf = r"C:\Users\tevin\OneDrive\Learning\GenAI_&_Langchain\GenAI_-_Langchain\8_PDF_chat_with_Langchain_openai\yolov7paper - Copy.pdf"
    if pdf is not None:
        pdf_reader = PdfReader(pdf)   # reads the information in the file
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        #Split text into multiple chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=200,
            length_function=len)
        
        chunks = text_splitter.split_text(text=text)
        
        #Create the embeddings
        # If the user uploads the same file again - to save on embeddings costs we can store the embeddings
        store_name = pdf.name[:-4]
        st.write(store_name)
        
        change_path = "C:\\Users\\tevin\\OneDrive\\Learning\\GenAI_&_Langchain\\GenAI_-_Langchain\\9_PDF_chat_with_ChatGPT_Streamlit"
        os.chdir(change_path)
        st.write(os.getcwd())
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pickle", 'rb') as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings loaded from the Disk")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pickle", 'wb') as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings Created')
            
        # Create an input for the user query
        query = st.text_input("Ask Question from your PDF file")
        #query = 'What is the paper about?'
        if query:
            docs= VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            
            #Check cost for embeddings and extraction
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            
            st.write(response)
    else:
        st.write("Upload your PDF to start chatting with it!")

if __name__ == '__main__':
    main()