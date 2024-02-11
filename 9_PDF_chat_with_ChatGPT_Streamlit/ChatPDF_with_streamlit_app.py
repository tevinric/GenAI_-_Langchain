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
import dill

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
        
        load_path = os.getcwd()+'\\'+str(store_name)
        st.write(load_path)
        
        if os.path.exists(load_path):
            VectorStore = FAISS.load_local(load_path,OpenAIEmbeddings())
            st.write('Loaded Locally')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)
            save_path = str(os.getcwd()+'\\'+str(store_name))
            st.write(save_path)
            VectorStore.save_local(save_path)
            st.write('Saved_locally')
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