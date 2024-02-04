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



if __name__ == '__main__':
    main()