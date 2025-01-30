import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from PyPDF2 import PdfReader  
import tempfile  
import os  

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
load_dotenv()


if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = None

if "questions" not in st.session_state:
    st.session_state["questions"] = []

if "answers" not in st.session_state:
    st.session_state["answers"] = []

def vector_db_from_file(pdf_file):
   
    # read the file
    print("Archivo")
    print("#"*100)
    print(pdf_file)
    print("#"*100)
    #path_pdf_file = pdf_file.name
    
    loader = PyPDFLoader(pdf_file)
    documents = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    print(chunks)
    db = FAISS.from_documents(chunks, OpenAIEmbeddings())

    return db

with st.sidebar:
    st.title("RAG WITH STREAMLIT AND GROQ")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
   
    load_buttom = st.button(label="let's go",type="primary")
    clear_buttom = st.button(label="clear chat", type="secondary")

    if load_buttom:
        if pdf_file is not None:  
            # Guardar el archivo en un directorio temporal  
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:  
                tmp_file.write(pdf_file.getvalue())  
                tmp_file_path = tmp_file.name  
            
            vector_db = vector_db_from_file(tmp_file_path)
            if vector_db:
                print("Vdb creado!")
                st.session_state["vector_db"] = vector_db
                st.session_state["answers"].append("Hi, how can I help you?")
                # Eliminar el archivo temporal  
                os.unlink(tmp_file_path)  

    if clear_buttom:
        st.session_state["questions"] = []
        st.session_state["answers"] = []
        st.session_state["vector_db"] = None

chat_container = st.container()



input_container = st.container()

with input_container:
    with st.form(key="my_form",clear_on_submit=True):
        query = st.text_area("write a prompt!", key="input", height=80)
        submit_button = st.form_submit_button(label="Submit")

        if query and submit_button:
            print(query)

            vector_db = st.session_state["vector_db"]

            ## prompt 
            prompt = """
                        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                        Question: {question} 
                        Context: {context} 
                        Answer:
            """
            prompt_template = PromptTemplate(
                template=prompt,
                #input_variables=["question","context"]
            )

            llm_openai = ChatOpenAI(model="gpt-4")

            retriever_db = vector_db.as_retriever()

            retriever_qa = RetrievalQA.from_chain_type(
                llm=llm_openai,
                retriever=retriever_db,
                chain_type="stuff"
            )

            answer = retriever_qa.run(query)

            ## save in st memory
            st.session_state["questions"].append(query)
            st.session_state["answers"].append(answer)

with chat_container:
    st.title("Talk with your pdf!")

    question_messages = st.session_state["questions"]
    answer_messages = st.session_state["answers"]

    for i in range(len(answer_messages)):
        message(answer_messages[i], key=str(i)+"_bot")
        if i<len(question_messages):
            message(question_messages[i], key=str(i)+"_user",is_user=True)
    