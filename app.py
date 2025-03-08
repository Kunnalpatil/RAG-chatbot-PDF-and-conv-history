## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import chromadb.api
chromadb.api.client.SharedSystemClient.clear_system_cache()
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
GROQ_API_KEY = os.environ['groq_api_key'] = os.getenv("GROQ_API_KEY")
st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## set up streamlit app
st.title("Conversational RAG with PDF uploads and conversation history")
st.write("Upload the PDF's to chat with it ")

## define model 
model = st.selectbox("Which model do you want to use", 
                     ("gemma2-9b-it", "llama-3.1-8b-instant"))

llm=ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model)

#chat interface 
session_id = st.text_input("Session ID : Enter a new session id to start a new conversation without chat-history", value="Default")

## Statefully manage chat history

if 'store' not in st.session_state:
    st.session_state.store = {}

uploades_files = st.file_uploader("upload a PDF file", type="pdf",accept_multiple_files=True)

## process the files 
if uploades_files:
    documents=[]
    for file in uploades_files:
        temppdf = f"./temp.pdf"
        with open(temppdf,"wb") as f:
            f.write(file.getvalue())
            f_name = file.name

        
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)


    ## split and create embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectore_store = Chroma.from_documents(documents=splits, embedding=st.session_state.embeddings)
    retriver = vectore_store.as_retriever()

    ## prompt to read store hystory and create a new prompt with history 
    contextulize_qa_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed , otherwise return it as is."

    )
    contextulize_qa_prompt= ChatPromptTemplate.from_messages(
        [
            ("system",contextulize_qa_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', "{input}")
        ]
    )  

    history_aware_retriver=create_history_aware_retriever(llm,retriver,contextulize_qa_prompt) 
    
    ## Answer question
    system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use ten sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human',"{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriver, question_answer_chain)

    def get_session_history(session: str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversationnal_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )


    user_input = st.text_input('how can I help you with the uploaded document')
    if user_input:
        session_history = get_session_history(session_id)
        response = conversationnal_rag_chain.invoke(
            {'input':user_input},
            config={
                "configurable":{"session_id":session_id}
            }
        )

        st.write(response['answer'])
        # st.write("These are the points i referred to answer the question")
        # for i in response['context']:
        #     st.write(i)
        # st.write("Chat History:", session_history.messages)
else:
    st.warning("please upload the file")
