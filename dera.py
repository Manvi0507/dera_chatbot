import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, TextLoader, UnstructuredHTMLLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder  # For cross-encoder reranking
import torch
from tenacity import retry, stop_after_attempt, wait_fixed
from google.api_core.exceptions import ResourceExhausted
import hashlib
import markdown

import os
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Dera Farm Chatbot")

# Load Google API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_api_key = os.getenv('GROQ_API_KEY')

# Cross-encoder setup for reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', default_activation_function=torch.nn.Sigmoid())  # Example model

def select_loader(file_path):
    if file_path.endswith('.pdf'):
        return PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        return UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith('.pptx'):
        return UnstructuredPowerPointLoader(file_path)
    elif file_path.endswith('.txt'):
        return TextLoader(file_path)
    elif file_path.endswith('.html'):
        return UnstructuredHTMLLoader(file_path)
    elif file_path.endswith('.md'):
        return UnstructuredMarkdownLoader(file_path)  # Markdown can be handled as a text file
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

# Load the data
loader = DirectoryLoader(
    path="web_data", 
    glob="**/*", 
    loader_cls=select_loader  
)
documents = loader.load()

# Split loaded documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_documents = text_splitter.split_documents(documents)

# Upload Chunks as vector embedding into FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.from_documents(split_documents, embeddings)
db.save_local("faiss_index")

def build_chat_history(chat_history_list):
    # This function takes in the Chat History Messages in a List of Tuples format
    # and turns it into a series of Human and AI Message objects
    chat_history = []
    for message in chat_history_list:        
        chat_history.append(HumanMessage(content=message[0]))
        chat_history.append(AIMessage(content=message[1]))
    
    return chat_history

def rerank_with_cross_encoder(query, retrieved_docs):
    """
    Function to rerank retrieved documents using a cross-encoder
    """
    texts = [doc.page_content for doc in retrieved_docs]
    pairs = [[query, text] for text in texts]

    # Cross-encoder scores the relevance between query and each document
    scores = cross_encoder.predict(pairs)

    # Combine the documents with their scores
    docs_with_scores = list(zip(retrieved_docs, scores))
    
    # Sort documents by score in descending order (higher score = more relevant)
    reranked_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
    
    # Return only the reranked documents
    return [doc for doc, score in reranked_docs]


def ask_query(question, chat_history):
    """
    This function performs the following tasks:
    1. Takes in a 'question' and 'chat_history'.
    2. Loads the local FAISS vector database.
    3. Retrieves relevant documents using a history-aware retriever.
    4. Reranks the documents using a cross-encoder.
    5. Uses the reranked documents to answer the query using an LLM.
    """
    chat_history = build_chat_history(chat_history)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    #llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)
    llm = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")  

    condense_question_system_template ="""
    Given a chat history and the latest user question, which might reference context in the chat history,
    formulate a standalone question that can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [("system", condense_question_system_template), ("placeholder", "{chat_history}"), ("human", "{input}")]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, new_db.as_retriever(), condense_question_prompt
    )

    system_prompt ="""
    You are an expert customer support agent at Dera Farms.
    Your task is to answer customer queries related to Dera Farms. Always speak highly of Dera Farms and emphasize its commitment to providing top-quality natural and fresh farm products, including pasture-raised chicken, eggs, and more. Showcase the value of Dera Farms’ offerings and how the customer is making a wise choice by supporting sustainable and farm-to-table practices. Avoid discussing any other companies, websites, resources, books, tools, or products unrelated to Dera Farms.
    Always promote the benefits of Dera Farms' products and services. If you're unsure of an answer, politely inform the customer and guide them to Dera Farms' customer support team through the website. Never guess or provide incorrect information.
    Ensure that your responses are detailed, well-formatted, and provide relevant resources or links within the Dera Farms website, such as product pages, blogs, or contact options. If relevant, include images (such as product images or promotional materials) to enhance the customer’s understanding. Never provide misleading or broken links.
    Follow these principles:
    - Never exaggerate or overpromise. 
    - Ask follow-up questions if necessary to better understand the customer's needs.
    - Provide a clear, concise, and helpful answer based on the context provided.
 
    Use the following pieces of context to answer the customer's query:

    ------
    {context}
    {chat_history}
    
    Follow up question:

    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("placeholder", "{chat_history}"), ("human", "{input}")]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    convo_qa_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Retrieve documents and rerank them
    # Instead of retrieve, use the invoke method for history_aware_retriever
    retrieval_output = history_aware_retriever.invoke({"input": question, "chat_history": chat_history})
    retrieved_docs = retrieval_output  # Extract documents from the output
    # Now rerank the retrieved documents
    reranked_docs = rerank_with_cross_encoder(question, retrieved_docs)
    # Pass the reranked docs to the QA chain
    return convo_qa_chain.invoke({"input": question, "chat_history": chat_history, "context": reranked_docs})

def show_ui():
    """
    1. This function implements the Streamlit UI for the Chatbot.
    2. It provides an interface where users can either ask their own questions or click on preset beginner queries (displayed as buttons).
    3. When the user clicks a button, the corresponding question is submitted, and the response is displayed.
    """
    #st.set_page_config(page_title="Chavera Medical Bot")
    #st.set_page_config(page_title="Chavera Medical Bot")
    st.title("Analytx4t Chatbot")
    st.subheader("Hey there! Let’s explore Analytx4t!")


    # List of beginner queries
    beginner_queries = [
        "What is this website about?",
        "What products does Dera Farms offer?",
        "How are Dera Farms' chickens raised?",
        "How can I place an order for Dera Farms products?"
    ]

    # Display beginner queries as buttons
    cols = st.columns(3)  # Create 3 columns for buttons

    selected_query = None
    for i, query in enumerate(beginner_queries):
        if cols[i % 3].button(query, key=f"query_button_{i}"):  # Add a unique key for each button
           selected_query = query

    # Initialize session state if not set
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []  # Initialize chat_history as an empty list

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []

    # Accept user input or use the selected query
    if selected_query:
        prompt = selected_query
    else:
        prompt = st.chat_input("Ask your query:")

    # Process user input
    if prompt:
        with st.spinner("Working on your query..."):
            chat_history = st.session_state.chat_history
            if not chat_history:
                chat_history = []
            response = ask_query(question=prompt, chat_history=chat_history)  # renamed function

            # Display messages
            with st.chat_message("user"):
                st.markdown(f"**Your Query:** {prompt}")

            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            # Append user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.extend([(prompt, response["answer"])])

if __name__ == "__main__":
    show_ui()












