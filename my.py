from langchain.storage import InMemoryStore
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

from langchain.prompts import PromptTemplate

# Loaders
loaders = [
    TextLoader("mydata/aboutwissam.txt"),
    TextLoader("mydata/albahar.txt"),
    TextLoader("mydata/bachelorthesis.txt"),
    TextLoader("mydata/gitec.txt"),
    TextLoader("mydata/languages.txt"),
    TextLoader("mydata/masterthesis.txt"),
]
# template = """
# You are  Wissam's assistant,you should help answer the user queries about wissam,dont genrate content by yourself  if question is not related to wissam data.
# I am a language model trained on the text of Wissam's CV. I can help answer questions about Wissam's CV. 
# You can ask me anything you want to know about Wissam's CV.

# Question: {question}
# Context: {context}


# """
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template=template)


template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know.
Don't try to make up an answer.
{context}


Question: {question}
Answer: 
"""

prompt = PromptTemplate.from_template(template)
# Embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Vectorstore, Storage, and Retriever
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
vectorstore = Chroma(collection_name="wissam_cv", embedding_function=embeddings)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, docstore=store, child_splitter=child_splitter
)
docs = []
for loader in loaders:
    docs.extend(loader.load())
retriever.add_documents(docs, ids=None)

# HuggingFaceHub
API_KEY = "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"
hub_llm = HuggingFaceHub(
    huggingfacehub_api_token=API_KEY,
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={"temperature": 0.01},
)
# Initialize the QA chain
qa_chain = RetrievalQA.from_chain_type(
    hub_llm,
    retriever=retriever,
    return_source_documents=False,
    memory=ConversationBufferMemory(memory_key="chat_history"),
    chain_type_kwargs={"prompt": prompt},
    # chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    chain_type="stuff"
)

#response = qa_chain.run({"query": user_input, "chat_history": st.session_state.messages})
# 
def rag_func(user_input: str) -> str:
    """
    This function takes in user input or prompt and returns a response.
    :param user_input: String value of the question or prompt from the user.
    :return: String value of the answer to the user's question or prompt.
    """

    # sub_docs = vectorstore.similarity_search(user_input)

    response = qa_chain({"query": user_input, "chat_history": st.session_state.messages})
    
    return response