import os
from pathlib import Path

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.document_loaders import AsyncChromiumLoader, PyMuPDFLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).parent

llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(embedding=embeddings_model, index_name="unstructurednew")

retriever = vectorstore.as_retriever()

template = """You are an expert in Fantasy Premiere League (FPL) football.
You can expect to receive questions about strategies for the current gameweek as well as general FPL strategic questions. In both cases, you'll be given contextual information.
Make sure you're answers are grounded in the context provided.
Today's Date is: March 25, 2024

Make sure you supplement your answers with clear terminology. For example, if you reference a "Double Gameweek", make sure you explain what you mean by this.

Answer the question based only on the following context.
If you cannot answer the question with the context, please respond with 'I don't know'.

Context:
{context}

Question:
{input}
"""
prompt = ChatPromptTemplate.from_template(template)

document_chain = create_stuff_documents_chain(llm, prompt)
unstructured_retrieval_chain = create_retrieval_chain(retriever, document_chain)
