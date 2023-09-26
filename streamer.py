#streamlit run main.py --server.port 5000
import os
import openai
import langchain
import streamlit as st
from typing import Any, Union
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

langchain.verbose = False
os.environ["OPENAI_API_KEY"] = ""
VECTOR_DB = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=os.getcwd()+"/chroma", collection_metadata={"hnsw:space": "cosine"}).as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold":0.5})

question = st.text_input("input your query", value="")
ask_button = st.button("ask")
st.markdown("### Answer")

if question and ask_button:

    system_template = """
You are a travel expert. Answer below question using the context mentioned.
Do not hallucinate, if you don't find answer in the context, just reply, i don't know.

Context: {context}
Question: {question}?

Relevant answer:
"""

    PROMPT = PromptTemplate(
                     input_variables=["context", "question"], 
                     template=system_template
         )

    QA_MODEL = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=500, verbose=False), VECTOR_DB, verbose=False, combine_docs_chain_kwargs={'prompt': PROMPT})

    data = {"question": question, "chat_history": []}
    result = QA_MODEL(data)
    st.markdown(result["answer"])
