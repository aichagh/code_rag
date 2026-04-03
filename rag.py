from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

# streamlit
st.set_page_config(
        page_title="Code Assist",
        page_icon="hammer_and_wrench:"
        )
st.title("Kotlin Code Assistant for Criticove codebase")
st.write("Ask our assistant about th codebase")

# path to load the persistent database
perma_dir = "./perma_dir"

# loading the same embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# loading the db from the path
vector_db = Chroma(
        persist_directory=perma_dir,
        embedding_function=embeddings
        )

# retriving the best 3 chunks
retriever = vector_db.as_retriever(search_kwargs={"k":3})

# retrieving the file tree specifically
file_tree_retriever = vector_db.as_retriever(
        search_kwarg={
            "filter": {"content": "file_tree"},
            "k": 1
            }
        )

# load ollama model
llm = ChatOllama(model="qwen2.5:0.5b", temperature="0.2")

# creating a prompt from a template
prompt_template = """
    You are an expert in Kotlin development. Using excerpt from the following source
    of code, answer the given question. If you do not know how to answer the
    question given the code base, simply say you do not know. Do not invent an
    answer outside of the code base.
    
    Project structure:
    {file_tree}

    Context source code:
    {context}

    User question:
    {question}

    Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# creating the rag chain
def formatting_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
        {"context": retriever | formatting_docs,
        "question": RunnablePassthrough(),
        "file_tree": file_tree_retriever}
        | prompt
        | llm
        | StrOutputParser()
         )

# managing message history
if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# user interaction
if prompt := st.chat_input("Ask about the codebase..."):
    
    # add message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
        })
    
    # user msg
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI answer
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(prompt)
            st.markdown(answer)

    # save answer
    st.session_state.messages.append({
        "role": "AI",
        "content": answer
        })
