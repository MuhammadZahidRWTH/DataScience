import os
from dotenv import load_dotenv
load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP multiple runtime error
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from ranker import rerank_documents


##load the API KEYS
#os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

## loading the llm model
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

## creating the prompt template
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    Question:{input}
    """
)
## setting up vectors embedding
def create_vector_embedding():
    if "vectors" not in st.session_state:
        
        st.session_state.vectors=None
        # Step 1: Initiaze OpenAI Embeddings
        st.session_state.embeddings=OpenAIEmbeddings()

        #Step 2 Load PDF Documents from folder
        st.session_state.loader=PyPDFDirectoryLoader("research_paper") ## Data Ingestion Step
        st.session_state.docs=st.session_state.loader.load()

        #Step 3: Split Documents into chunks
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200)

        docs_to_split = st.session_state.docs[:50] if len(st.session_state.docs) >= 50 else st.session_state.docs
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(docs_to_split)

        # Step 4: Create FAISS vector store
        st.session_state.vectors=FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings)



st.title("RAG Document Q&A With Groq and Llama 3")

user_prompt=st.text_input("Enter your query from the research papers")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is Ready")


import time

if user_prompt:
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.error("Please create the vector database first by clicking the 'Document Embedding' button.")
    else:
        # Step 1: Retrieve Documents (Raw from Retriever)
       
        #Create a BM25 retriever
        keyword_retriever=BM25Retriever.from_documents(st.session_state.final_documents)
        #Create a vector retriver
        vector_retriever = st.session_state.vectors.as_retriever(search_kwargs={"k":5})
        #Combine both with weights
        hybrid_retrievers=EnsembleRetriever(
            retrievers=[vector_retriever,keyword_retriever],
            weights=[0.5,0.5]
        )

        retrieved_docs=hybrid_retrievers.get_relevant_documents(user_prompt)

        # Step 2: Rerank the retrieved docs
        reranked_docs=rerank_documents(user_prompt,retrieved_docs,top_n=3)
        #retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Step 3: Run the reranked docs through the document chain
        start = time.process_time()
        
       #context_text = "\n\n".join([doc.page_content for doc in reranked_docs])

        document_chain=create_stuff_documents_chain(llm,prompt)
        response=document_chain.invoke({
            "context":reranked_docs,
            "input":user_prompt
        })

        #response = retrieval_chain.invoke({'input': user_prompt})
        print(f"Response time :{time.process_time() - start}")

        if isinstance(response,dict):
            final_answer=response.get('output_text',str(response))
        else:
            final_answer=str(response)

        st.subheader("Answer")
        st.write(final_answer)

        st.subheader("Comparison: Without vs With Reranking")

        col1,col2=st.columns(2)

        with col1:
            st.write("### Without Reranking")
            for i, doc in enumerate(retrieved_docs[:3]):
                st.write(f"Original Chunk {i+1}:")
                st.write(doc.page_content[:300] + "...")  # truncate for readability
                st.write(f"Original Score: {doc.metadata.get('rerank_score','N/A')}")
                st.write("---")

        with col2:
            st.write("### With Reranking")
            for i, doc in enumerate(reranked_docs):
                 st.write(f"Reranked Chunk {i+1}:")
                 st.write(doc.page_content[:300] + "...")
                 st.write(f"Rerank Score: {doc.metadata.get('rerank_score','N/A')}")
                 st.write("---")
                         












