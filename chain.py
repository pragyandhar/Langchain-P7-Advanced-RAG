from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful ai assistant that answers the questions strictly based on the context below:
     
     RULES:
     - You must answer the question based on the given context.
     - If the question cannot be answered based on the context, say "I don't know".
     - Always mention the source of the information in your answer, using the format [source: <source_name>].
     - Never use any outside knowledge or information that is not provided in the context.
     
     CONTEXT:
    {context}
    """
    ),
    
    ("human", "{input}")
])

def build_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create RAG Chain in two steps:
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return rag_chain
