from langchain_classic.retrievers import MultiQueryRetriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

def build_multiquery_retriever(vectorstore, llm, k=3):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )
    
    print("MultiQueryRetriever Ready")
    return multi_query_retriever

def build_compression_retriever(base_retriever, llm):
    compressor = LLMChainExtractor.from_llm(llm)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    print("ContextualCompressionRetriever Ready")
    return compression_retriever

def build_ensemble_retriever(chunks, vectorstore, k=3):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = int(k)

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6]
    )
    
    print("EnsembleRetriever Ready")
    return ensemble_retriever

def build_full_retriever(chunks, vectorstore, llm, k=3):
    # Layer-1: BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = int(k)

    # Layer-2: MultiQuery Retriever on top of Semantic Retriever
    multi_query_retriever = build_multiquery_retriever(vectorstore, llm, k)

    # Layer-3: Contextual Compression Retriever on top of MultiQuery Retriever
    compression_retriever = build_compression_retriever(multi_query_retriever, llm)

    # Layer-4: Ensemble Retriever combining BM25 and Compression Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, compression_retriever],
        weights=[0.4, 0.6]
    )

    print("Full Retriever Ready")
    return ensemble_retriever
