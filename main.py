from loader import load_file
from chain import build_chain
from splitter import splitter
from retriever import (
    build_multiquery_retriever,
    build_compression_retriever,
    build_ensemble_retriever,
    build_full_retriever
)
from vectorstore import build_vectorstore

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv()

RETRIEVER_OPTIONS = {
    "1": "Basic semantic",
    "2": "MultiQuery",
    "3": "Compression",
    "4": "Ensemble (BM25 + semantic)",
    "5": "Full nested (BM25 + compression + multiquery)",
}

def load_multiple_documents(file_paths):
    docs = []
    for path in file_paths:
        document = load_file(path)
        chunks = splitter(document, chunk_overlap=100, chunk_size=500)
        docs.extend(chunks)
        print(f"  '{path}' → {len(chunks)} chunks")
    return docs

def pick_retriever(vectorstore, chunks, llm):
    print("\nChoose retriever strategy:")
    for key, name in RETRIEVER_OPTIONS.items():
        print(f"  {key}. {name}")
    
    choice = input("\nEnter choice (default 1): ").strip() or "1"

    if choice == "1":
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    elif choice == "2":
        return build_multiquery_retriever(vectorstore, llm)
    elif choice == "3":
        base = vectorstore.as_retriever(search_kwargs={"k": 3})
        return build_compression_retriever(base, llm)
    elif choice == "4":
        return build_ensemble_retriever(chunks, vectorstore)
    elif choice == "5":
        return build_full_retriever(chunks, vectorstore, llm)
    else:
        print("Invalid choice, using basic semantic retriever")
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    
def display_sources(context_docs):
    print("\n[Sources used]")
    seen = set()
    for doc in context_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        label = f"{source} (page {page})" if page else source
        if label not in seen:
            print(f"  - {label}")
            seen.add(label)

def main():
    print("\n--- Advanced RAG Pipeline ---\n")

    print("Enter file paths one by one. Press Enter with no input when done.")
    file_paths = []
    while True:
        path = input(f"File {len(file_paths) + 1}: ").strip()
        if not path:
            if not file_paths:
                print("Please enter at least one file.")
                continue
            break
        if not os.path.exists(path):
            print(f"File not found: '{path}' — skipping")
            continue
        file_paths.append(path)

    # Step 1: Load and split documents
    print("\nLoading and splitting documents...")
    chunks = load_multiple_documents(file_paths)
    print(f"Total chunks created: {len(chunks)}")

    # Step 2: Build vectorstore
    print("\nBuilding vectorstore...")
    vectorstore = build_vectorstore(chunks)

    # Step 3: Pick retriever strategy
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = pick_retriever(vectorstore, chunks, llm)

    # Step 4: Build chain
    print("\nBuilding QA chain...")
    chain = build_chain(retriever)

    print("\nReady! Ask questions about your documents.")
    print("Type 'quit' to exit\n")
    while True:
        question = input("You: ").strip()

        if question.lower() == "quit":
            print("Goodbye!")
            break

        if not question:
            continue
        
        # import logging
        # logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        result = chain.invoke({"input": question})

        print(f"\nAssistant: {result['answer']}")
        display_sources(result["context"])
        print()

    pass

if __name__ == "__main__":
    main()
