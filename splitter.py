from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter(documents, chunk_size=1000, chunk_overlap=200):
    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)

    # Return the chunks
    print(f"\nSplit into {len(chunks)} chunks")
    return chunks
