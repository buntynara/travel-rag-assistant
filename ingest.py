import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv() 

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
TRAVEL_DOC_PATH = os.getenv("TRAVEL_DOC_PATH")

def main():
    # remove existing Chroma DB if it exists
    print(f"Checking if Chroma DB exists at: {CHROMA_DB_PATH}")
    if Path(CHROMA_DB_PATH).exists():
        print(f"Removing db...")
        shutil.rmtree(CHROMA_DB_PATH)


    print("Loading markdown documents...")

    loader = DirectoryLoader(
        TRAVEL_DOC_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents. Creating chunks...")    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")
    
    print(f"Creating embeddings using model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print(f"Creating vector database at: {CHROMA_DB_PATH}")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name="travel_docs"
    )

    # vectordb.persist() # not needed as Chroma.from_documents already persists the database

    print()
    print(f"Vector DB stored at: {CHROMA_DB_PATH}")

    # verify that the database was created and contains the expected number of chunks 
    collection = vectordb._collection
    print(f"Retrieved {collection.count()} chunks")


if __name__ == "__main__":
    main()
