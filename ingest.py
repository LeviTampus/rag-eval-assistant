from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def load_documents(docs_dir: str = "data/docs"):
    result = []
    if Path(docs_dir).exists():
        for path in Path(docs_dir).glob("*"):
            if path.suffix in (".txt", ".md"):
                text = path.read_text()
                doc = Document(page_content=text, metadata={"source": path.name})
                result.append(doc)
        if not result:
            print("No documents found")
        return result
    else:
        print(f"Folder not found: {docs_dir}")
        return result


def chunk_documents(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return chunks


def build_index(chunks: list[Document], persist_dir: str = "data/chroma"):
    store = Chroma(
        persist_directory=persist_dir,
        collection_name="anime",
    )
    store.reset_collection()
    store.add_documents(documents=chunks)
    return store





if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    store = build_index(chunks)
    print(f"Indexed {store._collection.count()} chunks -> data/chroma/")