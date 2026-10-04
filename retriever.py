from langchain_chroma import Chroma
from ingest import load_documents, chunk_documents
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever

from dotenv import load_dotenv
load_dotenv()


def retrieve(question: str, k: int = 4):
    store = Chroma(collection_name="anime", persist_directory="data/chroma")
    result = store.similarity_search(question, k=k)
    return result


def get_bm25_retriever(k=4):
    docs = load_documents()
    chunks = chunk_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks, k=k)
    return  bm25_retriever


def get_hybrid_retriever(k=4):
    store = Chroma(collection_name="anime", persist_directory="data/chroma")
    dense = store.as_retriever(search_kwargs={"k": k})
    bm25 = get_bm25_retriever(k=k)
    hybrid = EnsembleRetriever(retrievers=[dense, bm25], weights=[0.8, 0.2])
    return hybrid


def hybrid_retrieve(question, k=4):
    result = get_hybrid_retriever(k).invoke(question)
    return result


def get_rerank_retriever(k=4):
    hybrid = get_hybrid_retriever(k=k)
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=k)
    result = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=hybrid)
    return result


def rerank_retrieve(question, k=4):
    call = get_rerank_retriever(k).invoke(question)
    return call






if __name__ == "__main__":
    hybrid = get_hybrid_retriever()
    print("--- hybrid ---")
    for d in hybrid.invoke("What is an anime about volleyball?"):
        print(d.metadata["source"])