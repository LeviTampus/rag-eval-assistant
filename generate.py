from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from retriever import retrieve

load_dotenv()

CANNOT_ANSWER = "I couldn't find that in the documents."


def build_chain():
    prompt = ChatPromptTemplate.from_template(
        """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, say that you don't know.
        Context: {context}
        Question: {question}
        """
    )
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=0
    )
    return prompt | llm | StrOutputParser()


def answer(question):
    docs = retrieve(question)
    if not docs:
        return {"answer": CANNOT_ANSWER, "sources": []}

    context = format_context(docs)
    chain = build_chain()
    sources = []
    for i in docs:
        source = i.metadata.get("source", "unknown")
        sources.append(source)
    result = chain.invoke({"context": context, "question": question})
    return {"answer": result, "sources": sources}


def format_context(docs):

    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[{source}] {doc.page_content}")
    return "\n\n".join(blocks)