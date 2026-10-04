import json
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from generate import answer, format_context
from retriever import retrieve, hybrid_retrieve, rerank_retrieve




JUDGE_PROMPT = ChatPromptTemplate.from_template("""
You are grading whether an ANSWER is faithful to the CONTEXT.
Faithful means every claim in the answer is supported by the context.
Reply with ONLY a number from 0 to 1 (1 = fully supported, 0 = not supported).

Context:
{context}

Answer:
{answer}
""")


def load_eval_set(path="data/eval_set.jsonl"):
    data = []
    with open(path, "r") as files:
        for line in files:
            if line.strip() == "":
                continue
            data.append(json.loads(line))
    return data


def score_case(question, expected_source, k=4, retrieve_fn=retrieve):
    docs = retrieve_fn(question, k=k)

    for position, doc in enumerate(docs[:k], start=1):
        if doc.metadata["source"] == expected_source:
            return (True, position)

    return (False, None)


def evaluate(k=4, retrieve_fn=retrieve):
    cases = load_eval_set()
    hits = 0
    mrr_total = 0.0
    n = 0

    for case in cases:
        expected = case["expected_source"]
        if expected is None:
            continue
        n += 1
        hit, rank = score_case(case["question"], expected, k=k, retrieve_fn=retrieve_fn)
        if hit:
            hits += 1; mrr_total += 1 / rank
    recall = hits / n; mrr = mrr_total / n
    return {"recall@k": recall, "mrr": mrr, "cases": n}


def judge_faithfulness(answer, context):
    llm = ChatOpenAI(model=os.getenv("LLM_MODEL"),
                     base_url=os.getenv("LLM_BASE_URL"),
                     api_key=os.getenv("LLM_API_KEY"),
                     temperature=0)
    chain = JUDGE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"answer": answer, "context": context})
    return float(raw)


def evaluate_answers(k=4):
    cases = load_eval_set()
    total = 0.0
    n = 0

    for case in cases:
        if case["expected_source"] is None:
            continue
        docs = retrieve(case["question"], k=k)
        context = format_context(docs)
        result = answer(case["question"])
        score = judge_faithfulness(result["answer"], context)
        total += score; n += 1
    #    time.sleep(15)

    return {"faithfulness": total / n, "cases": n}


def report():
    print("=== Retrieval ===")
    for name, fn in [
        ("naive", retrieve),
        ("hybrid", hybrid_retrieve),
        ("rerank", rerank_retrieve),
    ]:
        r = evaluate(retrieve_fn=fn)
        print(f"{name:8} recall@k={r['recall@k']:.3f}  mrr={r['mrr']:.3f}")
    print("=== Answers ===")
    a = evaluate_answers()
    print(f"faithfulness={a['faithfulness']:.3f}  cases={a['cases']}")


if __name__ == "__main__":
    report()