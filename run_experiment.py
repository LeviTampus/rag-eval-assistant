from langsmith import Client, evaluate
from dotenv import load_dotenv

from generate import answer, format_context
from retriever import retrieve
from evaluate import judge_faithfulness

load_dotenv()

DATASET = "rag-eval-assistant"


def target(inputs):
    return answer(inputs["question"])


def source_match(run, example):
    expected = example.outputs.get("expected_source")
    if expected is None:
        return {"key": "source_match", "score": None}
    got = run.outputs.get("sources", [])
    return {"key": "source_match", "score": 1.0 if expected in got else 0.0}


def faithfulness(run, example):
    question = example.inputs["question"]
    docs = retrieve(question)
    context = format_context(docs)
    ans = run.outputs.get("answer", "")
    score = judge_faithfulness(ans, context)
    return {"key": "faithfulness", "score": score}


if __name__ == "__main__":
    results = evaluate(
        target,
        data=DATASET,
        evaluators=[source_match, faithfulness],
        experiment_prefix="rag-eval",
    )
    print(results)
