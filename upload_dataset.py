from langsmith import Client
from evaluate import load_eval_set

def upload(name="rag-eval-assistant"):
    client = Client()
    cases = load_eval_set()
    dataset = client.create_dataset(name)
    inputs = [{"question": c["question"]} for c in cases]
    outputs = [{"expected_source": c["expected_source"],"expected_answer": c["expected_answer"]} for c in cases]
    client.create_examples(dataset_id=dataset.id, inputs=inputs, outputs=outputs)
    print(f"Dataset: {dataset.name}, Count: {len(outputs)}")


if __name__ == "__main__":
    upload()