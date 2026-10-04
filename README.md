# Anime Lore Assistant — retrieval over 40 anime knowledge bases, with an eval harness

A **retrieval-augmented generation (RAG)** assistant that answers anime-lore
questions with **grounded, cited** answers — and ships with an **evaluation
harness** that measures retrieval and answer quality, then improves both
(naive → hybrid → reranked).

> Everyone builds a RAG. Almost nobody **measures** one. This project measures
> it, compares approaches, and reports the delta — the eval is the product.

---

## 1. Problem

Anime lore is scattered across seasons, arcs, and movies, so a question like
*"what is Haikyuu about?"* or *"how does Attack on Titan end?"* spans multiple
documents. A plain chatbot answers from vague memory; a fixed-pipeline RAG can
**cite the wrong passage** while sounding confident. The goal: answer **only**
from retrieved documents, **show which sources** were used, and **put a number**
on how well the system does.

## 2. Approach

The pipeline is `load → chunk → embed (Chroma, local MiniLM) → retrieve →
grounded prompt → LLM (Groq) → cited answer`. Retrieval is measured across three
configurations — **dense baseline**, **hybrid (BM25 + dense, reciprocal-rank
fusion)**, and **+ cross-encoder rerank** — against a hand-written **golden set**
(13 questions). Answers are scored for **faithfulness** with an **LLM-as-judge**.

- **Retrieval metrics:** `recall@4` and `MRR` (did the expected source come
  back, and how high?)
- **Answer metrics:** faithfulness — is the answer supported by the retrieved
  context?

## 3. Results

Measured on 13 hand-written cases (12 with an expected source; 1 out-of-corpus
control) using local MiniLM embeddings and `qwen/qwen3.8-27b` via Groq:

| Retrieval configuration | recall@4 | MRR |
|---|---|---|
| naive (dense top-k) | 0.917 | 0.806 |
| hybrid (BM25 + dense, RRF) | 0.917 | **0.854** |
| hybrid + cross-encoder rerank | 0.917 | **0.861** |

**Answer faithfulness (LLM-as-judge):** `1.000` across 12 scored answers.

**What the numbers say.** Dense retrieval already finds the right document in
the top-4 most of the time (recall is tied), so the win is in **ranking** — where
the correct passage sits — not recall. Hybrid lifted MRR from `0.806 → 0.854`;
a cross-encoder reranker added a smaller gain to `0.861`.

**A finding worth keeping.** Hybrid *initially* scored **worse** than naive
(MRR 0.778 vs 0.815) on the first eval set — because the eval set was
**semantic-only**, so BM25's keyword matches were just noise. Only after adding
**realistic exact-term cases** (names, codes, aliases) did hybrid's ranking win
appear. *Your test set decides what you can see.*

## 4. How to run

```bash
# 1. Install (Python 3.12, uv)
uv sync          # or: pip install -r requirements.txt

# 2. Configure — copy and fill in your keys
cp .env.example .env    # LLM_BASE_URL, LLM_MODEL, LLM_API_KEY

# 3. Build the index (one-time)
python ingest.py

# 4. Ask
python main.py                      # CLI chat
python app.py                       # Gradio web chat (localhost:7860)

# 5. Evaluate
python evaluate.py                  # recall@4, MRR, faithfulness, comparison table
python upload_dataset.py            # push the eval set to LangSmith
python run_experiment.py            # run the eval in LangSmith with evaluators
```

**Stack:** Python · LangChain (LCEL) · Chroma · MiniLM embeddings · BM25 ·
cross-encoder rerank · Groq (OpenAI-compatible) · LangSmith · Gradio.

## 5. What I'd do next (limitations, honestly)

1. **Stateless chat.** The chat UI ignores conversation history, so vague
   follow-ups ("give me a list") are treated as fresh questions. *Fix: pass prior
   turns into the prompt.*
2. **No relevance threshold.** `retrieve()` always returns top-k, so an
   out-of-corpus question can cite irrelevant documents instead of saying
   *"I don't know."* *Fix: `similarity_score_threshold` / drop low scores.*
3. **Naive retrieval by default in the app.** Hybrid + rerank improved MRR
   locally; wiring the best config into the deployed interface is next.
4. **Small eval set.** 13 cases prove the loop but are easy. *Next: grow to
   ~40 cases and add harder multi-hop questions.*

> **Deployment note:** this ships as a **repo + local CLI/Gradio demo** on
> purpose. Because retrieval still cites irrelevant docs on out-of-corpus
> questions, I chose **not** to expose it publicly yet — hardening first.

---

*Built as a portfolio project for AI Specialist / AI Operations roles: it
exercises retrieval, evaluation, observability, and honest reporting of results.*
