from __future__ import annotations
import asyncio, os, json, time
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
import numpy as np
import httpx                   # асинхронная альтернатива requests

DB_DIR        = "data/chroma"
COLLECTION    = "pdf_documents"
DENSE_MODEL   = "intfloat/multilingual-e5-base"
RERANK_MODEL  = "BAAI/bge-reranker-base"
OLLAMA_MODEL  = "mistral:7b-instruct-q4_K_M"
N_RETRIEVE    = 15             # из вектора
N_RERANK      = 5              # после rerank попадут в LLM
CTX_MAX_TOK   = 3500           # запас на вопрос и системные префиксы

client        = chromadb.PersistentClient(DB_DIR)
collection    = client.get_collection(COLLECTION)

embedder      = SentenceTransformer(DENSE_MODEL, device="cuda")
reranker      = CrossEncoder(RERANK_MODEL, device="cuda")

# ---------- вспомогательные функции ----------

async def ollama_async(prompt: str, stream: bool = True) -> str:
    url = "http://localhost:11434/api/generate"
    payload = dict(model=OLLAMA_MODEL, prompt=prompt,
                   stream=stream, temperature=0.2, top_p=0.95)
    async with httpx.AsyncClient(timeout=60.0) as ac:
        if stream:
            async with ac.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                chunks = [json.loads(line)["response"] async for line in resp.aiter_lines()]
                return "".join(chunks).strip()
        else:
            r = await ac.post(url, json=payload)
            r.raise_for_status()
            return r.json()["response"].strip()

def dense_retrieve(query: str, k: int) -> list[dict]:
    q_emb = embedder.encode(query, normalize_embeddings=True)
    res = collection.query(q_emb, n_results=k, include=["documents", "metadatas"])
    docs, metas = res["documents"][0], res["metadatas"][0]
    return [{"text": d, **m} for d, m in zip(docs, metas)]

def rerank(query: str, passages: list[dict], k: int) -> list[dict]:
    pairs = [[query, p["text"]] for p in passages]
    scores = reranker.predict(pairs, convert_to_numpy=True)
    for p, s in zip(passages, scores): p["score"] = float(s)
    return sorted(passages, key=lambda x: x["score"], reverse=True)[:k]

def shrink(passages: list[dict], max_tokens: int) -> str:
    out, tokens = [], 0
    for p in passages:
        t_est = len(p["text"].split())
        if tokens + t_est > max_tokens: break
        out.append(p["text"])
        tokens += t_est
    return "\n\n".join(out)

# ---------- событийный цикл ----------

async def handle_query(question: str) -> tuple[str, list[dict], float]:
    start = time.perf_counter()

    candidates = dense_retrieve(question, N_RETRIEVE)
    top_passages = rerank(question, candidates, N_RERANK)

    ctx = shrink(top_passages, CTX_MAX_TOK)

    prompt = (
        "You are an assistant that answers user queries using only the provided documents. For each claim you make, "
        "cite the exact source."
        "If information is not found in the documents, reply with “I don’t know.” Present answers clearly and concisely. "
        "Do not generate any information not supported by the sources. If sources conflict, "
        "mention the discrepancy and your confidence level.\n\n"
        f"### Documents:\n{ctx}\n\n"
        f"### Question:\n{question}\n\n### Your answer:"
    )

    answer = await ollama_async(prompt, stream=True)
    elapsed = time.perf_counter() - start

    return answer, top_passages, elapsed

async def main() -> None:
    print("Enter your question: ")
    while True:
        q = input(">> ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        if not q:
            continue

        answer, top_passages, elapsed = await handle_query(q)

        print("\nSources:")
        for i, p in enumerate(top_passages, 1):
            print(f"{i}. {p['source']} page{p.get('page','?')}")
        print(f"\nAnswer: ({elapsed:0.1f} s):\n{answer}\n" + "-"*50)

if __name__ == "__main__":
    asyncio.run(main())
