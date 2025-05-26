from __future__ import annotations
import asyncio, os, json, time
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import httpx
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv

DB_DIR        = "data/chroma"
COLLECTION    = "pdf_documents"
DENSE_MODEL   = "intfloat/multilingual-e5-base"
RERANK_MODEL  = "BAAI/bge-reranker-base"
OLLAMA_MODEL  = "mistral:7b-instruct-q4_K_M"
N_RETRIEVE    = 10             # из вектора
N_RERANK      = 5              # после rerank попадут в LLM
CTX_MAX_TOK   = 3500           # запас на вопрос и системные префиксы

client        = chromadb.PersistentClient(DB_DIR)
collection    = client.get_collection(COLLECTION)

embedder      = SentenceTransformer(DENSE_MODEL, device="cuda")
reranker      = CrossEncoder(RERANK_MODEL, device="cuda")

use_gigachat = True
if use_gigachat:
    load_dotenv()
    giga_api = os.getenv('Giga_api')
    giga = GigaChat(credentials=giga_api,
                    verify_ssl_certs=False)

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

def dense_retrieve(query: str, k: int, col = collection) -> list[dict]:
    q_emb = embedder.encode(query)
    res = col.query(q_emb, n_results=k, include=["documents", "metadatas"])
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

def gigachat_query(prompt: str) -> str:
    response = giga.generate(prompt)
    return response.choices[0].message.content

# ---------- событийный цикл ----------

def handle_multiple_queries(queries: list[str], documents: list[collection]) \
        -> list[tuple[str, list[str]]]:
    if len(queries) != len(documents):
        raise ValueError("queries and documents must have same length")
    messages = []
    ctxs = []
    for i in range(len(queries)):
        query = queries[i]
        col = documents[i]
        top_passages = dense_retrieve(query, N_RERANK, col)
        ctx = shrink(top_passages, CTX_MAX_TOK)
        prompt = (
            "Below you have some documents. If an answer to the user's question is in them, answer accordingly, also citing"
            "your sources. If an answer is not present, say that you don't know the answer. Do not describe irrelevant records."
            "DO NOT USE YOUR PRIOR KNOWLEDGE, ONLY THE INFO IN THE DOCUMENTS IS TRUSTWORTHY.\n\n"
            f"### Documents:\n{ctx}\n\n"
            f"### Question:\n{query}\n\n### Your answer:"
        )
        messages.append([SystemMessage(prompt)])
        ctxs.append(ctx)

    responses = giga.generate(messages)
    answers = [response[0].message.content for response in responses.generations]

    return answers, ctxs

async def handle_query(question: str, use_giga: bool) -> tuple[str, list[dict], float]:
    start = time.perf_counter()

    candidates = dense_retrieve(question, N_RETRIEVE)
    top_passages = rerank(question, candidates, N_RERANK)

    ctx = shrink(top_passages, CTX_MAX_TOK)

    prompt = (
        "You are an assistant that answers user queries using only the provided documents. For each claim you make, "
        "cite the exact source."
        "If information is not found in the documents, reply with “I don’t know.” Present answers clearly and concisely. "
        "Do not generate any information not supported by the sources.\n\n"
        f"### Documents:\n{ctx}\n\n"
        f"### Question:\n{question}\n\n### Your answer:"
    )

    if use_giga:
        answer = gigachat_query(prompt)
    else:
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
