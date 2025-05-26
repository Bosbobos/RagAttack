from __future__ import annotations
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.messages import SystemMessage
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv

DB_DIR        = "data/chroma"
COLLECTION    = "pdf_documents"
DENSE_MODEL   = "intfloat/multilingual-e5-base"
OLLAMA_MODEL  = "mistral:7b-instruct-q4_K_M"
N_RERANK      = 5              # сколько файлов берется из БД
CTX_MAX_TOK   = 3500           # запас на вопрос и системные префиксы

load_dotenv()
giga_api = os.getenv('Giga_api')
giga = GigaChat(credentials=giga_api,
                verify_ssl_certs=False)

client        = chromadb.PersistentClient(DB_DIR)
collection    = client.get_collection(COLLECTION)

device = os.getenv('device', 'cpu')
embedder      = SentenceTransformer(DENSE_MODEL, device=device)

# ---------- вспомогательные функции ----------

def dense_retrieve(query: str, k: int, col = collection) -> list[dict]:
    q_emb = embedder.encode(query)
    res = col.query(q_emb, n_results=k, include=["documents", "metadatas"])
    docs, metas = res["documents"][0], res["metadatas"][0]
    return [{"text": d, **m} for d, m in zip(docs, metas)]

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
