from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import chromadb
import random
import numpy as np
import interactive_rag
from copy import deepcopy

# Параметры атаки
N_TOKENS = 50  # длина d̃j
BATCH_SIZE = 16  # сколько кандидатов на шаг
MAX_ITERS = 500  # максимум итераций
STOP_AFTER_NO_IMPROVE = 50  # стоп если нет улучшений
SIM_DELTA = 1e-6 # На сколько должна поменяться близость к таргету, чтобы мы учли изменение
TARGET_SIM = 0.95 # Насколько близкий ответ считаем достаточно хорошим, чтобы закончить поиск
BLOCKER_ID = "blocker_doc"  # фиксированный id для блокера

# Целевая фраза отказа (ее будем оптимизировать)
TARGET_REFUSAL = "I don’t know"
ALTERNATIVE_REFUSAL = ["I'm sorry", 'no explicit information', 'do not provide']

# Частотный словарь токенов для замены (пример: a-z, пробел и знаки препинания)
TOKEN_VOCAB = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-?")

# Для эмбеддинга и оценки сходства целевого отказа
embedder = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")
target_emb = embedder.encode(TARGET_REFUSAL, convert_to_numpy=True)

# Для работы с RAG
DB_DIR = "data/chroma"
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="pdf_documents")

def random_token():
    return random.choice(TOKEN_VOCAB)

def embed_text(text: str) -> np.ndarray:
    return embedder.encode(text, convert_to_numpy=True)

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(cos_sim(a, b))

def find_most_similar(candidates: list[str], target_embedding: np.ndarray) \
    -> tuple[int, str, float]:
    index = 0
    min_dist = float("inf")
    min_dist_text = 'Standard min dist text.'
    for i in range(len(candidates)):
        candidate = candidates[i]
        emb = embed_text(candidate)
        dist = similarity(target_embedding, emb)
        if dist < min_dist:
            index, min_dist, min_dist_text = i, dist, candidate

    return index, min_dist_text, min_dist

def find_best_result(candidates: list[str], target_embedding: np.ndarray) \
    -> tuple[int, str, float]:
    for i in range(len(candidates)):
        candidate = candidates[i]
        if TARGET_REFUSAL in candidate or any(refusal in candidate for refusal in ALTERNATIVE_REFUSAL):
            return i, candidate, 1

    return find_most_similar(candidates, target_embedding)

def build_blocker_text(d_r, d_j_list):
    return d_r + "\n" + "".join(d_j_list)

def insert_blocker_document(doc_text: str, col_name: str):
    emb = embedder.encode(doc_text, convert_to_numpy=True)
    metadata = {"source": "blocker_document", "page": None}
    col = client.get_or_create_collection(name=col_name)
    col.upsert(ids=[BLOCKER_ID], embeddings=[emb], metadatas=[metadata], documents=[doc_text])

def create_collection_copies(num: int):
    docs = collection.get(include=['documents', 'metadatas', 'embeddings'])

    copies = []
    for i in range(num):
        new_collection = client.get_or_create_collection(name=f'col{i}')

        new_collection.upsert(
            ids=docs['ids'],
            embeddings=docs['embeddings'],
            metadatas=docs['metadatas'],
            documents=docs['documents'],
        )
        copies.append(new_collection)

    return copies

def attack(query: str, target_response: str):
    print(f"Starting attack on query:\n{query}\n")
    target_emb = embed_text(target_response)
    try:
        collections = [client.get_collection(name=f"col{i}") for i in range(BATCH_SIZE)]
    except ValueError:
        collections = create_collection_copies(BATCH_SIZE)

    queries = [query]*BATCH_SIZE

    # Инициализация блокера: d̃r = query, d̃j = '!' * N_TOKENS
    d_r = query # Первая часть документа - сам запрос, чтобы док-т выдавался при поиске
    best_d_j = ["!"] * N_TOKENS
    d_j = [["!"] * N_TOKENS for _ in range(BATCH_SIZE)]

    last_sim, iter_with_no_change = 0, 0
    for i in range(MAX_ITERS):
        # Выбираем позицию и у каждого текста на этой позиции заменяем старый токен на какой-то случайный
        pos = random.randint(0, N_TOKENS - 1)
        tokens = random.sample(TOKEN_VOCAB, BATCH_SIZE)
        for j in range(BATCH_SIZE):
            d_j[j][pos] = tokens[j]
            blocker = build_blocker_text(d_r, d_j[j])
            insert_blocker_document(blocker, 'col'+str(j))

        answers, docs = interactive_rag.handle_multiple_queries(
            queries, collections)
        j, best_ans, best_ans_dist = find_best_result(answers, target_emb)
        print(f'{i} Best answer: {best_ans} with sim {best_ans_dist} and document {build_blocker_text(d_r, d_j[j])}')
        if query not in docs[j]:
            print(f'This query did not use the blocker file!')
        best_d_j = deepcopy(d_j[j])
        d_j = [deepcopy(best_d_j) for _ in range(BATCH_SIZE)]

        if abs(last_sim - best_ans_dist) < SIM_DELTA:
            iter_with_no_change += 1
        else:
            iter_with_no_change = 0

        last_sim = best_ans_dist
        if best_ans_dist >= TARGET_SIM:
            print("Target refusal similarity reached.")
            break

        if iter_with_no_change >= STOP_AFTER_NO_IMPROVE:
            print(f"No improvement for {STOP_AFTER_NO_IMPROVE} iters, stopping.")
            break

    final_text = build_blocker_text(d_r, best_d_j)
    print(f"\nBest blocker document text:\n{final_text}\nSimilarity: {best_ans_dist:.4f}")
    print("Attack finished.")

if __name__ == "__main__":
    query = 'What is an AI?'
    attack(query, TARGET_REFUSAL)
