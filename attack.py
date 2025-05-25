import asyncio
import random
import string
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import chromadb
import interactive_rag

# Параметры атаки
N_TOKENS = 50  # длина d̃j
BATCH_SIZE = 16  # сколько кандидатов на шаг
MAX_ITERS = 500  # максимум итераций
STOP_AFTER_NO_IMPROVE = 100  # стоп если нет улучшений
BLOCKER_ID = "blocker_doc"  # фиксированный id для блокера

# Целевая фраза отказа (ее будем оптимизировать)
TARGET_REFUSAL = "I don’t know."

# Частотный словарь токенов для замены (пример: a-z, пробел и знаки препинания)
TOKEN_VOCAB = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-?!")

# Для эмбеддинга и оценки сходства целевого отказа
similarity_embedder = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")
target_emb = similarity_embedder.encode(TARGET_REFUSAL, convert_to_numpy=True)

DB_DIR = "data/chroma"
embedder = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="pdf_documents")

def random_token():
    return random.choice(TOKEN_VOCAB)


def embed_text(text: str) -> np.ndarray:
    return similarity_embedder.encode(text, convert_to_numpy=True)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(cos_sim(a, b))


async def insert_blocker_document(doc_text: str):
    emb = embedder.encode(doc_text, convert_to_numpy=True)
    # Метаданные для блокера
    metadata = {"source": "blocker_document", "page": None}
    collection.upsert(ids=[BLOCKER_ID], embeddings=[emb], metadatas=[metadata], documents=[doc_text])


async def test_blocker(query: str, blocker_text: str) -> tuple[str, float]:
    """
    Вставляем блокер, вызываем RAG, получаем ответ и считаем сходство с целевой фразой отказа
    """
    await insert_blocker_document(blocker_text)
    answer, _, _ = await interactive_rag.handle_query(query)
    answer_emb = embed_text(answer)
    sim = similarity(answer_emb, target_emb)
    return answer, sim


async def attack(query: str):
    print(f"Starting attack on query:\n{query}\n")

    # Инициализация блокера: d̃r = query, d̃j = '!' * N_TOKENS
    d_r = query
    d_j = ["!"] * N_TOKENS

    best_d_j = d_j.copy()
    best_sim = -1.0
    no_improve_iters = 0

    # Функция конкатенации
    def build_blocker_text(d_r, d_j_list):
        # Пробелы для читаемости
        return d_r + " " + "".join(d_j_list)

    for iteration in range(MAX_ITERS):
        # Выбираем случайную позицию для изменения
        pos = random.randint(0, N_TOKENS - 1)

        candidates = []
        # Включаем текущий d_j как кандидат
        candidates.append(best_d_j.copy())

        # Генерируем BATCH_SIZE вариантов с заменой токена в pos
        for _ in range(BATCH_SIZE):
            new_d_j = best_d_j.copy()
            new_d_j[pos] = random_token()
            candidates.append(new_d_j)

        # Проверяем кандидатов параллельно (или последовательно для простоты)
        best_candidate = best_d_j
        best_candidate_sim = best_sim

        for candidate in candidates:
            blocker_text = build_blocker_text(d_r, candidate)
            answer, sim = await test_blocker(query, blocker_text)
            print(f"Iter {iteration + 1} Pos {pos} Sim={sim:.4f} Answer snippet: {answer[:60]!r}")

            if sim > best_candidate_sim:
                best_candidate_sim = sim
                best_candidate = candidate

        # Если улучшение — обновляем
        if best_candidate_sim > best_sim:
            best_sim = best_candidate_sim
            best_d_j = best_candidate
            no_improve_iters = 0
            print(f"** Improved similarity to {best_sim:.4f} at iter {iteration + 1}")
        else:
            no_improve_iters += 1

        # Условия остановки
        if best_sim > 0.95:
            print("Target refusal similarity reached.")
            break
        if no_improve_iters >= STOP_AFTER_NO_IMPROVE:
            print(f"No improvement for {STOP_AFTER_NO_IMPROVE} iters, stopping.")
            break

    final_text = build_blocker_text(d_r, best_d_j)
    print(f"\nBest blocker document text:\n{final_text}\nSimilarity: {best_sim:.4f}")
    print("Attack finished.")


# Для запуска
if __name__ == "__main__":
    q = input("Enter the target query to attack:\n>> ")
    asyncio.run(attack(q))
