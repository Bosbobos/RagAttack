import torch
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Проверяем, доступна ли CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# Загружаем модель для эмбеддингов
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

# Пример предложений для векторизации
sentences = [
    "This is an example sentence.",
    "Each sentence is converted into a vector.",
    "Let's test if this works on CUDA."
]

# Получаем эмбеддинги для предложений
embeddings = sentence_model.encode(sentences, convert_to_tensor=True)

# Выводим результат эмбеддингов
print("\nEmbeddings for sentences:")
for i, embedding in enumerate(embeddings):
    print(f"Sentence {i+1}: {sentences[i]}")
    print(f"Embedding: {embedding[:5]}...")  # Выводим первые 5 значений эмбеддинга

# Загрузка модели Llama для генерации текста
llama_model = Llama(model_path="models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf", n_ctx=4096, gpu_layers=35)

# Тестируем работу Llama с простым вопросом
prompt = "What is the capital of France?"
response = llama_model(prompt, max_tokens=50)

print("\nLlama response:")
print(response['choices'][0]['text'].strip())
