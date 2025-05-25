import pathlib
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import tqdm
import hashlib

# Путь к папке с PDF
PDF_DIR = pathlib.Path("data/raw_pdfs")
# Путь для хранения базы
DB_DIR = "data/chroma"

# Загружаем эмбеддер на GPU (если доступен)
embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cuda")

# Создаём клиент Chroma с persistent-хранилищем
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="pdf_documents")

# Текстовый сплиттер на чанки ~512 токенов с перекрытием 50
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

# Обрабатываем каждый PDF
pdf_files = list(PDF_DIR.glob("*.pdf"))
print(f"Найдено PDF файлов: {len(pdf_files)}")

for pdf_path in tqdm.tqdm(pdf_files):
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load_and_split(text_splitter)
    for doc in documents:
        # Генерируем эмбеддинг
        emb = embedder.encode(doc.page_content, convert_to_numpy=True)
        # Уникальный id для чанка (md5 хеш контента)
        uid = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        # Метаданные
        metadata = {"source": str(pdf_path), "page": doc.metadata.get("page", None)}
        # Добавляем в коллекцию
        collection.upsert(ids=[uid], embeddings=[emb], metadatas=[metadata], documents=[doc.page_content])

print(f"Всего чанков в базе: {collection.count()}")
