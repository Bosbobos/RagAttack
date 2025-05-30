import pathlib
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import tqdm
import hashlib
import os

PDF_DIR = pathlib.Path("data/raw_pdfs")
DB_DIR = "data/chroma"

if __name__ == "__main__":
    device = os.getenv('device', 'cpu')
    embedder = SentenceTransformer("intfloat/multilingual-e5-base", device=device)

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name="pdf_documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"Найдено PDF файлов: {len(pdf_files)}")

    for pdf_path in tqdm.tqdm(pdf_files):
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load_and_split(text_splitter)
        for doc in documents:
            emb = embedder.encode(doc.page_content, convert_to_numpy=True)
            uid = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            metadata = {"source": str(pdf_path), "page": doc.metadata.get("page", None)}
            collection.upsert(ids=[uid], embeddings=[emb], metadatas=[metadata], documents=[doc.page_content])

    print(f"Всего чанков в базе: {collection.count()}")
