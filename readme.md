# Общая информация
В данном репозитории располагается код для повторения атаки, описанной в статье [Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents](https://arxiv.org/abs/2406.05870)

# Требования
Данная инструкция предполагает следующие требования:
- OS: Windows 11
- Python: 3.11
- CUDA Toolkit: 12.6

Данное пособие предполагает наличие видеокарты NVIDIA.

# Установка:
1. Активировать окружение:

``python -m venv rag-env``

``rag-env\Scripts\activate``

2. Установить llama-cpp-python:

- ``set CMAKE_ARGS="-DGGML_CUDA=on"``
- ``pip install llama-cpp-python``

3. Установить Pytorch:

``pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126``

4. Установить оставшиеся зависимости:

``pip install -r ./requirements.txt``

5. Установить langchain: ``pip install -U langchain-community``

5. Скачать модель: ``python download_model.py``

6. Скачать примеры pdf для построения БД: ``python download_pdfs.py``

7. Постройте БД: ``python index_pdfs.py``