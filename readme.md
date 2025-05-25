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

2. Установить модель согласно [гайду](https://github.com/abetlen/llama-cpp-python/issues/1963)

3. Установить Pytorch:

``pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126``

4. Установить оставшиеся зависимости:

``pip install -r ./requirements.txt``