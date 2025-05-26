# Общая информация
В данном репозитории располагается код для повторения атаки, описанной в статье [Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents](https://arxiv.org/abs/2406.05870)

Отчет по данной статье и пример атаки можно найти по [ссылке](https://disk.yandex.ru/d/gXKA3J_ra3_tOw)

# Требования
Данная инструкция предполагает следующие требования:
- OS: Windows 11
- Python: 3.11

# Установка:
1. Активировать окружение:

``python -m venv rag-env``

``rag-env\Scripts\activate``

2. Установить зависимости:

``pip install -r ./requirements.txt``

3. Скачать примеры pdf для построения БД: ``python download_pdfs.py``

4. Построить БД: ``python index_pdfs.py``

5. Создать .env файл со следующими переменными:
- Giga_api - api-ключ GigaChat (в будущем будет добавлена поддержка других API-провайдеров)
- device - 'cuda', если вы знаете, что она у вас работает; иначе можно прописать 'cpu' или оставить пустым

6. Запустить файл атаки ``python attack.py``