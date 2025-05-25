import arxiv
import pathlib
import random
import urllib.request as r
from tqdm import tqdm

def download_pdfs(n=30, out_dir="data/raw_pdfs"):
    path = pathlib.Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(query='AI',
                          max_results=300,
                          sort_by=arxiv.SortCriterion.Relevance)

    results = list(client.results(search))
    selected = random.sample(results, n)

    for result in tqdm(selected, desc="Downloading PDFs"):
        url = result.pdf_url
        fname = path / f"{result.entry_id.split('/')[-1]}.pdf"
        r.urlretrieve(url, fname)

    print(f"Downloaded {len(selected)} PDFs to {path}")

if __name__ == "__main__":
    download_pdfs()
