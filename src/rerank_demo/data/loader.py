import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def _find_dataset_dir(dataset: str, out_dir: str, base_path: str = None):
    candidates = []
    if base_path:
        candidates += [base_path, os.path.join(base_path, dataset), os.path.join(base_path, dataset, dataset)]
    candidates += [os.path.join(out_dir, dataset), os.path.join(out_dir, dataset, dataset)]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "corpus.jsonl")):
            return c
    return None

def download_beir_dataset(dataset: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    # use existing if present
    existing = _find_dataset_dir(dataset, out_dir)
    if existing:
        return existing

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    base = util.download_and_unzip(url, out_dir)

    found = _find_dataset_dir(dataset, out_dir, base)
    if not found:
        raise FileNotFoundError(f"Could not locate corpus.jsonl after unzip. base_path={base}, out_dir={out_dir}")
    return found

def load_beir(dataset: str, out_dir: str, split: str = "test"):
    dataset_dir = download_beir_dataset(dataset, out_dir)
    print(f"[loader] Using dataset dir: {dataset_dir}")
    return GenericDataLoader(dataset_dir).load(split=split)
