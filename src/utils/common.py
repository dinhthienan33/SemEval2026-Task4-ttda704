import os
import zipfile
import logging


def ensure_extracted(path):
    """If path is a zip file, extract it next to the zip and return extracted folder path."""
    if path is None:
        return None
    if os.path.isdir(path):
        return path
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        outdir = os.path.splitext(path)[0]
        os.makedirs(outdir, exist_ok=True)
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(outdir)
        return outdir
    return os.path.dirname(path)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def download_and_prepare(dataset_identifier: str):
    """Download dataset using kagglehub and check train/test paths."""
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError("kagglehub not installed. Run: pip install kagglehub")

    print(f"Downloading dataset: {dataset_identifier}")
    path = kagglehub.dataset_download(dataset_identifier)

    train_path = os.path.join(path, "dataset", "train")
    dev_path = os.path.join(path, "dataset", "test")

    if not os.path.exists(train_path):
        train_path = path
    if not os.path.exists(dev_path):
        dev_path = ""

    return train_path, dev_path
