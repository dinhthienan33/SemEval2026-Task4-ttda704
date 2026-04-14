import json
from typing import List, Tuple, Generator, Dict, Any

import torch
from torch.utils.data import Dataset

def read_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    """Read JSONL file line by line"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_synthetic_contrastive(path: str) -> List[Tuple[str, str, str]]:
    """Load contrastive learning triples"""
    triples = []
    for obj in read_jsonl(path):
        a = obj.get("anchor_story") or obj.get("anchor_text")
        p = obj.get("similar_story") or obj.get("text_a")
        n = obj.get("dissimilar_story") or obj.get("text_b")
        if a and p and n:
            triples.append((a, p, n))
    return triples


def load_synthetic_classification(path: str) -> List[Tuple[str, str, str]]:
    """Load classification triples (converts to contrastive format)"""
    triples = []
    for obj in read_jsonl(path):
        a = obj.get("anchor_text") or obj.get("anchor_story")
        A = obj.get("text_a") or obj.get("similar_story")
        B = obj.get("text_b") or obj.get("dissimilar_story")
        lbl = obj.get("text_a_is_closer")
        if a and A and B and lbl is not None:
            pos, neg = (A, B) if bool(lbl) else (B, A)
            triples.append((a, pos, neg))
    return triples


def load_dev_data(path: str) -> List[Tuple[str, str, str, bool]]:
    """
    Load dev data for evaluation.
    Returns: List of (anchor, text_a, text_b, label) where label is True if A is closer.
    """
    triples = []
    for obj in read_jsonl(path):
        a = obj.get("anchor_story") or obj.get("anchor_text")
        p = obj.get("similar_story") or obj.get("positive_text")
        n = obj.get("dissimilar_story") or obj.get("negative_text") or obj.get("text_b")

        if not p or not n:
            A = obj.get("text_a")
            B = obj.get("text_b")
            lbl = obj.get("text_a_is_closer")
            if A and B and lbl is not None:
                triples.append((a, A, B, bool(lbl)))
            elif a and p and n:
                triples.append((a, p, n, True))

    return triples


def load_synthetic_narrative(path: str) -> List[Tuple[str, str, str]]:
    """Load narrative triples from JSONL file"""
    triples = []
    for obj in read_jsonl(path):
        a = obj.get("anchor_story") or obj.get("anchor_text")
        p = obj.get("similar_story") or obj.get("positive_text")
        n = obj.get("dissimilar_story") or obj.get("negative_text")
        if a and p and n:
            triples.append((a, p, n))
    return triples

class TripletDataset(Dataset):
    """Dataset for triplet training"""

    def __init__(self, triples, tokenizer, max_len=512):
        self.triples = triples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        item = self.triples[idx]
        if len(item) == 4:
            a, p, n, _ = item
        else:
            a, p, n = item

        def tok_and_squeeze(text):
            enc = self.tok(text, truncation=True, padding='max_length',
                           max_length=self.max_len, return_tensors='pt')
            return {k: v.squeeze(0) for k, v in enc.items()}

        return dict(
            anchor=tok_and_squeeze(a),
            pos=tok_and_squeeze(p),
            neg=tok_and_squeeze(n)
        )
