import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, tokenizer, dev_triples, device, max_len=512):
    """Evaluate model on dev set — compute accuracy via cosine similarity."""
    model.eval()
    correct = 0
    total = len(dev_triples)

    for anchor, a, b, label in tqdm(dev_triples, desc="Evaluating"):
        def get_emb(t):
            tokens = tokenizer(t, truncation=True, padding='max_length',
                               max_length=max_len, return_tensors='pt').to(device)
            e = model(**tokens)
            return e.squeeze(0)

        va, vA, vB = get_emb(anchor), get_emb(a), get_emb(b)
        simA = torch.dot(va, vA)
        simB = torch.dot(va, vB)

        if (label and simA > simB) or ((not label) and simA < simB):
            correct += 1

    acc = correct / total if total > 0 else 0.0
    model.train()
    return acc
