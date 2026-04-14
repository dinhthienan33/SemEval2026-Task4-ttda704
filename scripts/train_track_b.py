#!/usr/bin/env python3
"""
SemEval Track B - Single Stage Training
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import warnings
warnings.filterwarnings('ignore')

import json
import random
import zipfile
import logging
import gc
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

import transformers
transformers.logging.set_verbosity_error()

try:
    import wandb
except Exception:
    wandb = None


CONFIG = {
    "backbone": "sentence-transformers/all-MiniLM-L6-v2",
    "dropout": 0.1,
    "max_len": 512,
    "num_epochs": 20,
    "batch_size": 8,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "patience": 5,
    "seed": 42,
    "margin": 0.3,
    "max_margin": 1.5,
    "output_dir": "hf_model_trackB_improved",
    "use_wandb": False,
    "wandb_project": "semeval-track-b-improved-v2",
    "wandb_entity": "22520010-uit",
    "wandb_key": os.environ.get("WANDB_API_KEY", ""),
    "path_train": "data/raw/synthetic_cleaned.jsonl",
    "path_dev": "data/raw/dev_track_a.jsonl",
    "path_test": "data/raw/dev_track_b.jsonl",
}


def read_jsonl(path):
    """Read JSONL file line by line"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dev_triples(path: str) -> List[Tuple[str, str, str, bool]]:
    """Load dev set for evaluation"""
    triples = []
    for o in read_jsonl(path):
        if {"anchor_text", "text_a", "text_b", "text_a_is_closer"} <= set(o.keys()):
            triples.append((o["anchor_text"], o["text_a"], o["text_b"], bool(o["text_a_is_closer"])))
    return triples


def load_synthetic_narrative(path: str) -> List[Tuple[str, str, str]]:
    """Load synthetic_narrative_data with anchor_text, positive_text, negative_text"""
    triples = []
    for obj in read_jsonl(path):
        a = obj.get("anchor_text")
        p = obj.get("text_a")
        n = obj.get("text_b")

        if not p or not n:
            A = obj.get("text_a")
            B = obj.get("text_b")
            lbl = obj.get("text_a_is_closer")
            if A and B and lbl is not None:
                p, n = (A, B) if bool(lbl) else (B, A)

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
        a, p, n = self.triples[idx]

        def tok_and_squeeze(text):
            enc = self.tok(text, truncation=True, padding='max_length',
                           max_length=self.max_len, return_tensors='pt')
            return {k: v.squeeze(0) for k, v in enc.items()}

        return dict(
            anchor=tok_and_squeeze(a),
            pos=tok_and_squeeze(p),
            neg=tok_and_squeeze(n)
        )


class EmbedModel(nn.Module):
    """Improved embedding model"""
    def __init__(self, backbone_name="sentence-transformers/all-MiniLM-L6-v2",
                 dropout=0.1, pooling='mean', use_gradient_checkpointing=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            trust_remote_code=True
        )
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling

    def mean_pooling(self, hidden_state, attention_mask):
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass with improved pooling"""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        else:
            hidden = out[0]

        if self.pooling == 'mean' and attention_mask is not None:
            emb = self.mean_pooling(hidden, attention_mask)
        else:
            emb = hidden.mean(dim=1)

        emb = self.dropout(emb)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        os.makedirs(save_directory, exist_ok=True)
        try:
            self.backbone.save_pretrained(save_directory)
            config = {
                'dropout': self.dropout.p,
                'pooling': self.pooling
            }
            with open(os.path.join(save_directory, 'embed_config.json'), 'w') as f:
                json.dump(config, f)
        except Exception as e:
            logging.warning(f"Could not save model properly: {e}")
            torch.save(self.backbone.state_dict(),
                       os.path.join(save_directory, "pytorch_model.bin"))


class AdaptiveTripletLoss(nn.Module):
    """Adaptive Triplet Loss with dynamic margin"""
    def __init__(self, base_margin=0.3, max_margin=1.0):
        super().__init__()
        self.base_margin = base_margin
        self.max_margin = max_margin

    def forward(self, anchor, pos, neg):
        pos_dist = (anchor - pos).pow(2).sum(1)
        neg_dist = (anchor - neg).pow(2).sum(1)

        difficulty = torch.clamp(neg_dist - pos_dist, min=0, max=self.max_margin - self.base_margin)
        adaptive_margin = self.base_margin + difficulty * 0.5

        loss = torch.relu(pos_dist - neg_dist + adaptive_margin)
        return loss.mean()


@torch.no_grad()
def evaluate(model, tokenizer, dev_triples, device, max_len=512):
    """Evaluate model on dev set"""
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


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=32, max_len=512):
    """Encode texts to embeddings with memory optimization"""
    model.eval()
    all_emb = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, truncation=True, padding=True,
                           max_length=max_len, return_tensors='pt').to(device)

        with autocast('cuda', enabled=torch.cuda.is_available()):
            emb = model(**tokens)

        all_emb.append(emb.cpu())

        del tokens, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_emb = torch.cat(all_emb, dim=0)
    all_emb = F.normalize(all_emb, p=2, dim=1)
    return all_emb.cpu().numpy().astype(np.float32)


def train_one_stage(model, train_loader, dev_triples, optimizer, scheduler,
                    criterion, device, num_epochs, stage_name, tokenizer=None,
                    use_amp=True, patience=5, eval_every=1):
    """Train model for one stage with validation-based checkpointing"""
    scaler = GradScaler('cuda') if use_amp else None
    best_dev_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Starting {stage_name}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"{stage_name} Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            a = {k: v.to(device) for k, v in batch["anchor"].items()}
            p = {k: v.to(device) for k, v in batch["pos"].items()}
            n = {k: v.to(device) for k, v in batch["neg"].items()}

            if use_amp:
                with autocast("cuda"):
                    ea = model(**a)
                    ep = model(**p)
                    en = model(**n)
                    loss = criterion(ea, ep, en)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                ea = model(**a)
                ep = model(**p)
                en = model(**n)
                loss = criterion(ea, ep, en)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            del a, p, n, ea, ep, en, loss
            if num_batches % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, num_batches)
        print(f"{stage_name} Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        dev_acc = 0.0
        if (epoch + 1) % eval_every == 0 and dev_triples:
            if tokenizer is None:
                raise ValueError("Tokenizer required for evaluation")
            dev_acc = evaluate(model, tokenizer, dev_triples, device)
            print(f"Dev Accuracy: {dev_acc:.4f} (Best: {best_dev_acc:.4f})")

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"New best model! Dev Acc: {best_dev_acc:.4f}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        if wandb and wandb.run is not None:
            wandb.log({
                f"{stage_name}_epoch": epoch + 1,
                f"{stage_name}_loss": avg_loss,
                f"{stage_name}_dev_acc": dev_acc,
            })

    return best_model_state, best_dev_acc


def main():
    logging.basicConfig(level=logging.INFO)
    set_seed(CONFIG["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {mem_total:.2f}GB total")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    run = None
    if CONFIG["use_wandb"] and wandb is not None:
        wandb.login(key=CONFIG["wandb_key"])
        run = wandb.init(
            project=CONFIG["wandb_project"],
            entity=CONFIG["wandb_entity"],
            name=f"trackB-{CONFIG['backbone'].split('/')[-1]}-single-stage",
            config=CONFIG
        )

    path_train = CONFIG["path_train"]
    path_dev = CONFIG["path_dev"]
    path_test = CONFIG["path_test"]

    dev_triples = []
    if os.path.exists(path_dev):
        dev_triples = load_dev_triples(path_dev)
        print(f"Loaded {len(dev_triples)} dev triples")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["backbone"], trust_remote_code=True)
    model = EmbedModel(
        backbone_name=CONFIG["backbone"],
        dropout=CONFIG["dropout"]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = AdaptiveTripletLoss(base_margin=CONFIG["margin"], max_margin=CONFIG["max_margin"])
    use_amp = device == "cuda"

    if os.path.exists(path_train):
        triples_train = load_synthetic_narrative(path_train)
        print(f"Loaded {len(triples_train)} training triples")

        if len(triples_train) > 0:
            dataset_train = TripletDataset(triples_train, tokenizer, max_len=CONFIG["max_len"])
            loader_train = DataLoader(
                dataset_train,
                batch_size=CONFIG["batch_size"],
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )

            total_steps = len(loader_train) * CONFIG["num_epochs"]
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=CONFIG["warmup_steps"],
                num_training_steps=total_steps
            )

            best_state, best_acc = train_one_stage(
                model, loader_train, dev_triples, optimizer, scheduler,
                criterion, device, CONFIG["num_epochs"], "Narrative",
                tokenizer=tokenizer, use_amp=use_amp, patience=CONFIG["patience"]
            )

            if best_state is not None:
                model.load_state_dict(best_state)
                print(f"Loaded best model (Dev Acc: {best_acc:.4f})")
                del best_state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    if dev_triples:
        final_acc = evaluate(model, tokenizer, dev_triples, device, max_len=CONFIG["max_len"])
        print(f"Final Dev Accuracy: {final_acc:.4f}")

    if os.path.exists(path_test):
        texts = [obj["text"] for obj in read_jsonl(path_test)]
        print(f"Encoding {len(texts)} test stories...")
        embs = encode_texts(model, tokenizer, texts, device, max_len=CONFIG["max_len"])
        np.save("track_b.npy", embs)
        print(f"track_b.npy saved {embs.shape} {embs.dtype}")

        with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write("track_b.npy", arcname="track_b.npy")
        print("submission.zip ready")

    print("TRAINING COMPLETE!")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
