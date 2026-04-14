#!/usr/bin/env python3
"""
SemEval Track A - Training entry point.
Uses configs/track_a_mpnet.yaml for hyperparameters.

Usage:
    python scripts/train_track_a.py --config configs/track_a_mpnet.yaml
"""

import os
import sys
import argparse
import gc
import logging

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.utils.seed import set_seed
from src.utils.metrics import evaluate
from src.models.backbone import EmbedModel
from src.training.losses import AdaptiveTripletLoss
from src.training.trainer_track_a import train_one_stage
from src.data_processing.dataset import (
    TripletDataset,
    load_synthetic_contrastive,
    load_synthetic_classification,
    load_synthetic_narrative,
    load_dev_data,
)


def main():
    parser = argparse.ArgumentParser(description="Train Track A model")
    parser.add_argument("--config", default="configs/track_a_mpnet.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)
    set_seed(cfg["training"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"], trust_remote_code=True)
    model = EmbedModel(
        backbone_name=cfg["model"]["backbone"],
        dropout=cfg["model"]["dropout"],
        pooling=cfg["model"]["pooling"],
    ).to(device)

    dev_triples = []
    if os.path.exists(cfg["data"]["path_dev"]):
        dev_triples = load_dev_data(cfg["data"]["path_dev"])
        print(f"Loaded {len(dev_triples)} dev triples")

    path_train = cfg["data"]["path_train"]
    if os.path.exists(path_train):
        triples = load_synthetic_narrative(path_train)
        print(f"Loaded {len(triples)} training triples")

        dataset = TripletDataset(triples, tokenizer, max_len=cfg["training"]["max_len"])
        loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=0)

        optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
        criterion = AdaptiveTripletLoss(
            base_margin=cfg["loss"]["base_margin"],
            max_margin=cfg["loss"]["max_margin"],
        )

        total_steps = len(loader) * cfg["training"]["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg["training"]["warmup_steps"],
            num_training_steps=total_steps,
        )

        use_amp = device == "cuda"
        best_state, best_acc = train_one_stage(
            model, loader, dev_triples, optimizer, scheduler,
            criterion, device, cfg["training"]["num_epochs"], "TrackA",
            tokenizer=tokenizer, use_amp=use_amp, patience=cfg["training"]["patience"],
        )

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"Loaded best model (Dev Acc: {best_acc:.4f})")

    output_dir = cfg["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
