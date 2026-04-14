"""
Training loop for Track B: Multi-view contrastive + self-supervised view alignment.
Extracted from 23thang1/train_multiview_alignment.py
"""

import os
import json
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.training.losses import MultiViewContrastiveLoss, SelfSupervisedViewAlignmentLoss


@torch.no_grad()
def evaluate_multiview(model, dev_triples, narratives=None, device=None):
    """
    Evaluate multi-view model on dev triples using ALL views (theme, plot, outcome, fusion).
    If narratives provided, use extracted theme/plot/outcome from LLM; otherwise use text prefixes.
    """
    model.model.eval()

    all_anchors = [t[0] for t in dev_triples]
    all_a = [t[1] for t in dev_triples]
    all_b = [t[2] for t in dev_triples]

    narrative_lookup = {}
    if narratives:
        for story_id, narrative in narratives.items():
            full_text = narrative.get("full_text", "")
            if full_text:
                narrative_lookup[full_text] = narrative

    anchor_embeddings = model.encode(all_anchors, normalize=True)
    text_a_embeddings = model.encode(all_a, normalize=True)
    text_b_embeddings = model.encode(all_b, normalize=True)

    theme_anchors = []
    for anchor_text in all_anchors:
        narrative = narrative_lookup.get(anchor_text)
        if narrative and narrative.get("theme"):
            theme_anchors.append(f"[THEME] {narrative['theme']}")
        else:
            theme_anchors.append(f"[THEME] {anchor_text}")
    theme_a = [f"[THEME] {text}" for text in all_a]
    theme_b = [f"[THEME] {text}" for text in all_b]

    anchor_theme_emb = model.encode(theme_anchors, normalize=True)
    text_a_theme_emb = model.encode(theme_a, normalize=True)
    text_b_theme_emb = model.encode(theme_b, normalize=True)

    plot_anchors = []
    for anchor_text in all_anchors:
        narrative = narrative_lookup.get(anchor_text)
        if narrative and narrative.get("plot"):
            plot_anchors.append(f"[PLOT] {narrative['plot']}")
        else:
            plot_anchors.append(f"[PLOT] Events: {anchor_text}")
    plot_a = [f"[PLOT] Events: {text}" for text in all_a]
    plot_b = [f"[PLOT] Events: {text}" for text in all_b]

    anchor_plot_emb = model.encode(plot_anchors, normalize=True)
    text_a_plot_emb = model.encode(plot_a, normalize=True)
    text_b_plot_emb = model.encode(plot_b, normalize=True)

    outcome_anchors = []
    for anchor_text in all_anchors:
        narrative = narrative_lookup.get(anchor_text)
        if narrative and narrative.get("outcome"):
            outcome_anchors.append(f"[OUTCOME] {narrative['outcome']}")
        else:
            outcome_anchors.append(f"[OUTCOME] Result: {anchor_text}")
    outcome_a = [f"[OUTCOME] Result: {text}" for text in all_a]
    outcome_b = [f"[OUTCOME] Result: {text}" for text in all_b]

    anchor_outcome_emb = model.encode(outcome_anchors, normalize=True)
    text_a_outcome_emb = model.encode(outcome_a, normalize=True)
    text_b_outcome_emb = model.encode(outcome_b, normalize=True)

    results = {
        "full_text": {"correct": 0},
        "theme": {"correct": 0},
        "plot": {"correct": 0},
        "outcome": {"correct": 0},
        "fusion_equal": {"correct": 0},
        "fusion_weighted": {"correct": 0},
    }

    for i, (_, _, _, label) in enumerate(tqdm(dev_triples, desc="Evaluating")):
        sim_a_full = torch.dot(anchor_embeddings[i], text_a_embeddings[i]).item()
        sim_b_full = torch.dot(anchor_embeddings[i], text_b_embeddings[i]).item()
        score_full = sim_a_full - sim_b_full
        if (label and score_full > 0) or (not label and score_full < 0):
            results["full_text"]["correct"] += 1

        sim_a_theme = torch.dot(anchor_theme_emb[i], text_a_theme_emb[i]).item()
        sim_b_theme = torch.dot(anchor_theme_emb[i], text_b_theme_emb[i]).item()
        score_theme = sim_a_theme - sim_b_theme
        if (label and score_theme > 0) or (not label and score_theme < 0):
            results["theme"]["correct"] += 1

        sim_a_plot = torch.dot(anchor_plot_emb[i], text_a_plot_emb[i]).item()
        sim_b_plot = torch.dot(anchor_plot_emb[i], text_b_plot_emb[i]).item()
        score_plot = sim_a_plot - sim_b_plot
        if (label and score_plot > 0) or (not label and score_plot < 0):
            results["plot"]["correct"] += 1

        sim_a_outcome = torch.dot(anchor_outcome_emb[i], text_a_outcome_emb[i]).item()
        sim_b_outcome = torch.dot(anchor_outcome_emb[i], text_b_outcome_emb[i]).item()
        score_outcome = sim_a_outcome - sim_b_outcome
        if (label and score_outcome > 0) or (not label and score_outcome < 0):
            results["outcome"]["correct"] += 1

        fusion_score_equal = 0.25 * (score_full + score_theme + score_plot + score_outcome)
        if (label and fusion_score_equal > 0) or (not label and fusion_score_equal < 0):
            results["fusion_equal"]["correct"] += 1

        fusion_score_weighted = (
            0.5 * score_full +
            0.1 * score_theme +
            0.2 * score_plot +
            0.2 * score_outcome
        )
        if (label and fusion_score_weighted > 0) or (not label and fusion_score_weighted < 0):
            results["fusion_weighted"]["correct"] += 1

    n_samples = len(dev_triples)
    for view_name in results:
        acc = results[view_name]["correct"] / n_samples if n_samples > 0 else 0.0
        results[view_name]["accuracy"] = acc
        print(f"  {view_name:20s}: {acc:.4f} ({results[view_name]['correct']}/{n_samples})")

    best_acc = results["fusion_weighted"]["accuracy"]
    return best_acc


def train_multiview(model, train_theme, train_plot, train_outcome,
                    dev_triples, narratives, args, device):
    """
    Train multi-view encoder with contrastive + alignment loss.
    Returns best_accuracy achieved on dev set.
    """
    optimizer = torch.optim.AdamW(
        list(model.model.parameters()) +
        list(model.projection_heads.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-5
    )

    contrastive_loss_fn = MultiViewContrastiveLoss(temperature=0.07)
    alignment_loss_fn = SelfSupervisedViewAlignmentLoss(weight=0.5)

    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        model.model.train()
        model.projection_heads.train()
        epoch_loss = 0.0
        num_batches = 0

        sample_size = min(len(train_theme), args.samples_per_epoch)
        indices = random.sample(range(len(train_theme)), sample_size)

        for batch_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[batch_start:batch_start + args.batch_size]

            theme_triplets = [train_theme[i] for i in batch_indices]
            plot_triplets = [train_plot[i] for i in batch_indices]
            outcome_triplets = [train_outcome[i] for i in batch_indices]

            theme_anchor_emb = model.encode([t[0] for t in theme_triplets], normalize=True)
            theme_pos_emb = model.encode([t[1] for t in theme_triplets], normalize=True)
            theme_neg_emb = model.encode([t[2] for t in theme_triplets], normalize=True)

            plot_anchor_emb = model.encode([t[0] for t in plot_triplets], normalize=True)
            plot_pos_emb = model.encode([t[1] for t in plot_triplets], normalize=True)
            plot_neg_emb = model.encode([t[2] for t in plot_triplets], normalize=True)

            outcome_anchor_emb = model.encode([t[0] for t in outcome_triplets], normalize=True)
            outcome_pos_emb = model.encode([t[1] for t in outcome_triplets], normalize=True)
            outcome_neg_emb = model.encode([t[2] for t in outcome_triplets], normalize=True)

            loss_theme = contrastive_loss_fn(theme_anchor_emb, theme_pos_emb, theme_neg_emb)
            loss_plot = contrastive_loss_fn(plot_anchor_emb, plot_pos_emb, plot_neg_emb)
            loss_outcome = contrastive_loss_fn(outcome_anchor_emb, outcome_pos_emb, outcome_neg_emb)

            theme_fusion = (theme_anchor_emb + theme_pos_emb + theme_neg_emb) / 3.0
            plot_fusion = (plot_anchor_emb + plot_pos_emb + plot_neg_emb) / 3.0
            outcome_fusion = (outcome_anchor_emb + outcome_pos_emb + outcome_neg_emb) / 3.0

            loss_align_theme = alignment_loss_fn(theme_anchor_emb, theme_pos_emb, theme_neg_emb, theme_fusion)
            loss_align_plot = alignment_loss_fn(plot_anchor_emb, plot_pos_emb, plot_neg_emb, plot_fusion)
            loss_align_outcome = alignment_loss_fn(outcome_anchor_emb, outcome_pos_emb, outcome_neg_emb, outcome_fusion)

            total_loss = (
                loss_theme + loss_plot + loss_outcome +
                loss_align_theme + loss_align_plot + loss_align_outcome
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.model.parameters()) +
                list(model.projection_heads.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.eval_every == 0:
            dev_acc = evaluate_multiview(model, dev_triples, narratives=narratives, device=device)

            if dev_acc > best_accuracy:
                best_accuracy = dev_acc
                best_epoch = epoch + 1
                print(f"New best accuracy: {best_accuracy:.4f} (Epoch {best_epoch})")

                os.makedirs(os.path.join(args.output_dir, "best_model"), exist_ok=True)
                model.model.save_pretrained(os.path.join(args.output_dir, "best_model", "encoder"))
                model.tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model", "encoder"))
                torch.save(
                    model.projection_heads.state_dict(),
                    os.path.join(args.output_dir, "best_model", "projection_heads.pt")
                )

                metadata = {"epoch": best_epoch, "accuracy": float(best_accuracy), "loss": float(avg_loss)}
                with open(os.path.join(args.output_dir, "best_model", "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

    return best_accuracy
