# SemEval 2026 Task 4 - Narrative Similarity

Implementation for **Track A** (classification) and **Track B** (embedding) of SemEval 2026 Task 4.

- **Track A**: Single-view contrastive learning with adaptive triplet loss
- **Track B**: Multi-view narrative embedding (Theme, Plot, Outcome) with self-supervised view alignment

## Repository Structure

```
SemEval2026-Task4-ttda704/
│
├── configs/                        # All hyperparameters (YAML)
│   ├── track_a_mpnet.yaml          # Track A config (margin, lr, layer freezing)
│   ├── track_b_multiview.yaml      # Track B config (fusion weights, alignment lambda)
│   └── prompt_templates.yaml       # LLM system prompt for o3-mini (no hardcoding)
│
├── data/                           # Data directory (NOT pushed — use .gitignore)
│   ├── raw/                        # Original SemEval data
│   ├── processed/                  # Post-pseudonymization (Track A)
│   └── llm_extracted/              # Theme, Plot, Outcome JSON from LLM (Track B)
│
├── src/                            # Core source modules
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── llm_extractor.py        # OpenAI Batch API caller for multi-view extraction
│   │   └── dataset.py              # PyTorch Dataset + JSONL loaders
│   ├── models/
│   │   ├── backbone.py             # EmbedModel wrapper (MPNet/MiniLM + mean pooling)
│   │   └── multiview_net.py        # Multi-view projection heads + fusion layer
│   ├── training/
│   │   ├── trainer_track_a.py      # Single-view training loop
│   │   ├── trainer_track_b.py      # Multi-view training loop (alternating losses)
│   │   └── losses.py               # AdaptiveTripletLoss, MultiViewContrastiveLoss, AlignmentLoss
│   └── utils/
│       ├── common.py               # Zip extraction, Kaggle download helpers
│       ├── metrics.py              # Accuracy via cosine similarity
│       └── seed.py                 # Fix random seed for reproducibility
│
├── scripts/                        # Entry points
│   ├── run_preprocess.sh           # Run LLM extraction pipeline
│   ├── train_track_a.py            # Train Track A (reads from configs/)
│   ├── train_track_b.py            # Train Track B (reads from configs/)
│   └── generate_submission.py      # Generate track_b.npy + submission.zip
│
├── docs/                           # Documentation
├── notebooks/                      # EDA and error analysis notebooks
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

```bash
git clone <repository-url>
cd SemEval2026-Task4-ttda704
pip install -r requirements.txt
```

### Data Setup

Download datasets via KaggleHub:

```bash
pip install kagglehub
python -c "from src.utils.common import download_and_prepare; download_and_prepare('dinhthienan33/semeval-2026-task-4-track-b')"
```

Or place data manually in `data/raw/`.

### Train Track A (Single-view)

```bash
python scripts/train_track_a.py --config configs/track_a_mpnet.yaml
```

### Train Track B (Multi-view)

```bash
python scripts/train_track_b.py
```

### Generate Submission

```bash
python scripts/generate_submission.py \
    --model-dir checkpoints/track_a \
    --dev-track-b data/raw/dev_track_b.jsonl \
    --dev-track-a data/raw/dev_track_a.jsonl
```

## Model Architecture

### Track A
- **Backbone**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Pooling**: Mean pooling + L2 normalization
- **Loss**: Adaptive Triplet Loss (dynamic margin based on sample difficulty)

### Track B
- **Backbone**: `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **Views**: Theme, Plot, Outcome (extracted via LLM)
- **Loss**: Multi-view contrastive + self-supervised view alignment
- **Fusion**: Weighted combination (50% full-text, 10% theme, 20% plot, 20% outcome)

## Configuration

All hyperparameters are in `configs/*.yaml`. Key settings:

| Parameter | Track A | Track B |
|-----------|---------|---------|
| Backbone | all-MiniLM-L6-v2 | all-mpnet-base-v2 |
| Learning rate | 2e-5 | 2e-5 |
| Batch size | 8 | 32 |
| Epochs | 20 | 15 |
| Loss margin | 0.3 | temperature=0.07 |

## Environment Variables

Set these before running LLM extraction or W&B logging:

```bash
export OPENAI_API_KEY="your-key-here"
export WANDB_API_KEY="your-key-here"
```

## References

- [Codabench Competition](https://www.codabench.org/competitions/10273)
- [Narrative Similarity Task](https://github.com/narrative-similarity-task/narrative-similarity-task.github.io)
- [SemEval 2026 Task 4 Baselines](https://github.com/narrative-similarity-task/semeval-2026-task-4-baselines)
