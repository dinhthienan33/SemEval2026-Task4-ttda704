#!/bin/bash
# Run narrative element extraction via LLM
# Requires: OPENAI_API_KEY environment variable

set -e

echo "=== SemEval 2026 Task 4 - Preprocessing Pipeline ==="

# Step 1: Extract narrative elements using LLM
echo "[1/1] Extracting narrative elements..."
python -m src.data_processing.llm_extractor

echo "=== Preprocessing complete ==="
