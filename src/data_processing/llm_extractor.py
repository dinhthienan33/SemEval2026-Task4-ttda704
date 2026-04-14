#!/usr/bin/env python3
"""
Extract Narrative Elements and Compute Similarity Scores
Save results to W&B Artifact
"""

import os
import json
import logging
import time
from typing import List, Tuple, Dict, Optional, Union
from tqdm import tqdm
import numpy as np
import shutil
from itertools import islice

try:
    import wandb
except:
    wandb = None

try:
    import openai
except:
    openai = None

openai_apikey = os.environ.get("OPENAI_API_KEY")
wandb_key = os.environ.get("WANDB_API_KEY")
NARRATIVE_ELEMENTS = {
    "characters": {
        "features": [
            "role", "backstory", "strengths", "weaknesses", 
            "psychology", "beliefs", "motivations", 
            "social_dynamics", "arc"
        ],
        "weight": 0.25
    },
    "plot": {
        "features": [
            "protagonist_intro", "inciting_incident", "rising_action",
            "climax", "resolution", "consequences", "final_outcome",
            "loose_ends", "subplots"
        ],
        "weight": 0.25
    },
    "setting": {
        "features": [
            "time_period", "geographical_location", "cultural_context",
            "social_class", "ideology_and_beliefs", "economic_and_political_context",
            "historical_events", "physical_environment", "technological_level"
        ],
        "weight": 0.25
    },
    "themes": {
        "features": [
            "main_theme", "secondary_themes", "tertiary_themes",
            "resolution_main_theme", "resolution_secondary_themes",
            "resolution_tertiary_themes"
        ],
        "weight": 0.25
    }
}


def read_jsonl(path, num_of_lines=None):
    with open(path, "r", encoding="utf-8") as f:
        lines = islice(f, num_of_lines) if num_of_lines else f
        for line in lines:
            line = line.strip()
            if line:
                yield json.loads(line)
                



def call_openai_api(prompt: str, model="gpt-4o-mini") -> Optional[Dict]:
    """Call OpenAI API to extract narrative elements"""
    if openai is None:
        logging.error("OpenAI not installed")
        return None
    
    try:
        client = openai.OpenAI(api_key=openai_apikey)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return None


def extract_all_narrative_elements(text: str) -> Dict:
    """Extract ALL narrative elements (4 types) in ONE API call
    
    Returns:
        Dict with keys: characters, plot, setting, themes
    """
    all_features = {}
    for element_type, config in NARRATIVE_ELEMENTS.items():
        all_features[element_type] = config["features"]
    
    prompt = f"""###PERSONA:
You are a world-famous narratologist and successful film scriptwriter.

###TEXT:
{text[:3000]}

###INSTRUCTIONS:
Analyze the provided text and extract specific narrative elements. 
Your goal is to deconstruct the story into a structured JSON format. 
Think about the subtext, not just the surface details.

1. **Characters**: Identify main and supporting characters. Analyze their roles, archetypes, and hidden motivations.
2. **Plot**: Break down the narrative arc (Setup, Inciting Incident, Rising Action, Climax, Resolution).
3. **Setting**: Describe both the physical location and the atmosphere/time period.
4. **Themes**: Identify the Primary Theme (the main message) and any Tertiary Themes (subtle motifs or symbols).

###TEMPLATE:
{{
    "characters": {{
        {", ".join([f'"{f}": "description"' for f in all_features["characters"]])}
    }},
    "plot": {{
        {", ".join([f'"{f}": "description"' for f in all_features["plot"]])}
    }},
    "setting": {{
        {", ".join([f'"{f}": "description"' for f in all_features["setting"]])}
    }},
    "themes": {{
        {", ".join([f'"{f}": "description"' for f in all_features["themes"]])}
    }}
}}

For each element type, provide detailed descriptions based on the text analysis."""
    
    result = call_openai_api(prompt)
    
    if result is None:
        logging.error("Failed to extract narrative elements. API call returned None.")
        raise ValueError("API call failed for narrative extraction")
    
    if not isinstance(result, dict):
        raise ValueError(f"Invalid response format: {type(result)}")
    
    if "characters" in result and isinstance(result["characters"], dict):
        chars = result["characters"]
        is_nested = any(isinstance(v, dict) and ("role" in v or "backstory" in v) for v in chars.values())
        if is_nested:
            best_char = None
            for name, info in chars.items():
                if isinstance(info, dict):
                    if not best_char:
                        best_char = info
                    role = info.get("role", "").lower()
                    if "protagonist" in role or "main" in role:
                        best_char = info
                        break
            if best_char:
                result["characters"] = best_char

    for element_type in NARRATIVE_ELEMENTS.keys():
        if element_type not in result:
            logging.warning(f"Missing {element_type} in response, creating empty dict")
            result[element_type] = {f: "" for f in NARRATIVE_ELEMENTS[element_type]["features"]}
        
        current_features = result[element_type]
        if isinstance(current_features, dict):
            for f in NARRATIVE_ELEMENTS[element_type]["features"]:
                if f not in current_features:
                    current_features[f] = ""
    
    return result


def extract_narrative_elements(text: str, element_type: str) -> Dict:
    """Extract narrative elements from text using LLM (legacy function for backward compatibility)"""
    all_elements = extract_all_narrative_elements(text)
    return all_elements.get(element_type, {f: "" for f in NARRATIVE_ELEMENTS[element_type]["features"]})


def feature_similarity(feat1: str, feat2: str) -> float:
    """Compute text similarity using word overlap"""
    if not feat1 or not feat2:
        return 0.0
    
    words1 = set(feat1.lower().split())
    words2 = set(feat2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1 & words2)
    total = len(words1 | words2)
    
    return overlap / total if total > 0 else 0.0


def compute_narrative_similarity(anchor_elements: Dict, 
                                 pos_elements: Dict, 
                                 neg_elements: Dict,
                                 element_type: str) -> Tuple[float, float]:
    """Compute similarity scores using narrative element alignment"""
    features = NARRATIVE_ELEMENTS[element_type]["features"]
    
    pos_scores = []
    neg_scores = []
    
    for feat in features:
        anchor_val = anchor_elements.get(feat, "")
        pos_val = pos_elements.get(feat, "")
        neg_val = neg_elements.get(feat, "")
        
        pos_sim = feature_similarity(anchor_val, pos_val)
        neg_sim = feature_similarity(anchor_val, neg_val)
        
        pos_scores.append(pos_sim)
        neg_scores.append(neg_sim)
    
    pos_sim_avg = np.mean(pos_scores) if pos_scores else 0.0
    neg_sim_avg = np.mean(neg_scores) if neg_scores else 0.0
    
    return pos_sim_avg, neg_sim_avg


def load_synthetic_narrative(path: str, num_of_lines: int = None) -> List[Tuple[str, str, str]]:
    """Load narrative triples from JSONL"""
    triples = []
    for obj in read_jsonl(path, num_of_lines):
        a = obj.get("anchor_text")
        p = obj.get("positive_text") or obj.get("text_a")
        n = obj.get("negative_text") or obj.get("text_b")
        
        if not p or not n:
            A = obj.get("text_a")
            B = obj.get("text_b")
            lbl = obj.get("text_a_is_closer")
            if A and B and lbl is not None:
                p, n = (A, B) if bool(lbl) else (B, A)
        
        if a and p and n:
            triples.append((a, p, n))
    return triples


def is_empty_cache(cached_dict: Dict) -> bool:
    """Check if cached dictionary is empty (all values are empty strings or empty dicts)"""
    if not cached_dict:
        return True
    
    for v in cached_dict.values():
        if isinstance(v, dict):
            if v and not is_empty_cache(v):
                return False
        elif isinstance(v, str):
            if v and v.strip():
                return False
        elif v:
            return False
    
    return True


def extract_or_cache_all_elements(text: str, text_id: str, cache_dir: str) -> Dict:
    """Extract ALL narrative elements with caching (optimized: 1 call instead of 4)"""
    cache_file = os.path.join(cache_dir, f"all_elements_{text_id}.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached = json.load(f)
            if cached and all(
                isinstance(cached.get(et), dict) and not is_empty_cache(cached.get(et, {}))
                for et in NARRATIVE_ELEMENTS.keys()
            ):
                return cached
            else:
                logging.warning(f"Found empty/invalid cache file {cache_file}, will re-extract")
                os.remove(cache_file)
    
    try:
        all_elements = extract_all_narrative_elements(text)
        
        if not all_elements or any(
            is_empty_cache(all_elements.get(et, {}))
            for et in NARRATIVE_ELEMENTS.keys()
        ):
            logging.error(f"API returned empty elements for {text_id}")
            raise ValueError(f"Empty extraction result for {text_id}")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(all_elements, f, indent=2)
        return all_elements
    except Exception as e:
        logging.error(f"Error extracting {text_id}: {e}")
        raise


def extract_or_cache_element(text: str, element_type: str, text_id: str, cache_dir: str) -> Dict:
    """Extract narrative element with caching (legacy function)"""
    all_elements = extract_or_cache_all_elements(text, text_id, cache_dir)
    return all_elements.get(element_type, {f: "" for f in NARRATIVE_ELEMENTS[element_type]["features"]})


def is_valid_narrative_cache(data: Dict) -> bool:
    """Check if narrative cache file has valid structure"""
    if not data or not isinstance(data, dict):
        return False
    
    required_keys = ["characters", "plot", "setting", "themes"]
    for element_type in required_keys:
        if element_type not in data:
            return False
        
        elem_data = data[element_type]
        if not isinstance(elem_data, dict):
            return False
        
        required_subkeys = ["pos_sim", "neg_sim", "margin", "anchor_elements", "pos_elements", "neg_elements"]
        for subkey in required_subkeys:
            if subkey not in elem_data:
                return False
        
        anchor_elem = elem_data.get("anchor_elements", {})
        if not isinstance(anchor_elem, dict) or not anchor_elem:
            return False
        
        features = NARRATIVE_ELEMENTS[element_type]["features"]
        for feat in features:
            if feat not in anchor_elem:
                return False
            feat_value = anchor_elem[feat]
            if isinstance(feat_value, str) and not feat_value.strip():
                return False
    
    return True


def process_triple(anchor: str, pos: str, neg: str, idx: Union[int, str], cache_dir: str, force_reextract: bool = False) -> Dict:
    """Process a single triple: extract elements and compute similarity (optimized: 3 calls instead of 12)
    
    Args:
        force_reextract: If True, re-extract even if cache exists
    """
    cache_file = os.path.join(cache_dir, f"narrative_{idx}.json")
    
    if os.path.exists(cache_file) and not force_reextract:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            if is_valid_narrative_cache(cached_data):
                return cached_data
            else:
                logging.warning(f"Invalid cache for {idx}, will re-extract")
                os.remove(cache_file)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Corrupted cache for {idx}: {e}, will re-extract")
            os.remove(cache_file)
    
    scores = {}
    
    anchor_all = extract_or_cache_all_elements(anchor, f"{idx}_anchor", cache_dir)
    pos_all = extract_or_cache_all_elements(pos, f"{idx}_pos", cache_dir)
    neg_all = extract_or_cache_all_elements(neg, f"{idx}_neg", cache_dir)
    
    for element_type in NARRATIVE_ELEMENTS.keys():
        anchor_elem = anchor_all.get(element_type, {})
        pos_elem = pos_all.get(element_type, {})
        neg_elem = neg_all.get(element_type, {})
        
        pos_sim, neg_sim = compute_narrative_similarity(
            anchor_elem, pos_elem, neg_elem, element_type
        )
        
        scores[element_type] = {
            'pos_sim': float(pos_sim),
            'neg_sim': float(neg_sim),
            'margin': float(pos_sim - neg_sim),
            'anchor_elements': anchor_elem,
            'pos_elements': pos_elem,
            'neg_elements': neg_elem
        }
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2)
    
    return scores


def main():
    logging.basicConfig(level=logging.INFO)
    
    if openai is None:
        logging.error("OpenAI not installed. pip install openai")
        return
    
    paths = {
        "path_train": "/kaggle/input/synthetic/synthetic_cleaned.jsonl",
        "path_dev": "/kaggle/input/semeval-task4/dev_track_a.jsonl",
        "path_test": "/kaggle/input/semeval-task4/dev_track_b.jsonl"
    }
    
    cache_dir = "/kaggle/working/cache_narrative"
    input_cache_dir = "/kaggle/input/cache-dir-narrative/cache_narrative"
    
    os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(input_cache_dir):
        import shutil as sh
        print(f"📂 Copying existing cache from {input_cache_dir} to {cache_dir}...")
        copied_count = 0
        for file in os.listdir(input_cache_dir):
            if file.endswith('.json'):
                src = os.path.join(input_cache_dir, file)
                dst = os.path.join(cache_dir, file)
                if not os.path.exists(dst):
                    sh.copy(src, dst)
                    copied_count += 1
        print(f"   ✅ Copied {copied_count} cache files")
    
    if wandb is None:
        logging.error("wandb not installed")
        return
    
    if wandb_key:
        wandb.login(key=wandb_key)
    
    run = wandb.init(
        project="semeval-narrative-extraction",
        entity="22520010-uit",
        name="narrative-extraction-full"
    )
    
    all_results = {}
    failed_samples = {"train": [], "dev": [], "test": []}
    
    if os.path.exists(paths["path_train"]):
        print(f"\n{'='*60}")
        print(f"📚 Processing TRAIN data: {paths['path_train']}")
        print(f"{'='*60}")
        triples = load_synthetic_narrative(paths["path_train"])
        print(f"   Loaded {len(triples)} triples")
        
        train_results = []
        for idx, (anchor, pos, neg) in enumerate(tqdm(triples, desc="Processing train")):
            try:
                result = process_triple(anchor, pos, neg, idx, cache_dir)
                train_results.append({
                    'idx': idx,
                    'anchor_text': anchor[:200],
                    'positive_text': pos[:200],
                    'negative_text': neg[:200],
                    'narrative_scores': result
                })
            except Exception as e:
                logging.warning(f"Failed train sample {idx}: {e}")
                failed_samples["train"].append(idx)
                continue
        
        all_results['train'] = train_results
        print(f"✅ Processed {len(train_results)}/{len(triples)} training samples")
        if failed_samples["train"]:
            print(f"⚠️  Failed: {len(failed_samples['train'])} samples")
    
    if os.path.exists(paths["path_dev"]):
        print(f"\n{'='*60}")
        print(f"📊 Processing DEV data: {paths['path_dev']}")
        print(f"{'='*60}")
        dev_triples = []
        for obj in read_jsonl(paths["path_dev"]):
            if {"anchor_text", "text_a", "text_b"} <= set(obj.keys()):
                dev_triples.append((
                    obj["anchor_text"],
                    obj["text_a"],
                    obj["text_b"]
                ))
        print(f"   Loaded {len(dev_triples)} triples")
        
        dev_results = []
        for idx, (anchor, a, b) in enumerate(tqdm(dev_triples, desc="Processing dev")):
            try:
                result = process_triple(anchor, a, b, f"dev_{idx}", cache_dir)
                dev_results.append({
                    'idx': idx,
                    'anchor_text': anchor[:200],
                    'text_a': a[:200],
                    'text_b': b[:200],
                    'narrative_scores': result
                })
            except Exception as e:
                logging.warning(f"Failed dev sample {idx}: {e}")
                failed_samples["dev"].append(idx)
                continue
        
        all_results['dev'] = dev_results
        print(f"✅ Processed {len(dev_results)}/{len(dev_triples)} dev samples")
        if failed_samples["dev"]:
            print(f"⚠️  Failed: {len(failed_samples['dev'])} samples")
    
    if os.path.exists(paths["path_test"]):
        print(f"\n{'='*60}")
        print(f"🧪 Processing TEST data: {paths['path_test']}")
        print(f"{'='*60}")
        test_triples = []
        for obj in read_jsonl(paths["path_test"]):
            if {"anchor_text", "text_a", "text_b"} <= set(obj.keys()):
                test_triples.append((
                    obj["anchor_text"],
                    obj["text_a"],
                    obj["text_b"]
                ))
        print(f"   Loaded {len(test_triples)} triples")
        
        test_results = []
        for idx, (anchor, a, b) in enumerate(tqdm(test_triples, desc="Processing test")):
            try:
                result = process_triple(anchor, a, b, f"test_{idx}", cache_dir)
                test_results.append({
                    'idx': idx,
                    'anchor_text': anchor[:200],
                    'text_a': a[:200],
                    'text_b': b[:200],
                    'narrative_scores': result
                })
            except Exception as e:
                logging.warning(f"Failed test sample {idx}: {e}")
                failed_samples["test"].append(idx)
                continue
        
        all_results['test'] = test_results
        print(f"✅ Processed {len(test_results)}/{len(test_triples)} test samples")
        if failed_samples["test"]:
            print(f"⚠️  Failed: {len(failed_samples['test'])} samples")
    
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    total_processed = sum(len(all_results.get(k, [])) for k in ['train', 'dev', 'test'])
    total_failed = sum(len(failed_samples[k]) for k in ['train', 'dev', 'test'])
    print(f"   Total processed: {total_processed}")
    print(f"   Total failed: {total_failed}")
    print(f"   Success rate: {total_processed/(total_processed+total_failed)*100:.1f}%" if total_processed+total_failed > 0 else "   N/A")
    
    output_file = "narrative_dataset_full.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Saved results to {output_file}")
    
    if any(failed_samples[k] for k in ['train', 'dev', 'test']):
        failed_file = "failed_samples.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_samples, f, indent=2)
        print(f"⚠️  Saved failed samples to {failed_file}")
    
    artifact = wandb.Artifact("narrative-dataset-full", type="dataset")
    artifact.add_file(output_file)
    
    cache_save_dir = "cache_saved"
    os.makedirs(cache_save_dir, exist_ok=True)

    for root, _, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.json'):
                src = os.path.join(root, file)
                dst = os.path.join(cache_save_dir, file)
                shutil.copy(src, dst)

    print(f"📁 Saved cache JSON files to: {cache_save_dir}")
    
    run.log_artifact(artifact)
    run.finish()
    print("\n🎉 EXTRACTION COMPLETE!")


if __name__ == "__main__":
    main()

