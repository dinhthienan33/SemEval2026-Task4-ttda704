#!/usr/bin/env python3

import os
import json
import argparse
import logging
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BigBirdModel, AutoConfig

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  peft not available. PeftModel loading will be skipped.")

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  unsloth not available. Install with: pip install unsloth")

logging.basicConfig(level=logging.INFO)


def read_jsonl(path):
    """Read JSONL file line by line"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_dev_triples(path: str) -> List[Tuple[str, str, str, bool]]:
    """Load dev set for evaluation"""
    triples = []
    for o in read_jsonl(path):
        if {"anchor_text", "text_a", "text_b", "text_a_is_closer"} <= set(o.keys()):
            triples.append((
                o["anchor_text"], 
                o["text_a"], 
                o["text_b"], 
                bool(o["text_a_is_closer"])
            ))
    return triples


class EmbedModel(nn.Module):
    """Embedding model - matches train_taskb.py and supports HuggingFace story-emb"""
    
    def __init__(self, backbone_name="sentence-transformers/all-mpnet-base-v2",
                 dropout=0.1, pooling='mean', backbone=None, use_query_prefix=False):
        super().__init__()
        if backbone is not None:
                                                                     
            self.backbone = backbone
        else:
                                     
            try:
                self.backbone = BigBirdModel.from_pretrained(backbone_name, trust_remote_code=True)
            except Exception as e:
                print(f"⚠️  Failed to load BigBirdModel: {e}")
                self.backbone = AutoModel.from_pretrained(backbone_name, trust_remote_code=True)
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.use_query_prefix = use_query_prefix
        self.query_prefix = "Retrieve stories with a similar narrative to the given story: "
    
    def mean_pooling(self, hidden_state, attention_mask):
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def last_token_pooling(self, hidden_state, attention_mask):
        """Last token pooling for causal LMs (Mistral)"""
                                                        
                                                           
        if attention_mask is not None:
                                                           
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_state.size(0)
            last_hidden_states = hidden_state[torch.arange(batch_size), sequence_lengths]
        else:
                                    
            last_hidden_states = hidden_state[:, -1]
        return last_hidden_states
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass"""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        else:
            hidden = out[0]
        
                          
        if self.pooling == 'last_token' and attention_mask is not None:
            emb = self.last_token_pooling(hidden, attention_mask)
        elif self.pooling == 'mean' and attention_mask is not None:
            emb = self.mean_pooling(hidden, attention_mask)
        else:
            emb = hidden.mean(dim=1)
        
        emb = self.dropout(emb)
        emb = F.normalize(emb, p=2, dim=1)
        return emb
    
    @classmethod
    def from_pretrained(cls, model_dir: str, use_query_prefix: bool = False):
        """
        Load model from saved directory or HuggingFace Hub
        
        Args:
            model_dir: Path to local directory or HuggingFace model ID (e.g., "uhhlt/story-emb")
            use_query_prefix: Whether to use query prefix (for story-emb model)
        """
                                              
                                                                                
        is_hf_hub = "/" in model_dir and not os.path.exists(model_dir) and not os.path.isabs(model_dir)
        
        if is_hf_hub:
            print(f"📦 Loading from HuggingFace Hub: {model_dir}")
            return cls._from_huggingface(model_dir, use_query_prefix)
        else:
            print(f"📦 Loading from local directory: {model_dir}")
            return cls._from_local(model_dir, use_query_prefix)
    
    @classmethod
    def _from_huggingface(cls, model_id: str, use_query_prefix: bool = False):
        """Load model from HuggingFace Hub (e.g., uhhlt/story-emb) using unsloth for memory efficiency"""
        backbone = None
        pooling = 'mean'
        
                                                                                               
        if UNSLOTH_AVAILABLE:
            try:
                print(f"🚀 Attempting to load with unsloth (memory-efficient)...")
                                                 
                is_peft = False
                base_model_name = model_id
                
                if PEFT_AVAILABLE:
                    try:
                        peft_config = PeftConfig.from_pretrained(model_id)
                        base_model_name = peft_config.base_model_name_or_path
                        is_peft = True
                        print(f"📋 Detected PeftModel with base: {base_model_name}")
                    except Exception:
                        pass                                          
                
                                                                 
                model_unsloth, tokenizer_unsloth = FastLanguageModel.from_pretrained(
                    model_name=base_model_name,
                    max_seq_length=2048,
                    dtype=None,               
                    load_in_4bit=True,                                                
                )
                
                                                                     
                if is_peft and PEFT_AVAILABLE:
                    print(f"📦 Loading PEFT adapters from {model_id}...")
                                                                        
                    backbone = PeftModel.from_pretrained(
                        model_unsloth.model if hasattr(model_unsloth, 'model') else model_unsloth,
                        model_id,
                        is_trainable=False,
                    )
                    print("✅ Loaded PeftModel with unsloth")
                else:
                                                            
                    backbone = model_unsloth.model if hasattr(model_unsloth, 'model') else model_unsloth
                    print("✅ Loaded model with unsloth")
                
                                                                      
                if "mistral" in base_model_name.lower() or "llama" in base_model_name.lower():
                    pooling = 'last_token'
                    use_query_prefix = True
                
            except Exception as e:
                print(f"⚠️  Failed to load with unsloth: {e}")
                print("🔄 Falling back to standard loading...")
                backbone = None
        
                                                                         
        if backbone is None:
            try:
                                                                              
                if PEFT_AVAILABLE:
                    try:
                        peft_config = PeftConfig.from_pretrained(model_id)
                        base_model_name = peft_config.base_model_name_or_path
                        print(f"📋 Detected PeftModel with base: {base_model_name}")
                        
                                         
                        base_model = AutoModel.from_pretrained(
                            base_model_name, 
                            trust_remote_code=True,
                            device_map="auto"
                        )
                        
                                       
                        backbone = PeftModel.from_pretrained(
                            base_model,
                            model_id,
                            is_trainable=False,
                            device_map="auto"
                        )
                        print("✅ Loaded PeftModel from HuggingFace")
                        
                                                                       
                        pooling = 'last_token'
                        use_query_prefix = True                                     
                        
                    except Exception as e:
                        print(f"⚠️  Not a PeftModel, trying AutoModel: {e}")
                                                          
                        backbone = AutoModel.from_pretrained(
                            model_id,
                            trust_remote_code=True,
                            device_map="auto"
                        )
                        print("✅ Loaded AutoModel from HuggingFace")
                        pooling = 'mean'
                else:
                                                                          
                                                                                    
                    backbone = AutoModel.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        device_map="auto"
                    )
                    print("✅ Loaded AutoModel from HuggingFace (PEFT not available)")
                    pooling = 'mean'
            except Exception as e:
                raise RuntimeError(f"Failed to load model from HuggingFace Hub {model_id}: {e}")
        
        model = cls(
            backbone_name="",
            dropout=0.1,
            pooling=pooling,
            backbone=backbone,
            use_query_prefix=use_query_prefix
        )
        model._loaded_with_device_map = True
        return model
    
    @classmethod
    def _from_local(cls, model_dir: str, use_query_prefix: bool = False):
        """Load model from local directory"""
        config_path = os.path.join(model_dir, 'embed_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            dropout = config.get('dropout', 0.1)
            pooling = config.get('pooling', 'mean')
            use_query_prefix = config.get('use_query_prefix', use_query_prefix)
            print(f"📋 Loaded config: dropout={dropout}, pooling={pooling}, use_query_prefix={use_query_prefix}")
        else:
            dropout = 0.1
            pooling = 'mean'
            print(f"⚠️  Config file not found, using defaults: dropout={dropout}, pooling={pooling}")
        
                                   
        adapter_config_path = os.path.join(model_dir, 'adapter_config.json')
        if os.path.exists(adapter_config_path) and PEFT_AVAILABLE:
            try:
                peft_config = PeftConfig.from_pretrained(model_dir)
                base_model_name = peft_config.base_model_name_or_path
                print(f"📋 Detected PeftModel with base: {base_model_name}")
                
                base_model = AutoModel.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    device_map="auto"
                )
                backbone = PeftModel.from_pretrained(
                    base_model,
                    model_dir,
                    is_trainable=False,
                    device_map="auto"
                )
                print("✅ Loaded PeftModel from local directory")
                if pooling == 'mean':
                    pooling = 'last_token'                                    
            except Exception as e:
                print(f"⚠️  Failed to load as PeftModel: {e}")
                                             
                try:
                    backbone = BigBirdModel.from_pretrained(model_dir, trust_remote_code=True)
                    print("✅ Loaded BigBirdModel")
                except Exception as e2:
                    backbone = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
                    print("✅ Loaded AutoModel")
        else:
                                                
                                                                
            try:
                backbone = BigBirdModel.from_pretrained(model_dir, trust_remote_code=True)
                print("✅ Loaded BigBirdModel")
            except Exception as e:
                print(f"⚠️  Failed to load BigBirdModel from {model_dir}: {e}")
                try:
                    backbone = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
                    print("✅ Loaded AutoModel")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model from {model_dir}: {e2}")
        
                                                        
        model = cls(
            backbone_name="",
            dropout=dropout,
            pooling=pooling,
            backbone=backbone,
            use_query_prefix=use_query_prefix
        )
        
        return model


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size=32, max_len=1024, use_query_prefix=False):
    """Encode texts to embeddings - matches train_taskb.py and story-emb"""
    model.eval()
    all_emb = []
    
                                                      
    if use_query_prefix and hasattr(model, 'query_prefix'):
        texts = [model.query_prefix + text for text in texts]
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding stories"):
        batch = texts[i:i+batch_size]
        
                                                     
        if hasattr(tokenizer, 'padding_side') and tokenizer.padding_side == 'left':
            tokens = tokenizer(batch, truncation=True, padding=True,
                              max_length=max_len, return_tensors='pt').to(device)
        else:
            tokens = tokenizer(batch, truncation=True, padding=True,
                              max_length=max_len, return_tensors='pt').to(device)
        
        emb = model(**tokens)
        all_emb.append(emb.cpu())
    
    all_emb = torch.cat(all_emb, dim=0)
    all_emb = F.normalize(all_emb, p=2, dim=1)
    return all_emb.cpu().numpy().astype(np.float32)


@torch.no_grad()
def evaluate(model, tokenizer, dev_triples, device, max_len=1024, use_query_prefix=False):
    """Evaluate model on dev set - matches semeval-track-b.py pattern"""
    model.eval()
    correct = 0
    total = len(dev_triples)
    
                                
    query_prefix = ""
    if use_query_prefix and hasattr(model, 'query_prefix'):
        query_prefix = model.query_prefix
    
    for anchor, a, b, label in tqdm(dev_triples, desc="Evaluating"):
        def get_emb(t):
            text = query_prefix + t if query_prefix else t
            tokens = tokenizer(text, truncation=True, padding='max_length',
                             max_length=max_len, return_tensors='pt').to(device)
            e = model(**tokens)
            return e.squeeze(0)
        
        va, vA, vB = get_emb(anchor), get_emb(a), get_emb(b)
        simA = torch.dot(va, vA).item()
        simB = torch.dot(va, vB).item()
        
                                                                       
        if (label and simA > simB) or ((not label) and simA < simB):
            correct += 1
    
    acc = correct / total if total > 0 else 0.0
    print(f"\n✅ Dev Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    
                                           
                                                                      
    is_hf_hub = "/" in args.model_dir and not os.path.exists(args.model_dir) and not os.path.isabs(args.model_dir)
    
                
    print(f"\n📦 Loading model from: {args.model_dir}")
    
    if is_hf_hub and not PEFT_AVAILABLE:
        print("⚠️  Warning: PEFT not available. Loading full model (may be slower and use more memory).")
        print("   Consider installing: pip install peft")
    
    if not is_hf_hub and not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
                    
    if is_hf_hub:
                                                            
        tokenizer_name = args.model_dir
        if PEFT_AVAILABLE:
            try:
                peft_config = PeftConfig.from_pretrained(args.model_dir)
                tokenizer_name = peft_config.base_model_name_or_path
            except:
                pass
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"                             
    
                
    use_query_prefix = args.use_query_prefix or is_hf_hub                             
    model = EmbedModel.from_pretrained(args.model_dir, use_query_prefix=use_query_prefix)
    
                                                                          
                                                                                           
    if hasattr(model, '_loaded_with_device_map') and model._loaded_with_device_map:
                                                       
        print("✅ Model already on device (loaded with device_map='auto')")
    elif hasattr(model.backbone, 'hf_device_map'):
                                                           
        print("✅ Model already on device (has hf_device_map)")
    else:
                                          
        model = model.to(device)
        print(f"✅ Model moved to {device}")
    
    model.eval()
    print("✅ Model loaded successfully")
    
    if model.use_query_prefix:
        print(f"📝 Using query prefix: {model.query_prefix}")
    
                                           
    if args.dev_track_b and os.path.exists(args.dev_track_b):
        print(f"\n📝 Encoding stories from: {args.dev_track_b}")
        texts = [obj["text"] for obj in read_jsonl(args.dev_track_b)]
        print(f"   Found {len(texts)} stories")
        
        embs = encode_texts(model, tokenizer, texts, device,
                          batch_size=args.batch_size, max_len=args.max_len,
                          use_query_prefix=model.use_query_prefix)
        
                         
        output_file = args.output_embeddings or "track_b_embeddings.npy"
        np.save(output_file, embs)
        print(f"✅ Saved embeddings to: {output_file} (shape: {embs.shape})")
    
                                   
    if args.dev_track_a and os.path.exists(args.dev_track_a):
        print(f"\n📊 Evaluating on: {args.dev_track_a}")
        dev_triples = load_dev_triples(args.dev_track_a)
        print(f"   Found {len(dev_triples)} evaluation triples")
        
        acc = evaluate(model, tokenizer, dev_triples, device, max_len=args.max_len,
                      use_query_prefix=model.use_query_prefix)
        
                                 
        if args.output_results:
            results = {
                "accuracy": float(acc),
                "total_samples": len(dev_triples),
                "model_dir": args.model_dir,
                "dev_file": args.dev_track_a
            }
            with open(args.output_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to: {args.output_results}")
    else:
        print(f"⚠️  Dev track A file not found: {args.dev_track_a}")
    
    print("\n🎉 Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SemEval Track B - Inference Script"
    )
    
           
    parser.add_argument("--model-dir", type=str, required=False, default="uhhlt/story-emb",
                       help="Directory containing saved model or HuggingFace model ID (e.g., 'hf_model_trackB_narrative' or 'uhhlt/story-emb')")
    parser.add_argument("--use-query-prefix", action="store_true", default=False,
                       help="Use query prefix for story-emb model (auto-detected for HuggingFace models)")
    
                
    parser.add_argument("--dev-track-b", type=str, default="test/dev_track_b.jsonl",
                       help="Path to dev_track_b.jsonl (stories to encode)")
    parser.add_argument("--dev-track-a", type=str, default="test/dev_track_a.jsonl",
                       help="Path to dev_track_a.jsonl (evaluation triples)")
    
                        
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size for encoding")
    parser.add_argument("--max-len", type=int, default=320,
                       help="Max sequence length")
    
            
    parser.add_argument("--output-embeddings", type=str, default=None,
                       help="Output file for embeddings (default: track_b_embeddings.npy)")
    parser.add_argument("--output-results", type=str, default=None,
                       help="Output JSON file for evaluation results")
    
    args = parser.parse_args()
    main(args)

