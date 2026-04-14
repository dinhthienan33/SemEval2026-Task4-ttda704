import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import numpy as np

from src.utils.metrics import evaluate

try:
    import wandb
except ImportError:
    wandb = None


def train_one_stage(model, train_loader, dev_triples, optimizer, scheduler, 
                    criterion, device, num_epochs, stage_name, tokenizer=None,
                    use_amp=True, patience=5, eval_every=1):
    """
    Train model for one stage with validation-based checkpointing
    """
    scaler = GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
    
    if device == 'cpu':
        use_amp = False
        scaler = None

    best_dev_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting {stage_name}")
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
        
        avg_loss = total_loss / max(1, num_batches)
        print(f"📊 {stage_name} Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        if (epoch + 1) % eval_every == 0 and dev_triples:
            if tokenizer is None:
                raise ValueError("Tokenizer required for evaluation")
            dev_acc = evaluate(model, tokenizer, dev_triples, device)
            print(f"✅ Dev Accuracy: {dev_acc:.4f} (Best: {best_dev_acc:.4f})")
            
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"🎯 New best model saved! Dev Acc: {best_dev_acc:.4f}")
                
            else:
                patience_counter += 1
                print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                    break
        
        if wandb and wandb.run is not None:
            wandb.log({
                f"{stage_name}_epoch": epoch + 1,
                f"{stage_name}_loss": avg_loss,
                f"{stage_name}_dev_acc": dev_acc if dev_triples else 0
            })
    
    return best_model_state, best_dev_acc
