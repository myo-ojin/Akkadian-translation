"""LoRA fine-tuning of blended ByT5 model on Akkademia + competition data.

Normalizes Akkademia transliteration format to match competition format,
then fine-tunes with LoRA for improved translation quality.

Usage:
    python scripts/lora_finetune.py
    python scripts/lora_finetune.py --epochs 5 --lr 3e-4
"""

from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Format normalization: Akkademia → Competition style
# ---------------------------------------------------------------------------
def normalize_akkademia_transliteration(text: str) -> str:
    """Convert Akkademia transliteration format to competition-like format.

    Akkademia: {d}-enlil  {KI}  {GIŠ}-TUKUL
    Competition: (d)enlil  KI  (GIŠ)TUKUL
    """
    # {d}-X → (d)X  (determinative with hyphen)
    text = re.sub(r"\{([^}]+)\}-", r"(\1)", text)
    # {X} → X  (standalone determinative)
    text = re.sub(r"\{([^}]+)\}", r"\1", text)
    return text


def normalize_akkademia_translation(text: str) -> str:
    """Clean Akkademia English translations for training."""
    # Remove parenthetical annotations like (lit. "xxx") but keep content parens
    text = re.sub(r'\(lit\.\s*"[^"]*"\)', '', text)
    # Remove (DN and) style annotations
    text = re.sub(r'\(DN\s+and\)', '', text)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TranslationDataset(Dataset):
    def __init__(self, sources: list[str], targets: list[str],
                 tokenizer, max_source_len: int = 512, max_target_len: int = 256):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src = "translate Akkadian to English: " + self.sources[idx]
        tgt = self.targets[idx]

        source_enc = self.tokenizer(
            src, max_length=self.max_source_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        target_enc = self.tokenizer(
            tgt, max_length=self.max_target_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )

        labels = target_enc.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc.input_ids.squeeze(),
            "attention_mask": source_enc.attention_mask.squeeze(),
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Model loading (same as run_baseline.py)
# ---------------------------------------------------------------------------
def load_blended_model(device):
    """Load 3 ByT5 models, blend weights, return (model, tokenizer)."""
    model_paths = [
        PROJECT_ROOT / "models" / "byt5-base-big-data2",
        PROJECT_ROOT / "models" / "byt5-akkadian-model",
        PROJECT_ROOT / "models" / "train-gap-all-2" / "train_GAP_all_2" / "byt5-base-akkadian_gap_setence2",
    ]
    weights = [0.99, 0.98, 0.39]
    total = sum(weights)
    W = [w / total for w in weights]

    print("  Loading model 1...")
    sd_m1 = AutoModelForSeq2SeqLM.from_pretrained(str(model_paths[0])).state_dict()
    print("  Loading model 2 (base)...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(str(model_paths[1]))
    final_sd = base_model.state_dict()
    print("  Loading model 3...")
    sd_m3 = AutoModelForSeq2SeqLM.from_pretrained(str(model_paths[2])).state_dict()

    print("  Blending weights...")
    for k in final_sd:
        val = W[1] * final_sd[k]
        norm = W[1]
        if k in sd_m1:
            val = val + W[0] * sd_m1[k]
            norm = norm + W[0]
        if k in sd_m3:
            val = val + W[2] * sd_m3[k]
            norm = norm + W[2]
        final_sd[k] = val / norm

    base_model.load_state_dict(final_sd)
    model = base_model.to(device).float()
    del sd_m1, sd_m3
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(str(model_paths[1]))
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Blended model: {param_count:,} parameters on {device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit training samples (0=all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # === 1. Load data ===
    print("[1/5] Loading and normalizing data...")

    # Akkademia data
    akk_dir = PROJECT_ROOT / "data" / "external" / "akkademia"
    with open(akk_dir / "train.tr") as f:
        akk_sources = [normalize_akkademia_transliteration(line.strip()) for line in f]
    with open(akk_dir / "train.en") as f:
        akk_targets = [normalize_akkademia_translation(line.strip()) for line in f]

    # Competition data
    comp_df = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
    comp_sources = comp_df["transliteration"].tolist()
    comp_targets = comp_df["translation"].tolist()

    # Filter out empty pairs
    paired = [(s, t) for s, t in zip(akk_sources, akk_targets)
              if s.strip() and t.strip() and len(s) > 5 and len(t) > 5]
    akk_sources, akk_targets = zip(*paired) if paired else ([], [])

    paired_comp = [(s, t) for s, t in zip(comp_sources, comp_targets)
                   if isinstance(s, str) and isinstance(t, str) and s.strip() and t.strip()]
    comp_sources, comp_targets = zip(*paired_comp) if paired_comp else ([], [])

    # Competition data gets 3x weight (more relevant)
    all_sources = list(akk_sources) + list(comp_sources) * 3
    all_targets = list(akk_targets) + list(comp_targets) * 3

    if args.max_samples > 0:
        all_sources = all_sources[:args.max_samples]
        all_targets = all_targets[:args.max_samples]

    print(f"  Akkademia: {len(akk_sources)} pairs")
    print(f"  Competition: {len(comp_sources)} pairs (x3 weight = {len(comp_sources)*3})")
    print(f"  Total training: {len(all_sources)} pairs")

    # === 2. Load model ===
    print("\n[2/5] Loading blended model...")
    model, tokenizer = load_blended_model(device)

    # === 3. Apply LoRA ===
    print(f"\n[3/5] Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q", "v", "o"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # === 4. Train ===
    print(f"\n[4/5] Training (epochs={args.epochs}, lr={args.lr}, "
          f"batch={args.batch_size}, grad_accum={args.grad_accum})...")

    dataset = TranslationDataset(all_sources, all_targets, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs // args.grad_accum
    print(f"  Steps per epoch: {len(dataloader)}, total optimization steps: {total_steps}")

    model.train()
    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    avg = epoch_loss / (batch_idx + 1)
                    elapsed = (time.time() - start_time) / 60
                    print(f"    Step {global_step}, loss: {avg:.4f}, time: {elapsed:.1f}min")

        avg_loss = epoch_loss / len(dataloader)
        elapsed = (time.time() - start_time) / 60
        print(f"  Epoch {epoch+1}/{args.epochs}: avg_loss={avg_loss:.4f} ({elapsed:.1f}min)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = PROJECT_ROOT / "models" / "lora_adapter"
            model.save_pretrained(str(save_path))
            print(f"  Saved best adapter → {save_path}")

    # === 5. Save final ===
    print(f"\n[5/5] Training complete!")
    save_path = PROJECT_ROOT / "models" / "lora_adapter"
    model.save_pretrained(str(save_path))

    elapsed = (time.time() - start_time) / 60
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Total time: {elapsed:.1f} minutes")
    print(f"  Adapter saved: {save_path}")


if __name__ == "__main__":
    main()
