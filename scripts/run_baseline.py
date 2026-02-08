"""Local baseline evaluation script.

Runs inference on val_sentences.csv using the weight-blended ByT5 model,
with and without MBR decoding, and reports scores.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --max-samples 50  # quick test
"""

from __future__ import annotations

import io
import math
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import AkkadianPreprocessor, AkkadianPostprocessor
from mbr_decoder import MBRDecoder
from scripts.evaluate import evaluate, format_report


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CFG = {
    "val_path": PROJECT_ROOT / "data" / "val_sentences.csv",
    "models": [
        PROJECT_ROOT / "models" / "byt5-base-big-data2",
        PROJECT_ROOT / "models" / "byt5-akkadian-model",
        PROJECT_ROOT / "models" / "train-gap-all-2" / "train_GAP_all_2" / "byt5-base-akkadian_gap_setence2",
    ],
    "weights": [0.99, 0.98, 0.39],
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "max_len": 496,
    "batch_size": 4,

    # Generation
    "num_beams": 12,
    "num_return_sequences": 8,
    "max_new_tokens": 496,
    "length_penalty": 1.3,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_blended_model(device: torch.device) -> tuple:
    """Load 3 ByT5 models, blend weights, return (model, tokenizer)."""
    total_score = sum(CFG["weights"])
    W = [w / total_score for w in CFG["weights"]]

    model_paths = CFG["models"]

    # Check all models exist
    for p in model_paths:
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")

    print(f"  Loading model 1: {model_paths[0].name}")
    sd_m1 = AutoModelForSeq2SeqLM.from_pretrained(str(model_paths[0])).state_dict()

    print(f"  Loading model 2 (base): {model_paths[1].name}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(str(model_paths[1]))
    final_sd = base_model.state_dict()

    print(f"  Loading model 3: {model_paths[2].name}")
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
    model = base_model.to(device).eval().float()

    # Free memory
    del sd_m1, sd_m3
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(str(model_paths[1]))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Blended model: {param_count:,} parameters on {device}")

    return model, tokenizer


def load_lora_model(device: torch.device) -> tuple:
    """Load blended model + LoRA adapter."""
    model, tokenizer = load_blended_model(device)
    lora_path = PROJECT_ROOT / "models" / "lora_adapter"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")
    print(f"  Loading LoRA adapter from {lora_path}")
    model = PeftModel.from_pretrained(model, str(lora_path))
    model = model.merge_and_unload()
    model.eval()
    trainable = sum(p.numel() for p in model.parameters())
    print(f"  LoRA merged model: {trainable:,} parameters")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model,
    tokenizer,
    texts: list[str],
    preprocessor: AkkadianPreprocessor,
    postprocessor: AkkadianPostprocessor,
    device: torch.device,
    num_beams: int = 12,
    num_return_sequences: int = 8,
    length_penalty: float = 1.3,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> tuple[list[str], list[list[str]]]:
    """Run inference, returning (top1_results, all_candidates_per_sample).

    Returns:
        top1: List of top-1 beam search results (no MBR).
        all_candidates: List of candidate lists for MBR.
    """
    top1_results = []
    all_candidates = []

    total = len(texts)

    with torch.inference_mode():
        for idx, raw_text in enumerate(texts):
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"    [{idx+1}/{total}]")

            raw_str = str(raw_text) if raw_text is not None else ""

            # Phase 2: Short input dictionary lookup
            short_translation = postprocessor.get_short_input_translation(raw_str)
            if short_translation is not None:
                top1_results.append(short_translation)
                all_candidates.append([short_translation] if short_translation else [""])
                continue

            # Phase 2: Broken text detection
            if postprocessor.is_broken_text(raw_str):
                top1_results.append("...")
                all_candidates.append(["..."])
                continue

            input_text = preprocessor.preprocess(raw_str)

            inputs = tokenizer(
                input_text,
                max_length=CFG["max_len"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                max_new_tokens=CFG["max_new_tokens"],
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )

            candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            candidates = [postprocessor.postprocess(c) for c in candidates]

            # Phase 2: Cap output length based on input length
            candidates = [
                postprocessor.cap_output_length(c, raw_str)
                for c in candidates
            ]

            # Top-1 (before MBR)
            top1 = candidates[0] if candidates and candidates[0].strip() else ""
            top1_results.append(top1)

            # Filter empty for MBR
            filtered = [c for c in candidates if c.strip()]
            if not filtered:
                filtered = [""]
            all_candidates.append(filtered)

    return top1_results, all_candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples (0=all)")
    parser.add_argument("--skip-mbr", action="store_true", help="Skip MBR evaluation")
    parser.add_argument("--lora", action="store_true", help="Load LoRA adapter on top of blended model")
    args = parser.parse_args()

    start_time = time.time()

    # Load data
    print("[1/4] Loading validation data...")
    df = pd.read_csv(CFG["val_path"])

    # Drop rows with NaN translations (can't evaluate)
    df = df.dropna(subset=["transliteration", "translation"])
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    texts = df["transliteration"].tolist()
    references = df["translation"].tolist()
    print(f"  {len(texts)} samples loaded")

    # Load model
    if args.lora:
        print("\n[2/4] Loading blended model + LoRA adapter...")
        model, tokenizer = load_lora_model(CFG["device"])
    else:
        print("\n[2/4] Loading blended model...")
        model, tokenizer = load_blended_model(CFG["device"])

    # Run inference
    preprocessor = AkkadianPreprocessor(max_length=800, add_prefix=True)
    postprocessor = AkkadianPostprocessor(remove_brackets=True, remove_big_gap=True)

    print(f"\n[3/4] Running inference (beams={CFG['num_beams']}, "
          f"seqs={CFG['num_return_sequences']}, LP={CFG['length_penalty']})...")

    top1_results, all_candidates = run_inference(
        model, tokenizer, texts, preprocessor, postprocessor,
        CFG["device"],
        num_beams=CFG["num_beams"],
        num_return_sequences=CFG["num_return_sequences"],
        length_penalty=CFG["length_penalty"],
        repetition_penalty=CFG["repetition_penalty"],
        no_repeat_ngram_size=CFG["no_repeat_ngram_size"],
    )

    # Evaluate
    print(f"\n[4/4] Evaluating...")

    # --- Pattern A: Top-1 (no MBR) ---
    print("\n" + "=" * 60)
    print("  Pattern A: Top-1 (no MBR)")
    scores_top1 = evaluate(top1_results, references)
    print(format_report(scores_top1))

    if not args.skip_mbr:
        # --- Pattern B: MBR with chrF++ ---
        print("\n  Pattern B: MBR (chrF++)")
        mbr_chrf = MBRDecoder(utility="chrf")
        mbr_chrf_results = []
        changed_b = 0
        for candidates in all_candidates:
            if len(candidates) > 1:
                best, _, details = mbr_chrf.decode(candidates)
                if details.get("best_idx", 0) != 0:
                    changed_b += 1
            else:
                best = candidates[0]
            mbr_chrf_results.append(best)

        scores_mbr_chrf = evaluate(mbr_chrf_results, references)
        print(format_report(scores_mbr_chrf))
        print(f"  MBR changed selection: {changed_b}/{len(all_candidates)}")

        # --- Pattern C: MBR with geo_mean ---
        print("\n  Pattern C: MBR (geo_mean)")
        mbr_geo = MBRDecoder(utility="geo_mean")
        mbr_geo_results = []
        changed_c = 0
        for candidates in all_candidates:
            if len(candidates) > 1:
                best, _, details = mbr_geo.decode(candidates)
                if details.get("best_idx", 0) != 0:
                    changed_c += 1
            else:
                best = candidates[0]
            mbr_geo_results.append(best)

        scores_mbr_geo = evaluate(mbr_geo_results, references)
        print(format_report(scores_mbr_geo))
        print(f"  MBR changed selection: {changed_c}/{len(all_candidates)}")

    # Summary
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Method':<25s} {'BLEU':>8s} {'chrF++':>8s} {'GeoMean':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'A: Top-1':<25s} {scores_top1['bleu']:>8.2f} {scores_top1['chrf']:>8.2f} {scores_top1['geo_mean']:>8.2f}")
    if not args.skip_mbr:
        print(f"  {'B: MBR chrF++':<25s} {scores_mbr_chrf['bleu']:>8.2f} {scores_mbr_chrf['chrf']:>8.2f} {scores_mbr_chrf['geo_mean']:>8.2f}")
        print(f"  {'C: MBR geo_mean':<25s} {scores_mbr_geo['bleu']:>8.2f} {scores_mbr_geo['chrf']:>8.2f} {scores_mbr_geo['geo_mean']:>8.2f}")
    print(f"{'=' * 60}")
    print(f"  Total time: {elapsed:.1f} minutes")

    # Save results
    results_path = PROJECT_ROOT / "data" / "baseline_results.csv"
    result_df = pd.DataFrame({
        "transliteration": texts,
        "reference": references,
        "top1": top1_results,
    })
    if not args.skip_mbr:
        result_df["mbr_chrf"] = mbr_chrf_results
        result_df["mbr_geo"] = mbr_geo_results
    result_df.to_csv(results_path, index=False, encoding="utf-8")
    print(f"  Results saved: {results_path}")


if __name__ == "__main__":
    main()
