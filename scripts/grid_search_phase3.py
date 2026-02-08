"""Grid search for Phase 3: Length cap coefficients + Diverse Beam Search.

Tests on multi-token val samples only (15+ tokens), which match test set profile.
Reuses already-generated model outputs to test different cap parameters quickly.

Usage:
    python scripts/grid_search_phase3.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import AkkadianPreprocessor, AkkadianPostprocessor
from scripts.evaluate import evaluate


def load_blended_model(device):
    """Load 3 ByT5 models, blend weights."""
    models_dir = PROJECT_ROOT / "models"
    model_paths = [
        models_dir / "byt5-base-big-data2",
        models_dir / "byt5-akkadian-model",
        models_dir / "train-gap-all-2" / "train_GAP_all_2" / "byt5-base-akkadian_gap_setence2",
    ]
    weights = [0.99, 0.98, 0.39]
    total = sum(weights)
    W = [w / total for w in weights]

    sd_m1 = AutoModelForSeq2SeqLM.from_pretrained(str(model_paths[0])).state_dict()
    base_model = AutoModelForSeq2SeqLM.from_pretrained(str(model_paths[1]))
    final_sd = base_model.state_dict()
    sd_m3 = AutoModelForSeq2SeqLM.from_pretrained(str(model_paths[2])).state_dict()

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
    del sd_m1, sd_m3
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(str(model_paths[1]))
    return model, tokenizer


def run_inference_raw(model, tokenizer, texts, preprocessor, device, **gen_kwargs):
    """Run inference and return raw candidates (before postprocessing)."""
    all_raw_candidates = []
    with torch.inference_mode():
        for idx, raw_text in enumerate(texts):
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"    [{idx+1}/{len(texts)}]")

            raw_str = str(raw_text) if raw_text is not None else ""
            input_text = preprocessor.preprocess(raw_str)
            inputs = tokenizer(
                input_text, max_length=496, padding=True,
                truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **gen_kwargs,
            )
            candidates = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_raw_candidates.append(candidates)
    return all_raw_candidates


def apply_postprocessing(raw_candidates, texts, postprocessor, coeff, offset):
    """Apply postprocessing with given cap parameters, return top-1."""
    results = []
    for candidates, raw_str in zip(raw_candidates, texts):
        processed = [postprocessor.postprocess(c) for c in candidates]
        # Custom cap
        capped = []
        for c in processed:
            input_len = len(str(raw_str))
            max_len = int(input_len * coeff + offset)
            if len(c) <= max_len:
                capped.append(c)
            else:
                words = c.split()
                result_words = []
                length = 0
                for w in words:
                    if length + len(w) + 1 > max_len and length > 0:
                        break
                    result_words.append(w)
                    length += len(w) + 1
                capped.append(" ".join(result_words))
        top1 = capped[0] if capped and capped[0].strip() else ""
        results.append(top1)
    return results


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data - multi-token only (matching test set profile)
    df = pd.read_csv(PROJECT_ROOT / "data" / "val_sentences.csv")
    df = df.dropna(subset=["transliteration", "translation"])
    # Filter to multi-token samples (>= 10 tokens to match test profile)
    df["token_count"] = df["transliteration"].str.split().str.len()
    df_multi = df[df["token_count"] >= 10].copy()
    print(f"Multi-token samples (10+ tokens): {len(df_multi)}")

    texts = df_multi["transliteration"].tolist()
    references = df_multi["translation"].tolist()

    # Load model
    print("\nLoading blended model...")
    model, tokenizer = load_blended_model(device)
    preprocessor = AkkadianPreprocessor(max_length=800, add_prefix=True)
    postprocessor = AkkadianPostprocessor(remove_brackets=True, remove_big_gap=True)

    # === Test 1: Standard beam search (baseline) ===
    print("\n=== Standard Beam Search (beams=12, LP=1.3) ===")
    raw_standard = run_inference_raw(
        model, tokenizer, texts, preprocessor, device,
        num_beams=12, num_return_sequences=8, max_new_tokens=496,
        length_penalty=1.3, early_stopping=True,
    )

    # === Test 2: Diverse Beam Search ===
    print("\n=== Diverse Beam Search (beams=12, groups=4, div=0.5) ===")
    try:
        raw_diverse = run_inference_raw(
            model, tokenizer, texts, preprocessor, device,
            num_beams=12, num_beam_groups=4, diversity_penalty=0.5,
            num_return_sequences=8, max_new_tokens=496,
            length_penalty=1.3, early_stopping=True,
            custom_generate="transformers-community/group-beam-search",
            trust_remote_code=True,
        )
        beam_configs = [("Standard", raw_standard), ("Diverse", raw_diverse)]
    except Exception as e:
        print(f"  Diverse Beam failed: {e}")
        print("  Continuing with Standard Beam only.")
        beam_configs = [("Standard", raw_standard)]

    # === Grid search over cap coefficients ===
    print("\n" + "=" * 70)
    print("  LENGTH CAP GRID SEARCH (multi-token samples)")
    print("=" * 70)

    coeffs = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    offsets = [20, 30, 40, 50]

    best_geo = 0
    best_cfg = {}

    for beam_type, raw_cands in beam_configs:
        print(f"\n--- {beam_type} Beam ---")
        print(f"  {'Coeff':>6s} {'Offset':>6s} {'BLEU':>8s} {'chrF++':>8s} {'GeoMean':>8s}")
        print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

        for coeff in coeffs:
            for offset in offsets:
                results = apply_postprocessing(raw_cands, texts, postprocessor, coeff, offset)
                scores = evaluate(results, references)
                geo = scores["geo_mean"]
                marker = " ***" if geo > best_geo else ""
                print(f"  {coeff:>6.1f} {offset:>6d} {scores['bleu']:>8.2f} {scores['chrf']:>8.2f} {geo:>8.2f}{marker}")
                if geo > best_geo:
                    best_geo = geo
                    best_cfg = {"beam": beam_type, "coeff": coeff, "offset": offset,
                                "bleu": scores["bleu"], "chrf": scores["chrf"], "geo": geo}

    # Also test NO cap
    for beam_type, raw_cands in [("Standard", raw_standard), ("Diverse", raw_diverse)]:
        results = apply_postprocessing(raw_cands, texts, postprocessor, 999, 9999)
        scores = evaluate(results, references)
        print(f"\n  {beam_type} (no cap): BLEU={scores['bleu']:.2f} chrF++={scores['chrf']:.2f} GeoMean={scores['geo_mean']:.2f}")

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'=' * 70}")
    print(f"  BEST CONFIG: {best_cfg}")
    print(f"  Total time: {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
