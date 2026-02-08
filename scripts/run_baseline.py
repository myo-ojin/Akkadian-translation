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
import re
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
from translation_retriever import TranslationRetriever
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
    parser.add_argument("--tr-try", action="store_true", help="Enable TR-TRY hybrid retrieval")
    parser.add_argument("--tr-try-only", action="store_true", help="Use TR-TRY only (no model inference)")
    parser.add_argument("--tr-threshold", type=float, default=0.14, help="TR-TRY min acceptance score")
    parser.add_argument("--tr-long-thresh", type=int, default=256, help="Input char length to trigger TR-TRY append")
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

    # Load TR-TRY retriever
    retriever = None
    tr_try_results = None
    if args.tr_try or args.tr_try_only:
        print("\n[TR-TRY] Loading translation retriever...")
        train_path = PROJECT_ROOT / "data" / "train.csv"
        retriever = TranslationRetriever(
            train_path, min_accept_score=args.tr_threshold
        )
        print(f"  Indexed {len(retriever.src_raw)} training pairs")

        # Run TR-TRY on all texts
        print("  Retrieving translations...")
        tr_try_results = []
        tr_try_scores = []
        for raw_text in texts:
            raw_str = str(raw_text) if raw_text is not None else ""
            trans, score = retriever.retrieve(raw_str)
            tr_try_results.append(trans)
            tr_try_scores.append(score)

        matched = sum(1 for s in tr_try_scores if s > args.tr_threshold)
        print(f"  TR-TRY matched: {matched}/{len(texts)} "
              f"(threshold={args.tr_threshold})")

    if args.tr_try_only:
        # TR-TRY only mode
        postprocessor = AkkadianPostprocessor(remove_brackets=True, remove_big_gap=True)

        # Use dictionary for 1-token, TR-TRY for rest
        final_tr = []
        for i, raw_text in enumerate(texts):
            raw_str = str(raw_text) if raw_text is not None else ""

            short = postprocessor.get_short_input_translation(raw_str)
            if short is not None:
                final_tr.append(short)
            elif postprocessor.is_broken_text(raw_str):
                final_tr.append("...")
            elif tr_try_results[i]:
                final_tr.append(tr_try_results[i])
            else:
                final_tr.append("")

        print(f"\n[TR-TRY Only] Evaluating...")
        scores_tr = evaluate(final_tr, references)
        print(format_report(scores_tr))

        elapsed = (time.time() - start_time) / 60
        print(f"  Total time: {elapsed:.1f} minutes")

        results_path = PROJECT_ROOT / "data" / "baseline_results.csv"
        pd.DataFrame({
            "transliteration": texts,
            "reference": references,
            "tr_try": final_tr,
        }).to_csv(results_path, index=False, encoding="utf-8")
        print(f"  Results saved: {results_path}")
        return

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

    # --- Pattern D/E/F: TR-TRY hybrid ---
    scores_hybrid = {}
    hybrid_results = {}
    if tr_try_results is not None:
        # Pattern D: TR-TRY standalone (with dict for 1-token)
        print("\n  Pattern D: TR-TRY standalone")
        final_d = []
        for i, raw_text in enumerate(texts):
            raw_str = str(raw_text) if raw_text is not None else ""
            short = postprocessor.get_short_input_translation(raw_str)
            if short is not None:
                final_d.append(short)
            elif postprocessor.is_broken_text(raw_str):
                final_d.append("...")
            elif tr_try_results[i]:
                final_d.append(tr_try_results[i])
            else:
                final_d.append("")
        scores_d = evaluate(final_d, references)
        print(format_report(scores_d))
        scores_hybrid["D: TR-TRY"] = scores_d
        hybrid_results["tr_try"] = final_d

        # Pattern E: Hybrid - use TR-TRY when score > high_threshold,
        # else use ByT5
        print("\n  Pattern E: Hybrid (TR-TRY if high score, else ByT5)")
        for high_thresh in [0.3, 0.5, 0.7]:
            final_e = []
            tr_used = 0
            for i, raw_text in enumerate(texts):
                if tr_try_scores[i] > high_thresh and tr_try_results[i]:
                    final_e.append(tr_try_results[i])
                    tr_used += 1
                else:
                    final_e.append(top1_results[i])
            scores_e = evaluate(final_e, references)
            label = f"E: Hybrid(>{high_thresh})"
            print(f"    {label}: BLEU={scores_e['bleu']:.2f} "
                  f"chrF++={scores_e['chrf']:.2f} "
                  f"GeoMean={scores_e['geo_mean']:.2f} "
                  f"(TR used: {tr_used})")
            scores_hybrid[label] = scores_e

        # Pattern F: SentenceAlign - for long inputs, append TR-TRY to ByT5
        print("\n  Pattern F: SentenceAlign (append TR-TRY for long inputs)")
        for long_thresh in [128, 256, 512]:
            final_f = []
            appended = 0
            for i, raw_text in enumerate(texts):
                raw_str = str(raw_text) if raw_text is not None else ""
                byt5_out = top1_results[i]
                tr_out = tr_try_results[i] if tr_try_results[i] else ""

                if (len(raw_str) >= long_thresh and tr_out
                        and len(tr_out) > len(byt5_out)):
                    # Find word boundary in TR-TRY near ByT5 length
                    byt5_len = min(len(byt5_out), 350)
                    spaces = [m.start() for m in re.finditer(r" ", tr_out)]
                    if spaces:
                        cut = min(spaces, key=lambda x: abs(byt5_len - x))
                        combined = byt5_out[:350] + tr_out[cut:]
                    else:
                        combined = byt5_out
                    final_f.append(combined)
                    appended += 1
                else:
                    final_f.append(byt5_out)

            scores_f = evaluate(final_f, references)
            label = f"F: SA(>={long_thresh}ch)"
            print(f"    {label}: BLEU={scores_f['bleu']:.2f} "
                  f"chrF++={scores_f['chrf']:.2f} "
                  f"GeoMean={scores_f['geo_mean']:.2f} "
                  f"(appended: {appended})")
            scores_hybrid[label] = scores_f

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
    for label, scores in scores_hybrid.items():
        print(f"  {label:<25s} {scores['bleu']:>8.2f} {scores['chrf']:>8.2f} {scores['geo_mean']:>8.2f}")
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
    if tr_try_results is not None:
        result_df["tr_try"] = final_d
    if not args.skip_mbr:
        result_df["mbr_chrf"] = mbr_chrf_results
        result_df["mbr_geo"] = mbr_geo_results
    result_df.to_csv(results_path, index=False, encoding="utf-8")
    print(f"  Results saved: {results_path}")


if __name__ == "__main__":
    main()
