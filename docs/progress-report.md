# Akkadian Translation Competition - Progress Report

## Overview

Kaggle "Deep Past Challenge: Translate Akkadian to English"
- Scoring: GeoMean = sqrt(BLEU * chrF++)
- Model: ByT5-base (580M params), 3-model weight blend
- Hardware: WSL2 + NVIDIA RTX 5060 Ti (16GB VRAM)

## Score Progress

| Phase | BLEU | chrF++ | GeoMean | Delta |
|-------|------|--------|---------|-------|
| Baseline (original) | 19.25 | 43.34 | 28.89 | - |
| Phase 1: Post-processing | 21.04 | 43.71 | 30.32 | +1.43 |
| Phase 2: Short input + Length cap | 27.36 | 43.67 | 34.56 | +4.24 |

**Total improvement: +5.67 GeoMean (+19.6%)**

## Phase 1: Post-processing Quick Wins

### Implemented
1. **Sentence truncation** (`truncate_to_complete_sentence`)
   - Cuts output at last period/comma for text >200 chars
   - Impact: +0.34 GeoMean on 50-sample test

2. **N-gram loop removal** (`_remove_ngram_loops`)
   - Detects repeating 3-8 gram patterns and removes duplicates
   - Prevents model output degeneration

3. **Zero-fraction fix** (`fix_zero_fraction`)
   - Converts "0 1/3" to "1/3" (Unicode fraction handling)

### Tested but Rejected
- **repetition_penalty=1.2**: Catastrophic (-9 GeoMean). ByT5 is byte-level, so penalty applies per-byte, not per-word.
- **no_repeat_ngram_size=4**: 4 bytes = ~1 character. Destroys output quality entirely.

**Key Learning: ByT5 operates at byte level. Never use repetition_penalty or no_repeat_ngram_size.**

## Phase 2: Data-Driven Improvements

### Implemented
1. **Short input dictionary lookup** (`get_short_input_translation`)
   - 42% of val samples (191/454) are 1-token inputs (e.g., "a-na", "IGI")
   - These cause hallucinations ("To the king, my lord: your servant...")
   - Built 22-entry lookup map for common Akkadian words
   - Impact: +0.84 GeoMean

2. **Output length capping** (`cap_output_length`)
   - Formula: `max_chars = 0.5 * input_len + 30`
   - Model over-generates by 1.82x on average
   - Truncates at word boundary
   - Impact: +4.0 GeoMean (largest single improvement)

3. **Broken text detection** (`is_broken_text`)
   - Detects heavily damaged transliterations (starts with "[..." or >40% gap tokens)
   - Returns "..." for these cases

### Tested but Rejected
- **MBR decoding (chrF++ utility)**: Worsens 73.5% of cases due to length bias in chrF++
- **MBR decoding (geo_mean utility)**: Also counterproductive
- **length_penalty=0.4 + beams=6**: Better on 50 samples (+1.52), worse on full 454 (-0.13). chrF++ dropped by 0.71.

## Phase 3: Grid Search + LoRA Fine-tuning (2026-02-08)

### Grid Search Results (247 multi-token val samples)

Tested 24 combinations of length cap coefficients x 2 beam strategies:

| Config | BLEU | chrF++ | GeoMean |
|--------|------|--------|---------|
| Standard Beam, no cap | 22.28 | 53.75 | 34.60 |
| **Standard Beam, coeff=0.4/offset=20** | **30.49** | **53.09** | **40.23** |
| Diverse Beam, best (0.4/20) | 29.65 | 52.55 | 39.47 |

**Key Findings:**
- Aggressive cap (0.4*input_len+20) improves multi-token GeoMean by +5.63
- But **degrades full 454-sample eval** (33.32 vs 34.56) due to short-input impact
- Diverse Beam Search is consistently worse than Standard Beam
- Current cap (0.5*input_len+30) remains optimal for full validation set
- Test set is all 16-34 token inputs, so aggressive cap may help on submission

### External Data: Akkademia Corpus
- Source: PNAS Nexus 2023 paper (gold-standard, expert translations)
- 50,478 parallel training pairs (Akkadian transliteration + English)
- From 5 ORACC sub-corpora: RINAP, RIAo, RIBo, SAAo, Suhu
- Format normalization: `{d}-enlil` → `(d)enlil`
- MIT license, publicly available

### LoRA Fine-tuning Pipeline
- Created `scripts/lora_finetune.py`
- LoRA config: rank=8, alpha=16, target_modules=[q, v, o], 1.66M trainable params (0.28%)
- Training mix: 47K Akkademia + 4.7K competition (3x weighted) = 51.7K total
- Local test: loss 0.74 → 0.68 (Step 800/9,692) before interruption
- **Moving to Kaggle for faster iteration** (HuggingFace Trainer with checkpointing)

## Dataset Analysis

### Val Set (454 evaluable samples)
- 191 (42%) are 1-token inputs from 50 unique documents
- 263 (58%) are multi-token inputs
- 14 samples have "..." reference (broken texts)

### Test Set (4 samples)
- All multi-token (16-34 tokens)
- All from ONE document (text_id: 332fda50)
- Genre: Kanesh trading colony administrative letter (Old Assyrian)

## Files

| File | Purpose |
|------|---------|
| `preprocessing.py` | Pre/post-processing pipeline |
| `mbr_decoder.py` | MBR decoding (unused - counterproductive) |
| `scripts/run_baseline.py` | Local evaluation (supports `--lora` flag) |
| `scripts/evaluate.py` | BLEU, chrF++, GeoMean metrics |
| `scripts/lora_finetune.py` | LoRA fine-tuning (local) |
| `scripts/grid_search_phase3.py` | Length cap + beam search grid search |
| `notebooks/kaggle_lora_training.ipynb` | Kaggle LoRA training notebook |
| `data/external/akkademia/` | Akkademia parallel corpus |

## Configuration

```python
CFG = {
    "num_beams": 12,
    "num_return_sequences": 8,
    "max_new_tokens": 496,
    "length_penalty": 1.3,
    "repetition_penalty": 1.0,    # Disabled (byte-level danger)
    "no_repeat_ngram_size": 0,     # Disabled (byte-level danger)
}
```

## Next Steps

1. LoRA fine-tuning on Kaggle (HuggingFace Trainer, checkpointing)
2. Evaluate LoRA-enhanced model on full val set
3. Test-set-specific cap tuning for submission
4. Multi-model ensemble (ByT5 + NLLB + mBART) for larger gains
