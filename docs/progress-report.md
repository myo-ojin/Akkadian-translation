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

## Dataset Analysis

### Val Set (454 evaluable samples)
- 191 (42%) are 1-token inputs from 50 unique documents
- 263 (58%) are multi-token inputs
- 14 samples have "..." reference (broken texts)

### Test Set (4 samples)
- All multi-token (16-34 tokens)
- All from ONE document (text_id: 332fda50)
- Genre: Kanesh trading colony administrative letter (Old Assyrian)

## Files Modified

| File | Changes |
|------|---------|
| `preprocessing.py` | Added `AkkadianPostprocessor` methods: `fix_zero_fraction`, `_remove_ngram_loops`, `truncate_to_complete_sentence`, `cap_output_length`, `get_short_input_translation`, `is_broken_text` |
| `scripts/run_baseline.py` | Added `repetition_penalty`, `no_repeat_ngram_size` to CFG. Updated `run_inference()` with short input handler, broken text detection, length capping |
| `scripts/evaluate.py` | Evaluation utilities (BLEU, chrF++, GeoMean) |

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

## Next Steps (Under Investigation)

- LoRA fine-tuning on Kanesh colony / Old Assyrian texts
- MBR with mixed BLEU+chrF++ utility
- Diverse Beam Search (num_beam_groups)
- Output length cap optimization for test set
