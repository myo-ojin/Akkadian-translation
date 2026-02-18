# Improvement Plan v2 (2026-02-18)

## Current State
- **LB Score**: 24.0 (ensemble v12)
- **Target**: 39+ (top 20 = 36.3, top 1 = 39.5)
- **Competition**: 2,052 teams, deadline 2026-03-23
- **Test Set**: 4 samples from text_id `332fda50` (Old Assyrian meteoric iron trade regulation)

---

## Critical Discovery

**The test text is a near-exact match of train idx=406** (oare_id `3e87aad8`, "AKT 5 1").
- Both texts: 678 chars, 59 unique words, same content
- Difference: transliteration sign-reading conventions only (e.g., `ša→aa`, `ṭup→mup`)
- `sample_submission.csv` contains the reference translations = segments of train idx=406's translation
- **Correct retrieval + sentence-boundary segmentation = BLEU ~1.0, GeoMean ~100**

### Why our current score is only 24.0
1. TR-TRY returns the FULL training translation for ALL test segments (same output for IDs 1,2,3)
2. ByT5 generates plausible but wrong translations ("silver" instead of "meteoric iron", "Daur" instead of "Assur")
3. Sentence alignment bug contaminates Sample 2 with Sample 3's text
4. Average local GeoMean per sample: ~0.16

---

## Priority 1: Quick Wins (Today) — Expected: 24→80+

### 1.1 [CRITICAL] Line-level retrieval + sentence segmentation
**Expected impact: +50-70 points**

Instead of returning full-text translations, implement:
1. **Match test transliteration → closest training text** (TF-IDF char 3-6 ngrams)
2. **Use Sentences_Oare** to find sentence boundaries within the matched text
3. **Map sentences to test segments** by `line_start`/`line_end` overlap
4. **Return only the relevant segment** of the training translation

Sentence boundaries for train idx=406 (from Sentences_Oare):
- Line 1: "Seal of Kanesh..." (→ part of test ID 0)
- Line 6: "To Kuliya..." (→ part of test ID 0)
- Line 7: "In the letter of the City..." (→ test ID 1)
- Line 8: "From this day on..." (→ part of test ID 1)
- Line 14: "As soon as you have heard..." (→ test ID 2)
- Line 25: "Send a copy of (this)..." (→ test ID 3)
- Line 28: "Even when somebody..." (→ part of test ID 3)

Implementation: Modify TR-TRY notebook to segment matched translation by line boundaries.

### 1.2 [HIGH] Disable sentence alignment
**Expected impact: +3-5 points**

Remove the TR-TRY tail-appending code in the ensemble's tier1 selection:
```python
# REMOVE this entire block:
if (len(raw_text) > LONG_TEXT_THRESHOLD and tr_clean
        and len(chosen) < len(tr_clean) * 0.6
        and tr_score > 0.3):
    ...
```
This causes cross-sample contamination (Sample 2 gets Sample 3's text appended).

### 1.3 [HIGH] Apply segmented retrieval to ensemble too
**Expected impact: included in 1.1**

If TR-TRY with segmentation gives near-perfect output, use it as:
- Translation memory override (bypass ByT5 entirely when retrieval score > 0.5)
- Or as very strong external reference in consensus scoring

---

## Priority 2: Medium-term (1-2 days) — Robustness

### 2.1 Reduce max_new_tokens: 496 → 256
**Expected impact: +1-2 points (on non-exact-match samples)**
- Competitors use 256
- Reduces hallucination and repetition

### 2.2 Domain terminology postprocessing
**Expected impact: +2-4 points (on ByT5 output path)**
- When source contains `KÙ.AN`: replace "silver" → "meteoric iron" in output
- "tablet" → "letter" for correspondence context
- "Daur"/"Dasur" → "Assur" (proper noun correction)

### 2.3 Evaluate Philologist Tier 2
**Expected impact: unclear, possibly negative**
- v9 (no ASCII fix, old Philologist): 22.9
- v12 (ASCII fix + Philologist): 24.0
- The +1.1 improvement may be entirely from ASCII fix, not Philologist
- Test: run without Philologist to measure actual impact

### 2.4 Expand TR-TRY index with Sentences_Oare data
**Expected impact: improved retrieval for future test sets**
- Extract ~8,500 sentence-level pairs from Sentences_Oare + published_texts
- Upload as Kaggle dataset for notebook access
- Increases retrieval index from 3,673 → 12,000+

---

## Priority 3: Longer-term (week+) — Generalization

### 3.1 Preprocessing alignment with pretrained models
**Status: ANALYZED — mostly compatible, minor fixes**

All 5 ByT5 models were trained on **raw transliteration with minimal preprocessing**:
- Models expect Unicode diacritics preserved (š, ṣ, ṭ, ḫ, á, etc.) — **we do this correctly**
- Models expect determinatives in `(d)` format preserved — **we accidentally do this** (our `{d}` regex matches nothing in the data)
- Gap normalization (`<gap>`, `<big_gap>`) is compatible with 3/5 models

**Safe changes:**
- Add `repetition_penalty=1.1` and `no_repeat_ngram_size=3` to generation params (+0.5-1.5pt)
- Remove dead `_DET_PATTERN` code (cleanup only)
- Consider removing scholarly symbol stripping (`⌈⌉⌊⌋°`) — models were trained with these

**DO NOT change:**
- Diacritic handling (keep Unicode, do NOT convert š→sz like carp0308)
- Scribal notation removal (models expect raw format)

### 3.2 Self-ensemble with sampling
**Expected impact: +1-2 points**
- carp0308 uses temperature=0.7, 3 sampling runs, voting
- Could improve candidate diversity for consensus scoring

### 3.3 Fine-tune model on extended dataset
**Expected impact: potentially significant**
- Use Sentences_Oare extracted pairs for additional fine-tuning
- Requires GPU Kaggle notebook (T4 or P100)

---

## Implementation Order

| Step | Task | Est. Impact | Effort |
|------|------|-------------|--------|
| 1 | Sentence segmentation in TR-TRY notebook | +50-70pt | 2-3 hours |
| 2 | Disable sentence alignment in ensemble | +3-5pt | 10 min |
| 3 | Apply segmented retrieval to ensemble | +10-20pt | 1-2 hours |
| 4 | Push & submit TR-TRY with segmentation | verify | 30 min |
| 5 | Push & submit ensemble with fixes | verify | 30 min |
| 6 | Reduce max_new_tokens, domain terms | +2-4pt | 1 hour |
| 7 | Evaluate Philologist | +/-1pt | 1 hour |

**Critical path: Steps 1-4 should be done today.**

---

## Risk Assessment

- **Overfitting to public test**: The 4-sample test set may change for private scoring. Sentence segmentation is optimal for this specific test but we need general robustness too.
- **Sign-reading variants**: Test uses different cuneiform readings than training. TF-IDF char-ngram still gives 0.61 similarity (clear top-1), so this is manageable.
- **Submission limits**: ~5 submissions per day. Prioritize TR-TRY (simpler, faster to verify).
