# Experiment Log

## Competition Info
- **Competition**: Deep Past Initiative - Machine Translation (Akkadian -> English)
- **Metric**: GeoMean = sqrt(BLEU * chrF++)
- **Deadline**: 2026-03-23
- **Test Set**: 4 samples, all from text_id `332fda50` (Old Assyrian commercial letter)
- **Kaggle**: Code Competition (notebook submission only)

---

## LB Scores

| Date | Notebook | Version | Score | Notes |
|------|----------|---------|-------|-------|
| 2026-02-17 | ensemble | v9 (pre-fix) | 22.9 | Non-ASCII issue, PN normalization enabled |
| 2026-02-17 | TR-TRY | v9 (ASCII fix) | 6.8 | IDs 1,2,3 return identical translation |
| 2026-02-17 | debug | v2 | 0.0 | Copies sample_submission.csv (baseline) |
| 2026-02-17 | ensemble | v12 (ASCII fix) | 24.0 | PN normalization disabled, sanitize_ascii() |

---

## Key Fixes Applied (2026-02-17)

### Non-ASCII Submission Error
- **Problem**: Submission.csv with non-ASCII chars (e.g. `š` U+0161) causes "Submission Scoring Error"
- **Root cause**: `normalize_proper_nouns()` maps ASCII names to diacritical forms (e.g. `Aba-ahu` -> `Abā-aḫu`)
- **Fix**: Added `sanitize_ascii()` function + disabled PN normalization in both notebooks
- **Files**: `tr_try_standalone.ipynb` (postprocess, save cells), `byt5_cpu_ensemble.ipynb` (postprocess, hybrid, save cells)

### Empty Translation Error
- **Problem**: `_SHORT_INPUT_MAP` had empty string values (`"lá": ""`, `"x": ""`, `"…": ""`)
- **Fix**: Changed to non-empty defaults (`"not"`, `"broken"`, `"broken"`)

---

## Approaches Tested & Rejected

| Approach | Result | Why |
|----------|--------|-----|
| MBR decoding | 73.5% worse | Length bias, MBR outputs longer than Top-1 |
| repetition_penalty / no_repeat_ngram_size | Catastrophic | ByT5 operates at byte level, penalties destroy output |
| Diverse Beam Search | Consistently worse | Standard beam outperforms |
| Back-translation | No improvement | Tested in earlier session |
| Cross-lingual transfer learning | BLEU < 8% | UC Berkeley experiment, far below ByT5 baseline |

---

## Data Sources

| Source | Rows | Type | Used In |
|--------|------|------|---------|
| `train.csv` | 1,561 | transliteration + translation | TR-TRY, Translation Memory |
| ORACC (Kaggle dataset) | 2,117 | transliteration + translation | TR-TRY, Translation Memory |
| `Sentences_Oare_FirstWord_LinNum.csv` | 9,771 | sentence-level translations (NO transliteration) | **NOT YET USED** |
| `published_texts.csv` | 7,702 | full-text transliterations (AICC_translation = URL only) | **NOT YET USED** |
| `publications.csv` | 216K pages | OCR'd PDF text (scholarly publications) | Not used |
| `OA_Lexicon_eBL.csv` | 4,014 PNs | Proper noun mappings | PN normalization (disabled) |

### Sentences_Oare + published_texts = Untapped Gold Mine
- `Sentences_Oare` has `text_uuid` that matches `published_texts.oare_id` (1,417 overlap)
- Join gives **8,484 rows** with full-text transliteration + sentence-level translation
- **8,565 sentences from texts NOT in train.csv** (from 1,447 new texts)
- Challenge: need to extract sentence-level transliteration from full-text using `sentence_obj_in_text` and `first_word_spelling`
- Would increase TR-TRY index from 3,673 to ~12,000+ pairs

---

## Architecture

### Ensemble (byt5_cpu_ensemble.ipynb)
- 5-model ByT5 weight-blended ensemble (580M params)
- Models: jeanjean111, llkh0a, qifeihhh666, assiaben, manwithacat (RAG)
- Tier 1: ByT5 multi-beam consensus + TR-TRY cross-validation
- Tier 2: Flan-T5 Philologist refinement (247M params)
- TR-TRY retrieval fallback
- Translation memory exact match (3,659 entries)
- ORACC corpus (2,117 pairs)

### TR-TRY (tr_try_standalone.ipynb)
- TF-IDF retrieval (char 3-6 ngrams + word 1-2 ngrams)
- Local val GeoMean: 40.52
- LB score: 6.8 (poor because IDs 1,2,3 all match same training example)

---

## TODO / Next Steps

1. **Extract sentence-level transliteration-translation pairs** from Sentences_Oare + published_texts
2. **Add extracted pairs to TR-TRY index** and translation memory in both notebooks
3. **Upload extracted data as Kaggle dataset** for use in notebooks
4. Consider fine-tuning or other model improvements
5. Target: GeoMean 50
