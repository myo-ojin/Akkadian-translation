# Akkadian Translation

Kaggle "[Deep Past Challenge: Translate Akkadian to English](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)" competition solution.

## Approach

**Model**: ByT5-base (580M params) with 3-model weight blending + LoRA fine-tuning

| Component | Detail |
|-----------|--------|
| Base Model | ByT5-base (byte-level T5) |
| Blending | 3 checkpoints weighted [0.99, 0.98, 0.39] |
| Fine-tuning | LoRA (rank=8, alpha=16, 1.66M trainable params) |
| External Data | Akkademia corpus (50K parallel pairs, PNAS Nexus 2023) |
| Decoding | Beam search (beams=12, LP=1.3) |
| Post-processing | Length capping, dictionary lookup, broken text detection |

## Score Progress

| Phase | BLEU | chrF++ | GeoMean |
|-------|------|--------|---------|
| Baseline | 19.25 | 43.34 | 28.89 |
| + Post-processing | 21.04 | 43.71 | 30.32 |
| + Dictionary & Length cap | 27.36 | 43.67 | 34.56 |
| + LoRA fine-tuning | TBD | TBD | TBD |

Scoring: GeoMean = sqrt(BLEU * chrF++)

## Project Structure

```
.
├── preprocessing.py          # Pre/post-processing pipeline
├── mbr_decoder.py            # MBR decoding (tested, not used)
├── scripts/
│   ├── run_baseline.py       # Evaluation (--lora flag for LoRA model)
│   ├── evaluate.py           # BLEU, chrF++, GeoMean metrics
│   ├── lora_finetune.py      # LoRA training (manual loop)
│   ├── lora_trainer.py       # LoRA training (HF Trainer)
│   ├── grid_search_phase3.py # Length cap + beam search grid search
│   ├── prepare_val.py        # Validation set preparation
│   └── split_sentences.py    # Sentence splitting utilities
├── notebooks/
│   └── kaggle_lora_training.ipynb  # Kaggle submission notebook
├── docs/
│   ├── progress-report.md    # Detailed experiment log
│   └── improvement-plan.md   # Strategy planning
├── data/
│   ├── train.csv             # Competition training data
│   ├── val_sentences.csv     # Validation split (454 samples)
│   └── external/akkademia/   # Akkademia parallel corpus
└── models/                   # Model checkpoints (not in git)
    ├── byt5-base-big-data2/
    ├── byt5-akkadian-model/
    ├── train-gap-all-2/
    └── lora_adapter/
```

## Key Findings

- **ByT5 is byte-level**: `repetition_penalty` and `no_repeat_ngram_size` operate at byte level. Using them destroys output quality.
- **42% of validation samples are single-token**: Dictionary lookup eliminates hallucinations for these.
- **Output length capping is critical**: Model over-generates by 1.82x. Cap `0.5 * input_len + 30` gives +4 GeoMean.
- **MBR decoding is counterproductive**: Worsens 73.5% of cases due to chrF++ length bias.
- **Diverse Beam Search underperforms**: Standard Beam is consistently better.

## Usage

### Evaluation

```bash
# Baseline evaluation
python scripts/run_baseline.py

# With LoRA adapter
python scripts/run_baseline.py --lora

# Quick test (50 samples)
python scripts/run_baseline.py --max-samples 50
```

### LoRA Fine-tuning

```bash
# 1 epoch (recommended first)
python scripts/lora_finetune.py --epochs 1

# Full training
python scripts/lora_finetune.py --epochs 3 --lr 3e-4 --batch-size 2 --grad-accum 8
```

### Kaggle Submission

Upload `notebooks/kaggle_lora_training.ipynb` to Kaggle with the required datasets.

## Requirements

- Python 3.12+
- PyTorch 2.10+ (CUDA)
- transformers, peft, sacrebleu, pandas

## External Data

- [Akkademia](https://github.com/gaigutherz/Akkademia) - 50,478 parallel Akkadian-English pairs (MIT license, PNAS Nexus 2023)

## Hardware

Developed on WSL2 + NVIDIA RTX 5060 Ti (16GB VRAM)
