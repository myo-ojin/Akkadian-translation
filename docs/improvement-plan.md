# Akkadian Translation 改善計画（4エージェント統合版）

## 現状ベースライン（2026-02-07測定）

| Method | BLEU | chrF++ | GeoMean |
|--------|------|--------|---------|
| **Top-1** | **19.25** | **43.34** | **28.89** |
| MBR chrF++ | 18.50 | 43.30 | 28.30 |
| MBR geo_mean | 18.82 | 43.29 | 28.55 |

- 454サンプル（val_sentences.csv、翻訳参照ありのみ）
- リーダーボードトップ: 38.7、公開ノートブック: 34-36

## 最重要発見

1. **前処理の修正だけで 28.89 → 34-37 に到達可能**（Kaggle 22位のhongan氏が証言）
2. **MBRが逆効果の原因**: 長さバイアス（悪化73.5%でMBR出力がTop-1より長い）
3. **繰り返し生成が深刻**: 17%のサンプルでn-gram繰り返し、repetition_penalty未設定

---

## Phase 1: Quick Wins（28.89 → 32-34目標）

### 1.1 repetition_penalty + no_repeat_ngram_size 追加
- `run_baseline.py` と `submission_notebook.py` の `model.generate()` に追加
- `repetition_penalty=1.2, no_repeat_ngram_size=4`
- 期待効果: +2〜4 GeoMean

### 1.2 "0 fraction" パターン修正
- `preprocessing.py` の `AkkadianPostprocessor` に追加
- `re.sub(r'\b0\s+([\u00bc-\u00be\u2150-\u215f])', r'\1', text)`
- "0 ⅓" → "⅓" に修正
- 期待効果: +0.5

### 1.3 出力切断修復
- max_new_tokens=496で途中切断された出力を最後の完全文で切る
- 不完全な文末を検出して除去
- 期待効果: +0.5〜1

### 1.4 強化型繰り返し除去
- スライディングウィンドウ類似度検出で変異繰り返しも除去
- 現在のremove_repeated_phrasesは単純な連続重複のみ
- 期待効果: +1〜2

---

## Phase 2: 翻字正規化（32-34 → 34-37目標）

### 2.1 ダイアクリティクス保持
- ASCII変換せず、š, ṣ, ṭ, á, à, í, ì, ú, ù 等を保持
- s/ṣ/šやt/ṭの区別が重要（DeepPast公式指摘）

### 2.2 ギャップ処理の完全正規化
- x → <gap>, 連続x → <big_gap>
- <gap> <big_gap> → <big_gap> に統合

### 2.3 短入力ハンドリング
- 1-3トークンの断片入力（約40-50件）でハルシネーション発生
- "a-na" → "To the king, my lord: your servant Aššur." 等
- 対策: 文脈連結/テンプレート翻訳/空出力

---

## Phase 3: MBR修正 + デコーディング最適化

### 3.1 長さペナルティ付きMBR
- `MBR_score(h) = avg_utility(h, refs) - alpha * max(0, len(h)/avg_ref_len - 1)`
- 悪化ケース73.5%が長さ起因なので大幅改善期待

### 3.2 length_penalty グリッドサーチ
- 現在1.3 → [0.6, 0.8, 1.0, 1.2, 1.5] でテスト

### 3.3 Epsilon Sampling + MBR（ICLR 2025最有力手法）
- `epsilon_cutoff=0.02, num_return_sequences=16`
- beam searchより多様な候補 → MBRが効果的に機能

---

## Phase 4: データ拡張（38+目標）

### 4.1 train.csvの文アラインメント修正
- 約半分がミスアラインメント（Tomorin氏: 28.4→31.4改善）

### 4.2 ORACC並列コーパス統合
- Kaggle公開データセット（manwithacat氏）

### 4.3 固有名詞辞典（onomasticon）の活用
- DeepPast公式が公開予定

---

## 参考文献
- Epsilon Sampling: arxiv.org/abs/2305.09860 (ICLR 2025)
- MBR Metric Bias: arxiv.org/abs/2411.03524v1 (WMT 2024)
- ByT5 vs mT5: doi.org/10.1162/tacl_a_00651
- Akkadian NMT: doi.org/10.1093/pnasnexus/pgad096 (PNAS Nexus 2023)

## Kaggleディスカッション
- https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/665209
- https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/670084
- https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/670040
