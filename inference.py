# inference.py - 統合推論パイプライン

import torch
import pandas as pd
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from string_matcher import StringMatchFilter


class AkkadianTranslator:
    """
    統合推論パイプライン
    
    フロー：
    1. Beam searchで複数候補生成
    2. 複数候補間の一貫性をチェック
    3. 低信頼度フィルタリングを適用
    4. 最適な翻訳候補を選択
    """
    
    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_length: int = 496,
                 num_beams: int = 8,
                 confidence_threshold: float = 0.5):
        """
        Args:
            model_path: ファインチューン済みByT5モデルのパス
            tokenizer_path: トークナイザーのパス
            device: 'cuda' or 'cpu'
            max_length: 最大トークン長
            num_beams: Beam search数
            confidence_threshold: 低信頼度判定の閾値
        """
        self.device = torch.device(device)
        self.max_length = max_length
        self.num_beams = num_beams
        self.confidence_threshold = confidence_threshold
        
        # モデルとトークナイザーをロード
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model = self.model.to(self.device).eval()
        
        # 文字列フィルタを初期化
        self.filter = StringMatchFilter(device=device)
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Device: {device}")
        print(f"✓ Beam search: {num_beams}")
        print(f"✓ Confidence threshold: {confidence_threshold}")
    
    def translate_single(self, akkadian_text: str) -> Dict:
        """
        単一テキストの翻訳
        
        Args:
            akkadian_text: アッカド語テキスト
            
        Returns:
            {
                'translation': str,          # 最終翻訳
                'confidence': float,         # 信頼度スコア
                'candidates': List[str],     # 全候補（top 3）
                'high_confidence': bool,     # 高信頼度フラグ
                'method': str,               # 選択方法
                'details': Dict              # 詳細スコア情報
            }
        """
        input_prompt = f"translate Akkadian to English: {akkadian_text}"
        
        # トークン化
        inputs = self.tokenizer(
            input_prompt,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Beam searchで複数候補生成
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,  # 全候補を返す
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                max_new_tokens=self.max_length
            )
        
        # デコード
        candidates = self.tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        
        # SequenceScoresの取得（複数候補の場合は複数スコア）
        sequence_scores = []
        if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
            sequence_scores = outputs.sequences_scores.cpu().numpy().tolist()
        else:
            sequence_scores = [None] * len(candidates)
        
        # 低信頼度フィルタリングを適用
        best_candidate, final_score, filter_details = (
            self.filter.select_best_with_confidence_filter(
                candidates=candidates,
                input_text=akkadian_text,
                sequence_scores=sequence_scores,
                confidence_threshold=self.confidence_threshold
            )
        )
        
        # 信頼度を判定
        high_confidence = (
            filter_details['high_confidence_count'] > 0 or
            final_score >= 0.7
        )
        
        return {
            'translation': best_candidate,
            'confidence': final_score,
            'candidates': candidates[:3],  # Top 3表示
            'high_confidence': high_confidence,
            'method': 'confidence_filtering' if not high_confidence else 'high_confidence',
            'details': filter_details
        }
    
    def translate_batch(self, akkadian_texts: List[str], 
                       batch_size: int = 4,
                       verbose: bool = False) -> List[Dict]:
        """
        複数テキストの翻訳（バッチ処理）
        
        Args:
            akkadian_texts: アッカド語テキストのリスト
            batch_size: バッチサイズ
            verbose: 進捗表示
            
        Returns:
            翻訳結果のリスト
        """
        results = []
        
        for i, text in enumerate(akkadian_texts):
            if verbose and i % batch_size == 0:
                print(f"Processing: {i+1}/{len(akkadian_texts)}")
            
            result = self.translate_single(text)
            results.append(result)
        
        return results
    
    def translate_from_csv(self, csv_path: str, 
                          input_column: str = 'transliteration',
                          batch_size: int = 4) -> pd.DataFrame:
        """
        CSVファイルから翻訳
        
        Args:
            csv_path: 入力CSVパス
            input_column: アッカド語を含む列名
            batch_size: バッチサイズ
            
        Returns:
            結果DataFrameの作成フロー: id, translation, confidence, high_confidence
        """
        df = pd.read_csv(csv_path)
        
        akkadian_texts = df[input_column].fillna('').tolist()
        results = self.translate_batch(akkadian_texts, batch_size, verbose=True)
        
        # 結果をDataFrameに変換
        result_df = pd.DataFrame({
            'id': range(len(results)),
            'translation': [r['translation'] for r in results],
            'confidence': [r['confidence'] for r in results],
            'high_confidence': [r['high_confidence'] for r in results],
            'method': [r['method'] for r in results]
        })
        
        return result_df


# 使用例
if __name__ == "__main__":
    # モデル設定
    translator = AkkadianTranslator(
        model_path="/path/to/byt5-model",
        tokenizer_path="/path/to/tokenizer",
        device='cuda',
        num_beams=8,
        confidence_threshold=0.5
    )
    
    # 単一翻訳テスト
    akkadian = "KIŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá"
    result = translator.translate_single(akkadian)
    
    print(f"\nInput: {akkadian}")
    print(f"Translation: {result['translation']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"High Confidence: {result['high_confidence']}")
    print(f"Method: {result['method']}")
    
    # バッチ翻訳テスト
    # results = translator.translate_from_csv("test.csv")
    # results.to_csv("output.csv", index=False)
