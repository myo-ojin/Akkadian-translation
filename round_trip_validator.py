# round_trip_validator.py - ラウンドトリップ検証モジュール

import re
import torch
from typing import Tuple, Dict
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RoundTripValidator:
    """
    アッカド語翻訳の逆翻訳で一貫性を検証
    
    フロー：
    1. 入力アッカド語 → モデル → 英語翻訳候補
    2. 英語候補 → 逆翻訳モデル → 再生成アッカド語
    3. 元のアッカド語と再生成言語の類似度を計算
    4. 類似度スコア = 信頼度スコア
    """
    
    def __init__(self, 
                 reverse_model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_length: int = 496):
        """
        Args:
            reverse_model_path: 英→アッカド翻訳モデルのパス
            device: 'cuda' or 'cpu'
            max_length: トークン最大長
        """
        self.device = torch.device(device)
        self.max_length = max_length
        
        # 逆翻訳モデルのロード
        self.tokenizer = AutoTokenizer.from_pretrained(reverse_model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(reverse_model_path)
        self.model = self.model.to(self.device).eval()
    
    def reverse_translate(self, english_text: str) -> str:
        """
        英語テキストをアッカド語に逆翻訳
        
        Args:
            english_text: 英語テキスト
            
        Returns:
            再生成されたアッカド語テキスト
        """
        input_prompt = f"translate English to Akkadian: {english_text}"
        
        # トークン化
        inputs = self.tokenizer(
            input_prompt,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # 推論
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=self.max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # デコード
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()
    
    def levenshtein_similarity(self, s1: str, s2: str) -> float:
        """
        正規化されたLevenshtein距離（類似度）
        
        Args:
            s1, s2: 比較文字列
            
        Returns:
            [0.0, 1.0] - 1.0は完全一致
        """
        # ギャップマークを無視
        s1_clean = re.sub(r'<[^>]+>', '', s1).strip()
        s2_clean = re.sub(r'<[^>]+>', '', s2).strip()
        
        # Levenshtein距離計算
        longer = max(len(s1_clean), len(s2_clean))
        if longer == 0:
            return 1.0
        
        # 簡易実装（本来はpython-Levenshteinライブラリを使用）
        seq_matcher = SequenceMatcher(None, s1_clean, s2_clean)
        return seq_matcher.ratio()
    
    def token_f1_similarity(self, s1: str, s2: str) -> float:
        """
        トークン（スペース分割単語）レベルのF1スコア
        
        Args:
            s1, s2: 比較文字列
            
        Returns:
            [0.0, 1.0] F1スコア
        """
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        
        intersection = tokens1 & tokens2
        
        if not tokens1 or not tokens2:
            return 0.0
        
        precision = len(intersection) / len(tokens2)
        recall = len(intersection) / len(tokens1)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def combined_similarity(self, original: str, roundtrip: str) -> float:
        """
        複合類似度スコア
        
        Args:
            original: 元のアッカド語テキスト
            roundtrip: ラウンドトリップ後のアッカド語テキスト
            
        Returns:
            [0.0, 1.0] - 統合スコア
        """
        lev_score = self.levenshtein_similarity(original, roundtrip)
        f1_score = self.token_f1_similarity(original, roundtrip)
        
        # 加重平均（Levenshteinを主軸に）
        combined = 0.7 * lev_score + 0.3 * f1_score
        
        return combined
    
    def validate(self, 
                 original_akkadian: str,
                 english_translation: str) -> Dict[str, float]:
        """
        翻訳候補を検証
        
        Args:
            original_akkadian: 入力アッカド語テキスト
            english_translation: 生成された英語翻訳
            
        Returns:
            {
                'score': float,           # 最終スコア [0.0, 1.0]
                'roundtrip_text': str,    # 逆翻訳テキスト
                'lev_similarity': float,  # Levenshtein類似度
                'f1_similarity': float,   # F1スコア
                'is_valid': bool          # スコア >= 0.3
            }
        """
        # 逆翻訳
        roundtrip_akkadian = self.reverse_translate(english_translation)
        
        # 類似度計算
        lev = self.levenshtein_similarity(original_akkadian, roundtrip_akkadian)
        f1 = self.token_f1_similarity(original_akkadian, roundtrip_akkadian)
        combined = self.combined_similarity(original_akkadian, roundtrip_akkadian)
        
        return {
            'score': combined,
            'roundtrip_text': roundtrip_akkadian,
            'lev_similarity': lev,
            'f1_similarity': f1,
            'is_valid': combined >= 0.3
        }
    
    def validate_batch(self, 
                      original_texts: list,
                      english_candidates: list) -> list:
        """
        バッチ検証
        
        Args:
            original_texts: 元のアッカド語リスト
            english_candidates: 英語翻訳候補リスト
            
        Returns:
            検証結果リスト
        """
        results = []
        for orig, eng in zip(original_texts, english_candidates):
            result = self.validate(orig, eng)
            results.append(result)
        return results


# 使用例
if __name__ == "__main__":
    # モデルパスを指定（実際のパスに変更）
    validator = RoundTripValidator(
        reverse_model_path="/path/to/reverse/model",
        device='cuda'
    )
    
    # テスト
    original = "KIŠIB ma-nu-ba-lúm-a-šur"
    english = "Seal of Mannum-balum-Aššur"
    
    result = validator.validate(original, english)
    print(f"Score: {result['score']:.3f}")
    print(f"Valid: {result['is_valid']}")
    print(f"Roundtrip: {result['roundtrip_text']}")
