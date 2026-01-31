# string_matcher.py - 文字列一致度フィルタモジュール

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from difflib import SequenceMatcher


class StringMatchFilter:
    """
    Beam searchで生成された複数の翻訳候補から最適なものを選択
    
    評価軸：
    1. Model Confidence (重み: 0.40)
       - SequenceScores（モデルの出力確率）
    
    2. Semantic Consistency (重み: 0.30)
       - ラウンドトリップ類似度 or 複数候補間一貫性
    
    3. Fluency Score (重み: 0.20)
       - 英語言語モデルの困惑度（オプション）
    
    4. Length Reasonableness (重み: 0.10)
       - 入力長と出力長の対応度
    """
    
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 language_model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            weights: スコア重み {名前: 重み値}
                   デフォルト: {'confidence': 0.4, 'consistency': 0.3, 
                             'fluency': 0.2, 'length': 0.1}
            language_model_path: 流暢性スコア計算用の言語モデルパス（オプション）
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        
        # デフォルト重み
        self.weights = weights or {
            'confidence': 0.40,
            'consistency': 0.30,
            'fluency': 0.20,
            'length': 0.10
        }
        
        # 重みの正規化
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # 言語モデル（流暢性スコア用）
        self.lm = None
        self.lm_tokenizer = None
        if language_model_path:
            self._load_language_model(language_model_path)
    
    def _load_language_model(self, model_path: str):
        """言語モデルをロード"""
        try:
            self.lm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            from transformers import AutoModelForCausalLM
            self.lm = AutoModelForCausalLM.from_pretrained(model_path)
            self.lm = self.lm.to(self.device).eval()
        except Exception as e:
            print(f"Warning: Could not load language model: {e}")
            self.lm = None
    
    def levenshtein_similarity(self, s1: str, s2: str) -> float:
        """
        正規化されたLevenshtein距離（類似度）
        
        Args:
            s1, s2: 比較文字列
            
        Returns:
            [0.0, 1.0] - 1.0は完全一致
        """
        seq_matcher = SequenceMatcher(None, s1.lower(), s2.lower())
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
    
    def combined_similarity(self, s1: str, s2: str) -> float:
        """
        複合類似度スコア
        
        Args:
            s1, s2: 比較文字列
            
        Returns:
            [0.0, 1.0] - 統合スコア
        """
        lev_score = self.levenshtein_similarity(s1, s2)
        f1_score = self.token_f1_similarity(s1, s2)
        
        # 加重平均（Levenshteinを主軸に）
        combined = 0.7 * lev_score + 0.3 * f1_score
        
        return combined
    
    def inter_candidate_consistency(self, candidates: List[str]) -> Dict[str, float]:
        """
        複数候補間の一貫性スコア計算
        複数の候補が似ている = モデルの確信度が高い
        
        Args:
            candidates: 翻訳候補リスト（通常8候補）
            
        Returns:
            {
                'consistency_score': float,  # [0.0, 1.0]
                'avg_similarity': float,     # 平均類似度
                'voting_score': float,       # 投票による信頼度
                'best_pair': Tuple[int, int], # 最も似た候補ペア
                'best_pair_sim': float        # その類似度
            }
        """
        if len(candidates) < 2:
            return {
                'consistency_score': 0.5,
                'avg_similarity': 0.5,
                'voting_score': 0.5,
                'best_pair': (0, 0),
                'best_pair_sim': 1.0
            }
        
        # 全ペアの類似度を計算
        pairwise_sims = []
        best_sim = 0.0
        best_pair = (0, 0)
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                sim = self.combined_similarity(candidates[i], candidates[j])
                pairwise_sims.append(sim)
                
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (i, j)
        
        # 平均類似度
        avg_similarity = np.mean(pairwise_sims) if pairwise_sims else 0.5
        
        # 投票スコア: 各候補の重複トークン数
        token_lists = [set(c.lower().split()) for c in candidates]
        all_tokens = set()
        for tokens in token_lists:
            all_tokens.update(tokens)
        
        if not all_tokens:
            voting_score = 0.5
        else:
            # 複数候補に出現するトークンの比率
            token_counts = {}
            for tokens in token_lists:
                for token in tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1
            
            shared_tokens = sum(1 for count in token_counts.values() if count > 1)
            voting_score = shared_tokens / len(all_tokens) if all_tokens else 0.5
        
        # 統合スコア: 平均類似度と投票スコアの加重平均
        consistency_score = 0.6 * avg_similarity + 0.4 * voting_score
        
        return {
            'consistency_score': min(1.0, consistency_score),
            'avg_similarity': avg_similarity,
            'voting_score': voting_score,
            'best_pair': best_pair,
            'best_pair_sim': best_sim
        }
    
    def confidence_score(self, sequence_score: float) -> float:
        """
        モデルの出力確率をスコアに変換
        
        Args:
            sequence_score: SequenceScore（負の値、通常-10～0）
            
        Returns:
            [0.0, 1.0] の正規化スコア
        """
        # exponentialスケーリング
        try:
            score = np.exp(sequence_score)
            # [0, 1]に正規化（sequence_score=-10なら≈0, 0なら=1）
            normalized = max(0.0, min(1.0, score))
            return normalized
        except:
            return 0.5
    
    def fluency_score(self, text: str) -> float:
        """
        英語の流暢性スコア（言語モデルの困惑度）
        
        Args:
            text: 評価対象テキスト
            
        Returns:
            [0.0, 1.0] スコア（高いほどより自然）
        """
        if self.lm is None:
            # 言語モデルが利用不可の場合、長さベースのシンプルなヒューリスティック
            # 単語数が5～20の範囲が好ましい
            word_count = len(text.split())
            if 5 <= word_count <= 20:
                return 1.0
            elif 3 <= word_count <= 25:
                return 0.8
            else:
                return 0.5
        
        try:
            inputs = self.lm_tokenizer(text, return_tensors='pt').to(self.device)
            
            with torch.inference_mode():
                outputs = self.lm(**inputs, labels=inputs.input_ids)
                loss = outputs.loss.item()
            
            # lossを[0,1]スコアに変換
            # loss < 5: 高品質, loss > 10: 低品質
            fluency = 1.0 / (1.0 + loss / 5.0)
            return min(1.0, fluency)
        except:
            return 0.5
    
    def length_reasonableness(self, 
                             input_tokens: int,
                             output_tokens: int) -> float:
        """
        入力・出力の長さの対応度
        
        Args:
            input_tokens: 入力トークン数
            output_tokens: 出力トークン数
            
        Returns:
            [0.0, 1.0] スコア
        """
        # 期待される出力長は入力長と同程度～1.5倍
        ratio = output_tokens / max(1, input_tokens)
        
        if 0.8 <= ratio <= 1.5:
            return 1.0
        elif 0.5 <= ratio <= 2.0:
            return 0.8
        else:
            return 0.5
    
    def score_candidate(self,
                       candidate: str,
                       input_text: str,
                       sequence_score: float = None,
                       consistency_score: float = None) -> Dict[str, float]:
        """
        候補を多次元で評価
        
        Args:
            candidate: 翻訳候補
            input_text: 入力テキスト（アッカド語）
            sequence_score: モデルのSequenceScore（オプション）
            consistency_score: ラウンドトリップスコア or 候補間一貫性スコア（オプション）
            
        Returns:
            {
                'confidence': float,
                'consistency': float,
                'fluency': float,
                'length': float,
                'final_score': float
            }
        """
        scores = {}
        
        # Confidence
        scores['confidence'] = (
            self.confidence_score(sequence_score) 
            if sequence_score is not None 
            else 0.5
        )
        
        # Consistency（デフォルトは中立値）
        scores['consistency'] = consistency_score if consistency_score is not None else 0.5
        
        # Fluency
        scores['fluency'] = self.fluency_score(candidate)
        
        # Length
        input_len = len(input_text.split())
        output_len = len(candidate.split())
        scores['length'] = self.length_reasonableness(input_len, output_len)
        
        # Final weighted score
        scores['final_score'] = (
            self.weights['confidence'] * scores['confidence'] +
            self.weights['consistency'] * scores['consistency'] +
            self.weights['fluency'] * scores['fluency'] +
            self.weights['length'] * scores['length']
        )
        
        return scores
    
    def select_best(self,
                   candidates: List[str],
                   input_text: str,
                   sequence_scores: Optional[List[float]] = None,
                   consistency_scores: Optional[List[float]] = None) -> Tuple[str, float, Dict]:
        """
        複数候補から最適なものを選択
        
        Args:
            candidates: 翻訳候補リスト
            input_text: 入力テキスト
            sequence_scores: 各候補のSequenceScore（オプション）
            consistency_scores: 各候補の一貫性スコア（オプション）
            
        Returns:
            (best_candidate, score, scoring_details)
        """
        if not candidates:
            return "", 0.0, {}
        
        sequence_scores = sequence_scores or [None] * len(candidates)
        consistency_scores = consistency_scores or [None] * len(candidates)
        
        scored_candidates = []
        for i, (cand, seq_score, cons_score) in enumerate(
            zip(candidates, sequence_scores, consistency_scores)
        ):
            scores = self.score_candidate(
                cand, input_text, seq_score, cons_score
            )
            scored_candidates.append((i, cand, scores))
        
        # 最高スコアを選択
        best_idx, best_cand, best_scores = max(
            scored_candidates, key=lambda x: x[2]['final_score']
        )
        
        return best_cand, best_scores['final_score'], {
            'best_idx': best_idx,
            'scores': best_scores,
            'all_candidates_scores': [x[2] for x in scored_candidates]
        }
    
    def select_best_with_confidence_filter(self,
                                          candidates: List[str],
                                          input_text: str,
                                          sequence_scores: Optional[List[float]] = None,
                                          confidence_threshold: float = 0.5) -> Tuple[str, float, Dict]:
        """
        低信頼度フィルタリングを適用した最適候補選択
        
        低信頼度候補（confidence < threshold）に対してのみ
        複数候補間の一貫性チェックを追加適用
        
        Args:
            candidates: 翻訳候補リスト
            input_text: 入力テキスト
            sequence_scores: 各候補のSequenceScore
            confidence_threshold: 信頼度の閾値（デフォルト: 0.5）
            
        Returns:
            (best_candidate, final_score, details)
        """
        if not candidates:
            return "", 0.0, {}
        
        sequence_scores = sequence_scores or [None] * len(candidates)
        
        # ステップ1: 各候補の信頼度スコアを計算
        confidence_scores = [
            self.confidence_score(score) if score is not None else 0.5
            for score in sequence_scores
        ]
        
        # ステップ2: 高信頼度候補と低信頼度候補を分離
        high_confidence_indices = [
            i for i, conf in enumerate(confidence_scores) if conf >= confidence_threshold
        ]
        low_confidence_indices = [
            i for i, conf in enumerate(confidence_scores) if conf < confidence_threshold
        ]
        
        # ステップ3: 信頼度スコアを適用
        consistency_scores = [0.5] * len(candidates)  # デフォルト
        
        # 高信頼度候補: 即採用（一貫性スコア = 1.0）
        for idx in high_confidence_indices:
            consistency_scores[idx] = 1.0
        
        # 低信頼度候補: 複数候補間一貫性をチェック
        if low_confidence_indices:
            low_conf_candidates = [candidates[i] for i in low_confidence_indices]
            consistency_result = self.inter_candidate_consistency(low_conf_candidates)
            
            # 複数候補間の一貫性スコアを反映
            inter_consistency = consistency_result['consistency_score']
            
            for idx in low_confidence_indices:
                # 信頼度と複数候補間一貫性を加重（30:70）
                consistency_scores[idx] = (
                    0.3 * confidence_scores[idx] + 
                    0.7 * inter_consistency
                )
        
        # ステップ4: 統合スコアで最適候補を選択
        scored_candidates = []
        for i, cand in enumerate(candidates):
            scores = self.score_candidate(
                cand, input_text, sequence_scores[i], consistency_scores[i]
            )
            scored_candidates.append((i, cand, scores))
        
        # 最高スコアを選択
        best_idx, best_cand, best_scores = max(
            scored_candidates, key=lambda x: x[2]['final_score']
        )
        
        return best_cand, best_scores['final_score'], {
            'best_idx': best_idx,
            'confidence_threshold': confidence_threshold,
            'high_confidence_count': len(high_confidence_indices),
            'low_confidence_count': len(low_confidence_indices),
            'scores': best_scores,
            'all_candidates_scores': [x[2] for x in scored_candidates],
            'confidence_scores': confidence_scores,
            'consistency_scores': consistency_scores
        }
    
    def ensemble_candidates(self,
                           candidates: List[str],
                           scores: List[float],
                           strategy: str = 'weighted_choice') -> str:
        """
        複数候補の統合（投票・加重平均）
        
        Args:
            candidates: 翻訳候補リスト
            scores: 各候補のスコア
            strategy: 'weighted_choice'（確率的選択）/ 'top_k_voting'（投票）
            
        Returns:
            統合結果テキスト
        """
        if not candidates:
            return ""
        
        if strategy == 'weighted_choice':
            # スコアに基づいた確率的選択
            probabilities = np.array(scores) / sum(scores)
            selected_idx = np.random.choice(len(candidates), p=probabilities)
            return candidates[selected_idx]
        
        elif strategy == 'top_k_voting':
            # トップ3の多数決投票
            top_k = min(3, len(candidates))
            top_indices = np.argsort(scores)[-top_k:]
            top_candidates = [candidates[i] for i in top_indices]
            
            # 最も多く出現する単語を投票
            from collections import Counter
            all_words = ' '.join(top_candidates).split()
            most_common_words = [w for w, _ in Counter(all_words).most_common(5)]
            
            return ' '.join(most_common_words) if most_common_words else candidates[0]
        
        else:
            # デフォルト: 最高スコア
            return candidates[np.argmax(scores)]


# 使用例
if __name__ == "__main__":
    filter = StringMatchFilter(device='cpu')
    
    # テスト1: 複数候補間の一貫性スコア
    candidates = [
        "Seal of Mannum-balum-Aššur son of Ṣill-Adad",
        "Seal of Mannum-balum-Aššur son of Ṣill-Adad",  # 同一
        "Seal of Mannum son of Adad",                   # 類似
        "The seal of Mannum"                            # 異なる
    ]
    
    consistency = filter.inter_candidate_consistency(candidates)
    print(f"\n=== Inter-Candidate Consistency ===")
    print(f"Consistency Score: {consistency['consistency_score']:.3f}")
    print(f"Avg Similarity: {consistency['avg_similarity']:.3f}")
    print(f"Voting Score: {consistency['voting_score']:.3f}")
    
    # テスト2: 低信頼度フィルタリング
    input_text = "KIŠIB ma-nu-ba-lúm-a-šur"
    sequence_scores = [-5.2, -3.1, -4.5, -8.0]  # 低信頼度2つ
    
    best, final_score, details = filter.select_best_with_confidence_filter(
        candidates, input_text, sequence_scores, confidence_threshold=0.5
    )
    
    print(f"\n=== Low Confidence Filtering ===")
    print(f"Best: {best}")
    print(f"Final Score: {final_score:.3f}")
    print(f"High Confidence Count: {details['high_confidence_count']}")
    print(f"Low Confidence Count: {details['low_confidence_count']}")
