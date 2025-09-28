# スコア値オブジェクト設計

## 📋 文書情報

| 項目       | 内容                     |
| ---------- | ------------------------ |
| 文書名     | スコア値オブジェクト設計 |
| バージョン | 1.0                      |
| 作成日     | 2025 年 9 月 28 日       |
| 作成者     | AI 開発チーム            |
| 承認者     | 技術責任者               |
| ステータス | 草案                     |

## 🎯 概要

スコア値オブジェクトは、1-5 スケールの評価スコアを表現する不変の値オブジェクトです。スコアの妥当性を検証し、信頼度と推論を含む包括的な評価情報を提供します。

## 🏗️ 値オブジェクト設計

### 1. 基本構造

```python
@dataclass(frozen=True)
class Score:
    """
    評価スコア値オブジェクト

    1-5スケールの評価スコアを表現。
    不変性を保証し、スコアの妥当性を検証。
    """

    value: float
    confidence: float  # 0.0-1.0
    reasoning: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not 1.0 <= self.value <= 5.0:
            raise ValueError(f"Score must be between 1.0 and 5.0, got {self.value}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if not self.reasoning.strip():
            raise ValueError("Reasoning cannot be empty")
```

### 2. 属性説明

| 属性名         | 型                       | 説明                     |
| -------------- | ------------------------ | ------------------------ |
| **value**      | float                    | 1-5 スケールの評価スコア |
| **confidence** | float                    | 0.0-1.0 の信頼度         |
| **reasoning**  | str                      | 評価理由の詳細説明       |
| **metadata**   | Optional[Dict[str, Any]] | 追加のメタデータ         |

## 🔧 ビジネスロジック

### 1. 信頼度判定

```python
def is_high_confidence(self, threshold: float = 0.8) -> bool:
    """高信頼度スコアかチェック"""
    return self.confidence >= threshold

def is_low_confidence(self, threshold: float = 0.5) -> bool:
    """低信頼度スコアかチェック"""
    return self.confidence < threshold

def get_confidence_level(self) -> str:
    """信頼度レベルを取得"""
    if self.confidence >= 0.9:
        return "Very High"
    elif self.confidence >= 0.8:
        return "High"
    elif self.confidence >= 0.6:
        return "Medium"
    elif self.confidence >= 0.4:
        return "Low"
    else:
        return "Very Low"
```

### 2. スコア分類

```python
def to_category(self) -> str:
    """スコアをカテゴリに変換"""
    if self.value >= 4.5:
        return "Excellent"
    elif self.value >= 3.5:
        return "Good"
    elif self.value >= 2.5:
        return "Fair"
    elif self.value >= 1.5:
        return "Poor"
    else:
        return "Very Poor"

def to_grade(self) -> str:
    """スコアをグレードに変換"""
    if self.value >= 4.5:
        return "A"
    elif self.value >= 3.5:
        return "B"
    elif self.value >= 2.5:
        return "C"
    elif self.value >= 1.5:
        return "D"
    else:
        return "F"
```

### 3. 比較演算

```python
def __lt__(self, other: 'Score') -> bool:
    """スコアの大小比較（値のみ）"""
    if not isinstance(other, Score):
        return NotImplemented
    return self.value < other.value

def __le__(self, other: 'Score') -> bool:
    """スコアの大小比較（値のみ）"""
    if not isinstance(other, Score):
        return NotImplemented
    return self.value <= other.value

def __gt__(self, other: 'Score') -> bool:
    """スコアの大小比較（値のみ）"""
    if not isinstance(other, Score):
        return NotImplemented
    return self.value > other.value

def __ge__(self, other: 'Score') -> bool:
    """スコアの大小比較（値のみ）"""
    if not isinstance(other, Score):
        return NotImplemented
    return self.value >= other.value

def __eq__(self, other: 'Score') -> bool:
    """スコアの等価性比較（値のみ）"""
    if not isinstance(other, Score):
        return NotImplemented
    return abs(self.value - other.value) < 0.01

def __ne__(self, other: 'Score') -> bool:
    """スコアの非等価性比較（値のみ）"""
    return not self.__eq__(other)
```

### 4. 統計演算

```python
@classmethod
def average(cls, scores: List['Score']) -> 'Score':
    """複数スコアの平均を計算"""
    if not scores:
        raise ValueError("Cannot calculate average of empty list")

    total_value = sum(score.value for score in scores)
    total_confidence = sum(score.confidence for score in scores)
    avg_value = total_value / len(scores)
    avg_confidence = total_confidence / len(scores)

    # 推論を統合
    combined_reasoning = f"Average of {len(scores)} scores: " + \
                        " | ".join(score.reasoning[:50] + "..." if len(score.reasoning) > 50
                                  else score.reasoning for score in scores)

    return cls(
        value=avg_value,
        confidence=avg_confidence,
        reasoning=combined_reasoning,
        metadata={"calculation_type": "average", "count": len(scores)}
    )

@classmethod
def weighted_average(cls, scores: List[Tuple['Score', float]]) -> 'Score':
    """重み付け平均を計算"""
    if not scores:
        raise ValueError("Cannot calculate weighted average of empty list")

    total_weight = sum(weight for _, weight in scores)
    if abs(total_weight - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    weighted_value = sum(score.value * weight for score, weight in scores)
    weighted_confidence = sum(score.confidence * weight for score, weight in scores)

    # 重み付け推論を統合
    combined_reasoning = f"Weighted average of {len(scores)} scores: " + \
                        " | ".join(f"{score.reasoning[:30]}... (w:{weight:.2f})"
                                  for score, weight in scores)

    return cls(
        value=weighted_value,
        confidence=weighted_confidence,
        reasoning=combined_reasoning,
        metadata={"calculation_type": "weighted_average", "count": len(scores)}
    )
```

## 🧪 テスト設計

### 1. 単体テスト

```python
class TestScore:
    """スコア値オブジェクトのテスト"""

    def test_valid_score_creation(self):
        """有効なスコアの作成テスト"""
        score = Score(
            value=4.5,
            confidence=0.9,
            reasoning="明確な根拠に基づく評価"
        )

        assert score.value == 4.5
        assert score.confidence == 0.9
        assert score.reasoning == "明確な根拠に基づく評価"

    def test_invalid_score_value_raises_error(self):
        """無効なスコア値でエラーが発生することをテスト"""
        with pytest.raises(ValueError):
            Score(
                value=6.0,  # 無効な値
                confidence=0.9,
                reasoning="無効なスコア"
            )

    def test_invalid_confidence_raises_error(self):
        """無効な信頼度でエラーが発生することをテスト"""
        with pytest.raises(ValueError):
            Score(
                value=4.0,
                confidence=1.5,  # 無効な値
                reasoning="無効な信頼度"
            )

    def test_empty_reasoning_raises_error(self):
        """空の理由でエラーが発生することをテスト"""
        with pytest.raises(ValueError):
            Score(
                value=4.0,
                confidence=0.9,
                reasoning=""  # 空の理由
            )

    def test_is_high_confidence(self):
        """高信頼度判定のテスト"""
        high_confidence_score = Score(
            value=4.0,
            confidence=0.9,
            reasoning="高信頼度のスコア"
        )

        low_confidence_score = Score(
            value=4.0,
            confidence=0.6,
            reasoning="低信頼度のスコア"
        )

        assert high_confidence_score.is_high_confidence()
        assert not low_confidence_score.is_high_confidence()

    def test_to_category(self):
        """スコアカテゴリ変換のテスト"""
        excellent_score = Score(value=4.8, confidence=0.9, reasoning="優秀")
        good_score = Score(value=4.0, confidence=0.8, reasoning="良好")
        fair_score = Score(value=3.0, confidence=0.7, reasoning="普通")
        poor_score = Score(value=2.0, confidence=0.6, reasoning="不良")
        very_poor_score = Score(value=1.2, confidence=0.5, reasoning="非常に不良")

        assert excellent_score.to_category() == "Excellent"
        assert good_score.to_category() == "Good"
        assert fair_score.to_category() == "Fair"
        assert poor_score.to_category() == "Poor"
        assert very_poor_score.to_category() == "Very Poor"

    def test_comparison_operators(self):
        """比較演算子のテスト"""
        score1 = Score(value=3.0, confidence=0.8, reasoning="スコア1")
        score2 = Score(value=4.0, confidence=0.9, reasoning="スコア2")
        score3 = Score(value=3.0, confidence=0.7, reasoning="スコア3")

        assert score1 < score2
        assert score2 > score1
        assert score1 <= score2
        assert score2 >= score1
        assert score1 == score3  # 値が同じ
        assert score1 != score2

    def test_average_calculation(self):
        """平均計算のテスト"""
        scores = [
            Score(value=3.0, confidence=0.8, reasoning="スコア1"),
            Score(value=4.0, confidence=0.9, reasoning="スコア2"),
            Score(value=5.0, confidence=0.7, reasoning="スコア3")
        ]

        avg_score = Score.average(scores)

        assert avg_score.value == 4.0  # (3+4+5)/3
        assert avg_score.confidence == 0.8  # (0.8+0.9+0.7)/3
        assert "Average of 3 scores" in avg_score.reasoning
        assert avg_score.metadata["calculation_type"] == "average"
        assert avg_score.metadata["count"] == 3

    def test_weighted_average_calculation(self):
        """重み付け平均計算のテスト"""
        scores_with_weights = [
            (Score(value=3.0, confidence=0.8, reasoning="スコア1"), 0.3),
            (Score(value=4.0, confidence=0.9, reasoning="スコア2"), 0.5),
            (Score(value=5.0, confidence=0.7, reasoning="スコア3"), 0.2)
        ]

        weighted_avg = Score.weighted_average(scores_with_weights)

        expected_value = 3.0 * 0.3 + 4.0 * 0.5 + 5.0 * 0.2  # 3.9
        expected_confidence = 0.8 * 0.3 + 0.9 * 0.5 + 0.7 * 0.2  # 0.83

        assert abs(weighted_avg.value - expected_value) < 0.01
        assert abs(weighted_avg.confidence - expected_confidence) < 0.01
        assert "Weighted average of 3 scores" in weighted_avg.reasoning
        assert weighted_avg.metadata["calculation_type"] == "weighted_average"

    def test_immutability(self):
        """不変性のテスト"""
        score = Score(value=4.0, confidence=0.8, reasoning="テスト")

        # 属性の変更を試行（frozen=Trueにより失敗するはず）
        with pytest.raises(AttributeError):
            score.value = 5.0

        with pytest.raises(AttributeError):
            score.confidence = 0.9

        with pytest.raises(AttributeError):
            score.reasoning = "変更された理由"
```

### 2. 統合テスト

```python
class TestScoreIntegration:
    """スコア値オブジェクトの統合テスト"""

    def test_score_in_evaluation_result(self):
        """評価結果でのスコア使用テスト"""
        score = Score(
            value=4.2,
            confidence=0.85,
            reasoning="包括的で正確な回答",
            metadata={"metric": "accuracy", "provider": "openai"}
        )

        evaluation_result = EvaluationResult(
            id=EvaluationResultId.generate(),
            evaluation_id=EvaluationId.generate(),
            overall_score=score,
            metric_scores={"accuracy": score},
            evaluation_type="standard",
            created_at=datetime.utcnow(),
            metadata={}
        )

        assert evaluation_result.overall_score == score
        assert evaluation_result.get_metric_score("accuracy") == score
        assert evaluation_result.is_overall_success(threshold=4.0)

    def test_score_aggregation(self):
        """スコア集約のテスト"""
        scores = [
            Score(value=3.5, confidence=0.8, reasoning="正確性: 良好"),
            Score(value=4.0, confidence=0.9, reasoning="関連性: 優秀"),
            Score(value=3.0, confidence=0.7, reasoning="明確性: 普通")
        ]

        # 平均スコア計算
        avg_score = Score.average(scores)

        # 重み付け平均スコア計算
        weighted_scores = [
            (scores[0], 0.4),  # 正確性: 40%
            (scores[1], 0.4),  # 関連性: 40%
            (scores[2], 0.2)   # 明確性: 20%
        ]
        weighted_avg = Score.weighted_average(weighted_scores)

        assert avg_score.value == 3.5  # (3.5+4.0+3.0)/3
        assert weighted_avg.value == 3.5 * 0.4 + 4.0 * 0.4 + 3.0 * 0.2  # 3.6
        assert avg_score.confidence == 0.8  # (0.8+0.9+0.7)/3
```

## 🔧 実装ガイドライン

### 1. 不変性の保証

- **frozen=True**: dataclass の frozen 属性を使用
- **検証**: 作成時の妥当性検証
- **エラー処理**: 無効な値での例外発生

### 2. 等価性の実装

- **値ベース**: 値による等価性判定
- **精度考慮**: 浮動小数点の精度を考慮
- **ハッシュ**: ハッシュ可能な実装

### 3. 比較演算の実装

- **一貫性**: 比較演算の一貫性
- **型安全性**: 型チェックの実装
- **NotImplemented**: 未対応型での適切な処理

## 📊 パフォーマンス考慮事項

### 1. メモリ使用量

- **不変性**: オブジェクトの共有によるメモリ効率
- **メタデータ**: メタデータのサイズ制限
- **キャッシュ**: 頻繁に使用されるスコアのキャッシュ

### 2. 計算効率

- **統計演算**: 大量スコアの効率的な集約
- **比較演算**: 高速な比較演算の実装
- **文字列処理**: 推論文字列の効率的な処理

## 🔄 将来の拡張性

### 1. 新しい統計演算

- **中央値**: 中央値の計算
- **標準偏差**: 分散の計算
- **分位数**: 分位数の計算

### 2. 高度な分析機能

- **トレンド分析**: 時系列でのスコア変化分析
- **異常検出**: 異常なスコアの検出
- **予測**: スコアの予測機能

### 3. 可視化機能

- **グラフ生成**: スコアの可視化
- **レポート**: スコアレポートの生成
- **ダッシュボード**: リアルタイムダッシュボード

---

_このスコア値オブジェクト設計により、Disclosure Evaluator は正確で信頼性の高い評価システムを実現します。_
