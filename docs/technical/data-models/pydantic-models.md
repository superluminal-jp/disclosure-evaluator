# Pydantic モデル設計

## 📋 文書情報

| 項目       | 内容                |
| ---------- | ------------------- |
| 文書名     | Pydantic モデル設計 |
| バージョン | 1.0                 |
| 作成日     | 2025 年 9 月 28 日  |
| 作成者     | AI 開発チーム       |
| 承認者     | 技術責任者          |
| ステータス | 草案                |

## 🎯 概要

Disclosure Evaluator では、Pydantic を使用して型安全なデータモデルを定義します。これにより、データの検証、シリアライゼーション、ドキュメント生成を自動化します。

## 🏗️ 基本モデル設計

### 1. 基底モデル

```python
from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum

class BaseEntity(BaseModel):
    """基底エンティティモデル"""

    id: str = Field(..., description="一意識別子")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="作成日時")
    updated_at: Optional[datetime] = Field(None, description="最終更新日時")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="追加メタデータ")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True
        use_enum_values = True
```

### 2. 列挙型定義

```python
class ProviderType(str, Enum):
    """LLMプロバイダータイプ"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    GEMINI = "gemini"
    OLLAMA = "ollama"

class EvaluationStatus(str, Enum):
    """評価ステータス"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class MetricType(str, Enum):
    """評価指標タイプ"""
    STANDARD = "standard"
    LEGAL = "legal"
    CUSTOM = "custom"

class LegalArticle(str, Enum):
    """情報公開法条文"""
    PERSONAL_INFO = "1"  # 個人情報保護
    CORPORATE_INFO = "2"  # 法人等情報保護
    NATIONAL_SECURITY = "3"  # 国家安全保障
    PUBLIC_SAFETY = "4"  # 公共の安全と秩序
    INTERNAL_DELIBERATION = "5"  # 内部審議等
    ADMINISTRATIVE_OPERATION = "6"  # 行政運営等

class DisclosureDecision(str, Enum):
    """開示判断"""
    DISCLOSE = "disclose"
    CONSIDER_DISCLOSURE = "consider_disclosure"
    NON_DISCLOSURE = "non_disclosure"
```

## 📊 評価モデル

### 1. 評価エンティティモデル

```python
class EvaluationModel(BaseEntity):
    """評価エンティティモデル"""

    provider_id: str = Field(..., description="LLMプロバイダーID")
    prompt: str = Field(..., min_length=1, description="評価対象のプロンプト")
    response: str = Field(..., min_length=1, description="評価対象の応答")
    status: EvaluationStatus = Field(..., description="評価ステータス")
    criteria: "EvaluationCriteriaModel" = Field(..., description="評価基準")
    results: List["EvaluationResultModel"] = Field(default_factory=list, description="評価結果")

    @validator('prompt', 'response')
    def validate_non_empty(cls, v):
        if not v.strip():
            raise ValueError("プロンプトと応答は空にできません")
        return v.strip()

    @validator('results')
    def validate_results_consistency(cls, v, values):
        if 'status' in values and values['status'] == EvaluationStatus.COMPLETED:
            if not v:
                raise ValueError("完了ステータスの評価には結果が必要です")
        return v

    class Config:
        schema_extra = {
            "example": {
                "id": "eval_12345",
                "provider_id": "openai",
                "prompt": "What is artificial intelligence?",
                "response": "Artificial intelligence is...",
                "status": "completed",
                "criteria": {
                    "metrics": ["accuracy", "relevance"],
                    "weights": {"accuracy": 0.6, "relevance": 0.4},
                    "evaluation_type": "standard"
                },
                "results": [],
                "created_at": "2025-09-28T10:30:00Z",
                "metadata": {}
            }
        }
```

### 2. 評価結果モデル

```python
class ScoreModel(BaseModel):
    """スコアモデル"""

    value: float = Field(..., ge=1.0, le=5.0, description="1-5スケールのスコア")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0.0-1.0の信頼度")
    reasoning: str = Field(..., min_length=1, description="評価理由")
    metadata: Optional[Dict[str, Any]] = Field(None, description="追加メタデータ")

    @validator('reasoning')
    def validate_reasoning_not_empty(cls, v):
        if not v.strip():
            raise ValueError("評価理由は空にできません")
        return v.strip()

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """高信頼度スコアかチェック"""
        return self.confidence >= threshold

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

class EvaluationResultModel(BaseEntity):
    """評価結果モデル"""

    evaluation_id: str = Field(..., description="評価ID")
    overall_score: ScoreModel = Field(..., description="総合スコア")
    metric_scores: Dict[str, ScoreModel] = Field(..., description="評価指標別スコア")
    evaluation_type: str = Field(..., description="評価タイプ")
    execution_time: Optional[float] = Field(None, description="実行時間（秒）")

    @validator('metric_scores')
    def validate_metric_scores_not_empty(cls, v):
        if not v:
            raise ValueError("評価指標スコアは空にできません")
        return v

    @validator('evaluation_type')
    def validate_evaluation_type(cls, v):
        valid_types = ["standard", "legal", "custom"]
        if v not in valid_types:
            raise ValueError(f"評価タイプは {valid_types} のいずれかである必要があります")
        return v

    def get_metric_score(self, metric_id: str) -> Optional[ScoreModel]:
        """指定された評価指標のスコアを取得"""
        return self.metric_scores.get(metric_id)

    def is_overall_success(self, threshold: float = 3.0) -> bool:
        """総合的な成功判定"""
        return self.overall_score.value >= threshold
```

### 3. 情報公開法評価結果モデル

```python
class LegalEvaluationResultModel(BaseEntity):
    """情報公開法評価結果モデル"""

    evaluation_id: str = Field(..., description="評価ID")
    disclosure_decision: DisclosureDecision = Field(..., description="開示判断")
    article_scores: Dict[str, ScoreModel] = Field(..., description="条文別スコア")
    legal_reasoning: str = Field(..., min_length=1, description="法的推論")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="総合信頼度")

    @validator('article_scores')
    def validate_article_scores(cls, v):
        valid_articles = [article.value for article in LegalArticle]
        for article in v.keys():
            if article not in valid_articles:
                raise ValueError(f"無効な条文: {article}")
        return v

    @validator('legal_reasoning')
    def validate_legal_reasoning_not_empty(cls, v):
        if not v.strip():
            raise ValueError("法的推論は空にできません")
        return v.strip()

    def get_article_score(self, article: str) -> Optional[ScoreModel]:
        """指定された条文のスコアを取得"""
        return self.article_scores.get(article)

    def get_failed_articles(self, threshold: float = 2.0) -> List[str]:
        """失敗した条文を取得"""
        return [
            article for article, score in self.article_scores.items()
            if score.value < threshold
        ]
```

## 📋 評価指標モデル

### 1. 評価指標モデル

```python
class MetricDefinitionModel(BaseModel):
    """評価指標定義モデル"""

    id: str = Field(..., description="評価指標ID")
    name: str = Field(..., min_length=1, description="評価指標名")
    description: str = Field(..., min_length=1, description="説明")
    metric_type: MetricType = Field(..., description="評価指標タイプ")
    weight: float = Field(..., ge=0.0, le=1.0, description="重み")
    scoring_criteria: Dict[int, str] = Field(..., description="スコア基準")
    evaluation_prompt: str = Field(..., min_length=1, description="評価プロンプト")
    evaluation_steps: List[str] = Field(..., min_items=1, description="評価ステップ")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="評価例")
    reference_text: Optional[str] = Field(None, description="参考テキスト")

    @validator('scoring_criteria')
    def validate_scoring_criteria(cls, v):
        if len(v) != 5:
            raise ValueError("スコア基準は5つ必要です")
        if not all(1 <= score <= 5 for score in v.keys()):
            raise ValueError("スコアは1-5の範囲である必要があります")
        return v

    @validator('evaluation_steps')
    def validate_evaluation_steps_not_empty(cls, v):
        if not v:
            raise ValueError("評価ステップは空にできません")
        return v

    @validator('examples')
    def validate_examples_format(cls, v):
        for example in v:
            if not isinstance(example, dict):
                raise ValueError("評価例は辞書形式である必要があります")
        return v

class EvaluationCriteriaModel(BaseModel):
    """評価基準モデル"""

    metrics: List[str] = Field(..., min_items=1, description="評価指標IDリスト")
    weights: Dict[str, float] = Field(..., description="指標別重み付け")
    evaluation_type: str = Field(..., description="評価タイプ")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="評価パラメータ")

    @validator('weights')
    def validate_weights_sum_to_one(cls, v, values):
        if 'metrics' in values:
            total_weight = sum(v.values())
            if abs(total_weight - 1.0) > 0.001:
                raise ValueError(f"重み付けの合計は1.0である必要があります: {total_weight}")

            if set(v.keys()) != set(values['metrics']):
                raise ValueError("重み付けのキーは評価指標と一致する必要があります")
        return v

    @validator('evaluation_type')
    def validate_evaluation_type(cls, v):
        valid_types = ["standard", "legal", "custom"]
        if v not in valid_types:
            raise ValueError(f"評価タイプは {valid_types} のいずれかである必要があります")
        return v

    def get_weighted_score(self, scores: Dict[str, ScoreModel]) -> float:
        """重み付けされた総合スコアを計算"""
        total_score = 0.0
        total_weight = 0.0

        for metric_id, score in scores.items():
            if metric_id in self.weights:
                weight = self.weights[metric_id]
                total_score += score.value * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0
```

## 🔧 プロバイダーモデル

### 1. プロバイダー設定モデル

```python
class ProviderConfigModel(BaseModel):
    """プロバイダー設定モデル"""

    name: str = Field(..., description="プロバイダー名")
    provider_type: ProviderType = Field(..., description="プロバイダータイプ")
    api_key: Optional[str] = Field(None, description="APIキー")
    model: str = Field(..., description="モデル名")
    base_url: Optional[str] = Field(None, description="ベースURL")
    timeout: int = Field(30, ge=1, le=300, description="タイムアウト時間（秒）")
    max_tokens: int = Field(4000, ge=1, le=100000, description="最大トークン数")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="温度パラメータ")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="追加パラメータ")

    @validator('api_key')
    def validate_api_key_not_empty_if_required(cls, v, values):
        if 'provider_type' in values and values['provider_type'] in [ProviderType.OPENAI, ProviderType.ANTHROPIC]:
            if not v:
                raise ValueError("APIキーは必須です")
        return v

    @validator('base_url')
    def validate_base_url_format(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("ベースURLはhttp://またはhttps://で始まる必要があります")
        return v

    def get_connection_params(self) -> Dict[str, Any]:
        """接続パラメータを取得"""
        params = {
            "model": self.model,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self.additional_params
        }

        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["base_url"] = self.base_url

        return params
```

### 2. プロバイダー応答モデル

```python
class ProviderResponseModel(BaseModel):
    """プロバイダー応答モデル"""

    content: str = Field(..., description="応答内容")
    model: str = Field(..., description="使用モデル")
    usage: Dict[str, int] = Field(..., description="使用量情報")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="メタデータ")
    response_time: Optional[float] = Field(None, description="応答時間（秒）")

    @validator('content')
    def validate_content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("応答内容は空にできません")
        return v.strip()

    @validator('usage')
    def validate_usage_contains_required_fields(cls, v):
        required_fields = ["total_tokens", "prompt_tokens", "completion_tokens"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"使用量情報に {field} が必要です")
        return v
```

## 🧪 テストモデル

### 1. テストデータモデル

```python
class TestDataModel(BaseModel):
    """テストデータモデル"""

    test_id: str = Field(..., description="テストID")
    test_name: str = Field(..., description="テスト名")
    test_type: str = Field(..., description="テストタイプ")
    input_data: Dict[str, Any] = Field(..., description="入力データ")
    expected_output: Dict[str, Any] = Field(..., description="期待される出力")
    actual_output: Optional[Dict[str, Any]] = Field(None, description="実際の出力")
    test_status: str = Field("pending", description="テストステータス")
    test_result: Optional[Dict[str, Any]] = Field(None, description="テスト結果")

    @validator('test_status')
    def validate_test_status(cls, v):
        valid_statuses = ["pending", "running", "passed", "failed", "skipped"]
        if v not in valid_statuses:
            raise ValueError(f"テストステータスは {valid_statuses} のいずれかである必要があります")
        return v

    def is_passed(self) -> bool:
        """テストが成功したかチェック"""
        return self.test_status == "passed"

    def is_failed(self) -> bool:
        """テストが失敗したかチェック"""
        return self.test_status == "failed"
```

## 🔧 実装ガイドライン

### 1. バリデーション設計

```python
def validate_model_data(model_class: Type[BaseModel], data: Dict[str, Any]) -> BaseModel:
    """モデルデータの検証"""
    try:
        return model_class(**data)
    except ValidationError as e:
        raise ValueError(f"データ検証エラー: {e}")
    except Exception as e:
        raise ValueError(f"予期しないエラー: {e}")
```

### 2. シリアライゼーション

```python
def serialize_model(model: BaseModel, format: str = "json") -> str:
    """モデルのシリアライゼーション"""
    if format == "json":
        return model.json(indent=2, ensure_ascii=False)
    elif format == "yaml":
        import yaml
        return yaml.dump(model.dict(), default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"サポートされていない形式: {format}")
```

### 3. デシリアライゼーション

```python
def deserialize_model(model_class: Type[BaseModel], data: str, format: str = "json") -> BaseModel:
    """モデルのデシリアライゼーション"""
    if format == "json":
        return model_class.parse_raw(data)
    elif format == "yaml":
        import yaml
        return model_class(**yaml.safe_load(data))
    else:
        raise ValueError(f"サポートされていない形式: {format}")
```

## 📊 パフォーマンス考慮事項

### 1. メモリ使用量

- **大きなデータ**: 大量データの効率的な処理
- **キャッシュ**: 頻繁に使用されるモデルのキャッシュ
- **遅延読み込み**: 必要時のみデータを読み込み

### 2. 検証性能

- **バリデーション**: 効率的なデータ検証
- **エラーハンドリング**: 適切なエラーメッセージ
- **型変換**: 高速な型変換処理

## 🔄 将来の拡張性

### 1. 新しいモデルタイプ

- **カスタムモデル**: ユーザー定義のモデル
- **動的モデル**: 実行時モデル生成
- **バージョン管理**: モデルのバージョン管理

### 2. 高度な検証

- **カスタムバリデーター**: ドメイン固有の検証
- **条件付き検証**: 条件に基づく検証
- **相互検証**: 複数フィールド間の検証

---

_この Pydantic モデル設計により、Disclosure Evaluator は型安全で信頼性の高いデータ処理システムを実現します。_
