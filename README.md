# 情報公開法評価システム (Disclosure Evaluator)

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 概要

情報公開法評価システムは、情報公開法第 5 条に基づく 6 つの不開示事由を体系的に評価するための AI 駆動システムです。大規模言語モデル（LLM）を活用し、行政機関が保有する情報の開示・非開示判断を支援します。

### 主要機能

- **段階的評価プロセス**: 情報公開法第 5 条の各不開示事由について、法的根拠に基づく段階的評価を実施
- **複数 LLM プロバイダー対応**: OpenAI、Anthropic、AWS Bedrock など複数の LLM プロバイダーをサポート
- **バッチ処理機能**: 大量の文書を効率的に処理
- **並列処理**: 最大 30 の並列ワーカーによる高速処理
- **詳細な評価基準**: 各不開示事由について具体的な評価基準とスコアリング（1-5 スケール）

## 特徴

### LLM ベースの段階的評価プロセス

- 情報公開法の専門知識に基づく厳密な評価
- 「知る権利」の保障を最優先に考慮
- 不開示事由の該当性を厳格に判断
- 疑わしい場合は開示に有利な判断を選択

### 対応 LLM プロバイダー

| プロバイダー    | モデル                               | 特徴                 |
| --------------- | ------------------------------------ | -------------------- |
| **OpenAI**      | GPT-4, GPT-5-nano                    | 高精度な法的判断     |
| **Anthropic**   | Claude Sonnet 4                      | 専門的な法的分析     |
| **AWS Bedrock** | Claude Sonnet 4, Amazon Nova Premier | エンタープライズ対応 |

### バッチ処理機能

- フォルダ単位での一括処理
- 特定ファイルの選択的処理
- 処理状況の追跡と管理
- 失敗した文書の再処理
- **対応ファイル形式**: テキストファイル（.txt）のみ

### 並列処理サポート

- 最大 30 の並列ワーカー
- タイムアウト設定（デフォルト 300 秒）
- リトライ機能（最大 3 回）
- ファイルサイズ制限（デフォルト 50MB）

### ファイル形式の制限

- **対応形式**: テキストファイル（.txt）のみ
- **非対応形式**: PDF、DOCX、XLSX、CSV
- **事前変換**: 他の形式を使用する場合はテキスト形式への変換が必要

## インストール

### 前提条件

- Python 3.12 以上
- pip（Python パッケージマネージャー）

### セットアップ手順

1. **リポジトリのクローン**

```bash
git clone <repository-url>
cd disclosure-evaluator
```

2. **仮想環境の作成とアクティベート**

```bash
# 仮想環境の作成
python -m venv venv

# アクティベート
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate       # Windows
```

3. **依存関係のインストール**

```bash
pip install -r requirements.txt
```

### 必要な依存関係

| パッケージ       | バージョン | 用途                 |
| ---------------- | ---------- | -------------------- |
| `pydantic`       | >=2.0.0    | データバリデーション |
| `python-dotenv`  | >=1.0.0    | 環境変数管理         |
| `openai`         | >=1.0.0    | OpenAI API           |
| `anthropic`      | >=0.60.0   | Anthropic API        |
| `boto3`          | >=1.39.0   | AWS Bedrock          |
| `pytest`         | >=7.0.0    | テスト実行           |
| `pytest-asyncio` | >=0.21.0   | 非同期テスト         |
| `pytest-cov`     | >=4.0.0    | テストカバレッジ     |

### 開発用依存関係

```bash
# 開発用パッケージのインストール
pip install pytest-xdist  # 並列テスト実行
pip install black          # コードフォーマッター
pip install flake8         # リンター
```

> **注意**: 現在のシステムはテキストファイルのみをサポートしています。PDF、DOCX、XLSX、CSV ファイルを処理する場合は、事前にテキスト形式に変換してください。

## 設定

### 環境変数の設定

`.env`ファイルを作成し、使用する LLM プロバイダーの API キーを設定してください：

```bash
# OpenAI使用の場合
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic使用の場合
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# AWS Bedrock使用の場合
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1
```

### config.json の設定

`config.json`ファイルでシステムの動作をカスタマイズできます：

```json
{
  "application": {
    "name": "Disclosure Evaluator",
    "version": "2.0.0"
  },
  "llm": {
    "provider": "anthropic",
    "anthropic": {
      "model": "claude-sonnet-4-5-20250929",
      "temperature": 0.1,
      "max_tokens": 2000,
      "timeout": 30
    }
  },
  "evaluation": {
    "parallel": {
      "enabled": true,
      "max_workers": 30,
      "timeout": 300
    }
  }
}
```

### 主要設定項目

| 設定項目                          | 説明                      | デフォルト値 |
| --------------------------------- | ------------------------- | ------------ |
| `llm.provider`                    | 使用する LLM プロバイダー | `anthropic`  |
| `evaluation.parallel.max_workers` | 並列処理数                | `30`         |
| `evaluation.parallel.timeout`     | タイムアウト時間（秒）    | `300`        |
| `llm.temperature`                 | モデルの創造性            | `0.1`        |
| `llm.max_tokens`                  | 最大トークン数            | `2000`       |

## 使用方法

このツールは、行政機関が保有する情報が情報公開法第 5 条の不開示事由に該当するかどうかを事前に評価するために使用します。ユーザーは、開示・非開示の判断に迷う文書について、AI による専門的な評価を受けることで、適切な判断の参考とすることができます。

### 主な使用シナリオ

- **情報公開請求への対応**: 請求された文書が不開示事由に該当するかどうかの事前評価
- **文書管理**: 新規作成・受領した文書の開示可能性の事前確認
- **法務チェック**: 文書の開示判断における法的根拠の確認
- **リスク評価**: 不適切な開示によるリスクの事前把握
- **教育・研修**: 職員の情報公開法理解の向上

### 単一文書の評価

#### 引数の説明

```bash
python main.py <input_text> [context] [output_text] [--format json|summary] [--provider openai|anthropic|bedrock|bedrock_nova]
```

| 引数          | 説明                                                        | 必須       |
| ------------- | ----------------------------------------------------------- | ---------- |
| `input_text`  | 評価したい文書の内容                                        | 必須       |
| `context`     | 文書の開示目的・用途（オプション）                          | オプション |
| `output_text` | 評価結果の識別子（オプション）                              | オプション |
| `--format`    | 出力形式（json\|summary、デフォルト: json）                 | オプション |
| `--provider`  | AI プロバイダー（openai\|anthropic\|bedrock\|bedrock_nova） | オプション |

> **ヒント**: `context`引数は評価の文脈を提供し、AI の判断精度を向上させます。

#### サンプル

```bash
# 基本的な単一文書評価
python main.py "Personal information: John Doe" --format summary

# 特定のプロバイダーを指定
python main.py "Personal information: John Doe" --format summary --provider openai
python main.py "Personal information: John Doe" --format summary --provider anthropic
python main.py "Personal information: John Doe" --format summary --provider bedrock
python main.py "Personal information: John Doe" --format summary --provider bedrock_nova

# コンテキスト付き評価
python main.py "Personal information: John Doe" "Legal review for disclosure" "doc_001" --format json
```

### バッチ処理

#### サンプル文書の一括処理

```bash
# 全サンプル文書の一括評価（個人情報、法人情報、国家安全保障、内部審議等の各不開示事由に該当する具体的な文書）
python main.py --batch --folder sample_documents --max-workers 5

# 特定の文書タイプの評価（個人情報保護・法人等情報保護の不開示事由に該当する文書）
python main.py --batch --documents "personal_info_sample.txt,corporate_info_sample.txt"

# 並列処理数の調整
python main.py --batch --folder sample_documents --max-workers 10

# 再帰的ディレクトリ検索（デフォルト: true）
python main.py --batch --folder documents --recursive

# ファイルタイプフィルタリング
python main.py --batch --folder documents --file-types "text/plain"

# 除外パターンの指定
python main.py --batch --folder documents --exclude "*.tmp,*.bak"

# ファイルサイズ制限
python main.py --batch --folder documents --file-size-limit 10485760  # 10MB

# タイムアウト設定
python main.py --batch --folder documents --timeout 600  # 10分

# リトライ回数設定
python main.py --batch --folder documents --retry-attempts 5

# 全文書に共通のコンテキストを追加
python main.py --batch --folder documents --context "Legal review for disclosure"

# 複数出力形式の指定
python main.py --batch --folder documents --output-formats "json,summary,csv"
```

#### バッチ処理の管理

```bash
# バッチ処理の状況確認
python main.py --batch-status <batch_id>

# バッチ処理の結果取得（複数形式対応）
python main.py --batch-results <batch_id> --format json
python main.py --batch-results <batch_id> --format summary
python main.py --batch-results <batch_id> --format csv

# 中断されたバッチ処理の再開
python main.py --resume-batch <batch_id>

# 失敗した文書の再処理
python main.py --retry-documents <batch_id> <document_id1,document_id2>

# ヘルプの表示
python main.py --help
```

## 評価基準

### 情報公開法第 5 条の 6 つの不開示事由

| 事由                    | 条文           | 評価ポイント                                   |
| ----------------------- | -------------- | ---------------------------------------------- |
| **1. 個人情報保護**     | 第 5 条第 1 号 | 個人識別可能性、権利利益への影響、開示例外要件 |
| **2. 法人等情報保護**   | 第 5 条第 2 号 | 競争上の地位への影響、正当な利益の保護         |
| **3. 国家安全保障**     | 第 5 条第 3 号 | 国の安全、国際関係、外交交渉への影響           |
| **4. 公共の安全と秩序** | 第 5 条第 4 号 | 犯罪予防・捜査、公訴維持、公共安全への影響     |
| **5. 内部審議等**       | 第 5 条第 5 号 | 率直な意見交換の保護、意思決定の中立性         |
| **6. 行政運営等**       | 第 5 条第 6 号 | 監査・検査、契約・交渉、調査研究への影響       |

### スコアリング基準（1-5 スケール）

| スコア | 意味                                       | 判断                 |
| ------ | ------------------------------------------ | -------------------- |
| **1**  | 明確に不開示事由に該当し、不開示が必要     | 不開示               |
| **2**  | 不開示事由に該当し、不開示が適切           | 不開示の可能性が高い |
| **3**  | 不開示事由の該当性があり、慎重な検討が必要 | 慎重な検討が必要     |
| **4**  | 不開示事由の該当性は低く、開示を検討可能   | 開示の可能性が高い   |
| **5**  | 不開示事由に該当せず、開示が適切           | 開示                 |

## テスト

### テスト構造

このプロジェクトは包括的なテストスイートを提供し、単体テスト、統合テスト、エンドツーエンドテストを含みます。

```
tests/
├── conftest.py              # 共有フィクスチャとモック
├── fixtures/                # テスト用データ
│   ├── criteria_test.json
│   ├── mock_responses.json
│   └── sample_documents/
├── unit/                    # 単体テスト
│   ├── test_batch_evaluator.py
│   ├── test_batch_persistence.py
│   ├── test_cli.py
│   ├── test_config_manager.py
│   ├── test_criterion_evaluator.py
│   ├── test_disclosure_evaluator.py
│   ├── test_document_discovery.py
│   ├── test_llm_providers.py
│   ├── test_load_criteria.py
│   ├── test_models.py
│   ├── test_parallel_processing.py
│   ├── test_result_aggregator.py
│   └── test_step_evaluator.py
└── integration/             # 統合テスト
    ├── test_batch_processing.py
    ├── test_batch_processing_simple.py
    ├── test_cli_commands.py
    └── test_end_to_end_evaluation.py
```

### テスト実行

#### 全テストの実行

```bash
# 全テストを実行
pytest

# 詳細出力でテスト実行
pytest -v

# カバレッジレポート付きでテスト実行
pytest --cov=main --cov-report=html

# 特定のテストファイルを実行
pytest tests/unit/test_disclosure_evaluator.py

# 特定のテストクラスを実行
pytest tests/unit/test_disclosure_evaluator.py::TestDisclosureEvaluator
```

#### テストカテゴリ別実行

```bash
# 単体テストのみ実行
pytest tests/unit/

# 統合テストのみ実行
pytest tests/integration/

# 特定のパターンでテスト実行
pytest -k "test_batch"
```

#### 並列テスト実行

```bash
# pytest-xdistを使用した並列実行（インストールが必要）
pip install pytest-xdist
pytest -n auto  # CPUコア数に応じて並列実行
```

### テストカバレッジ

#### カバレッジレポートの生成

```bash
# HTMLカバレッジレポートを生成
pytest --cov=main --cov-report=html

# コンソールにカバレッジサマリーを表示
pytest --cov=main --cov-report=term-missing

# カバレッジ閾値を設定
pytest --cov=main --cov-fail-under=80
```

#### カバレッジレポートの確認

HTML レポートは`htmlcov/`ディレクトリに生成され、ブラウザで確認できます：

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### テストデータとフィクスチャ

#### 共有フィクスチャ（conftest.py）

- **mock_openai_client**: OpenAI API のモック
- **mock_anthropic_client**: Anthropic API のモック
- **mock_bedrock_client**: AWS Bedrock API のモック
- **sample_criteria**: 評価基準のテストデータ
- **sample_documents_dir**: サンプル文書ディレクトリ
- **test_batch_configuration**: バッチ処理設定

#### テスト用データ

```bash
# テスト用サンプル文書
tests/fixtures/sample_documents/
├── doc1.txt
├── doc2.txt
└── doc3.txt

# モックレスポンス
tests/fixtures/mock_responses.json

# テスト用評価基準
tests/fixtures/criteria_test.json
```

### テストの種類

#### 1. 単体テスト（Unit Tests）

各コンポーネントの個別機能をテスト：

- **LLM プロバイダー**: OpenAI、Anthropic、AWS Bedrock
- **評価エンジン**: 段階的評価ロジック
- **バッチ処理**: 並列処理、状態管理
- **設定管理**: 設定ファイルの読み込み
- **CLI**: コマンドライン引数の処理

#### 2. 統合テスト（Integration Tests）

コンポーネント間の連携をテスト：

- **バッチ処理ワークフロー**: 作成から完了まで
- **エンドツーエンド評価**: 入力から出力まで
- **CLI コマンド**: 実際のコマンド実行

#### 3. モックとスタブ

外部依存関係をモック化：

```python
# LLM プロバイダーのモック
@pytest.fixture
def mock_openai_client():
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mock response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client
```

### テスト設定

#### pytest.ini の設定

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
```

#### 環境変数の設定

テスト用の環境変数は自動的にモックされます：

```python
# テスト用APIキーが自動設定
OPENAI_API_KEY=test_key
ANTHROPIC_API_KEY=test_key
AWS_ACCESS_KEY_ID=test_key
```

### 継続的インテグレーション

#### GitHub Actions でのテスト実行

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=main --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### テストのベストプラクティス

#### 1. テストの命名規則

```python
def test_<component>_<action>_<expected_result>():
    """Test description."""
    pass

# 例
def test_disclosure_evaluator_init_openai():
    """Test DisclosureEvaluator initialization with OpenAI provider."""
    pass
```

#### 2. テストの独立性

各テストは独立して実行可能：

```python
def test_something(self, mock_client):
    """Test that doesn't depend on other tests."""
    # テストロジック
    pass
```

#### 3. アサーションの明確性

```python
# 良い例
assert result.score == 2
assert result.decision == "non_disclosure"
assert "個人情報" in result.reasoning

# 悪い例
assert result
```

### トラブルシューティング

#### よくあるテストエラー

1. **ImportError**: パスの問題

   ```bash
   # 解決方法
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **API Key Error**: 環境変数の問題

   ```bash
   # 解決方法
   export OPENAI_API_KEY=test_key
   ```

3. **Timeout Error**: 長時間実行テスト
   ```bash
   # 解決方法
   pytest --timeout=300
   ```

#### テストのデバッグ

```bash
# 特定のテストを詳細出力で実行
pytest -v -s tests/unit/test_disclosure_evaluator.py::TestDisclosureEvaluator::test_specific_method

# デバッガーでテスト実行
pytest --pdb tests/unit/test_disclosure_evaluator.py

# 失敗したテストのみ再実行
pytest --lf
```

## 開発者向け情報

### プロジェクト構造

```
disclosure-evaluator/
├── main.py                   # CLI エントリーポイント
├── config.json              # システム設定
├── requirements.txt          # 依存関係
├── src/                      # モジュラーソースコード
│   ├── __init__.py
│   ├── batch/                # バッチ処理モジュール
│   │   ├── evaluator.py       # バッチ評価エンジン
│   │   └── services/          # バッチ処理サービス
│   │       ├── discovery.py   # 文書発見サービス
│   │       ├── persistence.py # 状態永続化
│   │       └── processing.py   # 並列処理
│   ├── cli/                  # CLI コマンド
│   │   └── commands.py        # コマンドハンドラー
│   ├── config/               # 設定管理
│   │   ├── manager.py        # 設定マネージャー
│   │   └── prompts.py        # プロンプト管理
│   ├── evaluators/           # 評価エンジン
│   │   ├── disclosure_evaluator.py  # メイン評価ロジック
│   │   ├── criterion_evaluator.py   # 基準評価
│   │   ├── step_evaluator.py        # 段階的評価
│   │   └── result_aggregator.py     # 結果集約
│   ├── llm/                  # LLM プロバイダー
│   │   ├── factory.py        # プロバイダーファクトリー
│   │   └── providers.py      # 各プロバイダー実装
│   ├── models/               # データモデル
│   │   ├── batch.py          # バッチ処理モデル
│   │   └── evaluation.py     # 評価結果モデル
│   └── utils/                # ユーティリティ
│       ├── criteria.py      # 基準管理
│       ├── formatters.py    # 出力フォーマッター
│       └── logging.py       # ログ管理
├── criteria/
│   └── disclosure_evaluation_criteria.json  # 評価基準
├── sample_documents/         # サンプル文書（テキスト形式）
├── tests/                    # テストスイート
├── htmlcov/                  # カバレッジレポート
├── batch_state/              # バッチ処理状態
├── logs/                     # ログファイル
├── outputs/                  # 評価結果出力
└── venv/                    # 仮想環境
```

**注意**: システムはテキストファイル（.txt）のみをサポートしています。他の形式の文書を使用する場合は、事前にテキスト形式に変換してください。

### アーキテクチャ概要

システムはモジュラー設計により、各機能が独立したコンポーネントとして実装されています：

#### コアコンポーネント

- **ConfigManager** (`src/config/manager.py`): 設定管理と環境変数処理
- **DisclosureEvaluator** (`src/evaluators/disclosure_evaluator.py`): メイン評価ロジック
- **BatchEvaluator** (`src/batch/evaluator.py`): バッチ処理管理
- **LLMFactory** (`src/llm/factory.py`): LLM プロバイダーの動的生成

#### 評価エンジン

- **CriterionEvaluator** (`src/evaluators/criterion_evaluator.py`): 個別基準の評価
- **StepEvaluator** (`src/evaluators/step_evaluator.py`): 段階的評価プロセス
- **ResultAggregator** (`src/evaluators/result_aggregator.py`): 結果の集約とスコアリング

#### バッチ処理

- **DocumentDiscovery** (`src/batch/services/discovery.py`): 文書の発見とフィルタリング
- **BatchPersistence** (`src/batch/services/persistence.py`): バッチ状態の永続化
- **ParallelProcessor** (`src/batch/services/processing.py`): 並列処理管理

#### LLM プロバイダー

- **OpenAIProvider** (`src/llm/providers.py`): OpenAI API 統合
- **AnthropicProvider** (`src/llm/providers.py`): Anthropic API 統合
- **BedrockProvider** (`src/llm/providers.py`): AWS Bedrock 統合

### 主要コンポーネント

#### LLMProvider 抽象クラス

```python
class LLMProvider:
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """LLMからの応答を生成"""
        raise NotImplementedError
```

#### BatchEvaluator クラス

```python
class BatchEvaluator:
    def __init__(self, config_manager: ConfigManager):
        """バッチ評価エンジンの初期化"""

    def process_batch(self, configuration: BatchConfiguration) -> BatchResult:
        """バッチ処理の実行"""
```

#### DisclosureEvaluator クラス

```python
class DisclosureEvaluator:
    def evaluate(self, document: str, context: str = "") -> EvaluationResult:
        """単一文書の評価実行"""
```

#### 評価プロセス

1. 文書の前処理と分類
2. 各不開示事由の段階的評価
3. LLM による専門的判断
4. スコアの算出と根拠の生成
5. 総合的な開示判断

### カスタマイズ方法

#### 新しい LLM プロバイダーの追加

1. **プロバイダークラスの実装** (`src/llm/providers.py`):

```python
class CustomProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # カスタムプロバイダーの初期化

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        # カスタムプロバイダーの実装
        pass
```

2. **ファクトリーの更新** (`src/llm/factory.py`):

```python
def create_provider(provider_name: str, config: Dict[str, Any]) -> LLMProvider:
    if provider_name == "custom":
        return CustomProvider(config)
    # 既存のプロバイダー...
```

3. **設定ファイルの更新** (`config.json`):

```json
{
  "llm": {
    "provider": "custom",
    "custom": {
      "model": "your-model",
      "temperature": 0.1,
      "max_tokens": 2000,
      "timeout": 30
    }
  }
}
```

#### 評価基準のカスタマイズ

`criteria/disclosure_evaluation_criteria.json`を編集して、評価基準をカスタマイズできます。

#### 新しい評価基準の追加

1. **基準定義の追加** (`criteria/disclosure_evaluation_criteria.json`):

```json
{
  "criteria": {
    "article_5_7": {
      "name": "新たな不開示事由",
      "description": "カスタム不開示事由の説明",
      "steps": [
        {
          "step": "該当性の確認",
          "description": "ステップの説明"
        }
      ]
    }
  }
}
```

2. **評価ロジックの実装** (`src/evaluators/criterion_evaluator.py`):

```python
def evaluate_custom_criterion(self, document: str, context: str) -> CriterionResult:
    # カスタム評価ロジックの実装
    pass
```

#### バッチ処理のカスタマイズ

1. **新しい文書発見ロジック** (`src/batch/services/discovery.py`):

```python
class CustomDocumentDiscovery(DocumentDiscovery):
    def discover_documents(self, path: Path) -> List[DocumentInput]:
        # カスタム文書発見ロジック
        pass
```

2. **カスタム並列処理** (`src/batch/services/processing.py`):

```python
class CustomParallelProcessor(ParallelProcessor):
    def process_document(self, document: DocumentInput) -> EvaluationResult:
        # カスタム処理ロジック
        pass
```

### 開発ワークフロー

#### モジュラー開発のベストプラクティス

1. **コンポーネントの独立性**: 各モジュールは独立してテスト・開発可能
2. **インターフェースの一貫性**: 共通のインターフェースを維持
3. **設定の分離**: 設定は`config.json`で一元管理
4. **ログの構造化**: 全コンポーネントで統一されたログ形式

#### 開発環境のセットアップ

```bash
# 開発用仮想環境の作成
python -m venv venv-dev
source venv-dev/bin/activate  # Linux/Mac
# または
venv-dev\Scripts\activate     # Windows

# 開発用依存関係のインストール
pip install -r requirements.txt
pip install pytest-xdist black flake8 mypy

# コードフォーマット
black src/ tests/

# リンター実行
flake8 src/ tests/

# 型チェック
mypy src/

# テスト実行
pytest tests/ -v --cov=src
```

#### 新機能の追加手順

1. **機能の設計**: どのモジュールに属するかを決定
2. **インターフェースの定義**: 共通インターフェースの実装
3. **単体テストの作成**: テストファースト開発
4. **実装**: 機能の実装
5. **統合テスト**: 他コンポーネントとの連携確認
6. **ドキュメント更新**: README とコードコメントの更新

## 出力形式

### JSON 形式

#### 個人情報を含む文書の評価結果例

```json
{
  "document_id": "resident_record_2024_001",
  "evaluation_results": {
    "article_5_1": {
      "score": 2,
      "reasoning": "個人情報保護規定に該当する可能性が高い。氏名、住所、生年月日等の個人識別情報が含まれており、開示例外要件に該当しない限り不開示が適切",
      "steps": [
        {
          "step": "個人に関する情報か",
          "result": "YES",
          "reasoning": "氏名、住所、生年月日等の個人識別情報が明確に含まれている"
        },
        {
          "step": "開示例外要件に該当するか",
          "result": "NO",
          "reasoning": "公務員の職務遂行上の必要性や公的活動の透明性確保の観点から開示が求められる特別な事情は認められない"
        }
      ]
    },
    "article_5_2": {
      "score": 5,
      "reasoning": "法人等情報保護規定には該当しない",
      "steps": [
        {
          "step": "法人等の競争上の地位に関する情報か",
          "result": "NO",
          "reasoning": "個人情報であり、法人等の競争上の地位に関する情報ではない"
        }
      ]
    }
  },
  "overall_decision": "不開示",
  "confidence": 0.92,
  "context": "住民基本台帳に基づく人口統計資料の作成"
}
```

#### 企業情報を含む文書の評価結果例

```json
{
  "document_id": "public_works_bid_2024_003",
  "evaluation_results": {
    "article_5_2": {
      "score": 3,
      "reasoning": "法人等情報保護規定の該当性について慎重な検討が必要。技術仕様の一部は競争上の地位に影響する可能性があるが、公共事業の透明性確保の観点も重要",
      "steps": [
        {
          "step": "法人等の競争上の地位に関する情報か",
          "result": "YES",
          "reasoning": "技術仕様、価格情報等の競争上重要な情報が含まれている"
        },
        {
          "step": "開示により正当な利益が害されるか",
          "result": "PARTIAL",
          "reasoning": "一部の技術的詳細は開示により競争上の不利益が生じる可能性があるが、入札の透明性確保の観点から開示が求められる部分もある"
        }
      ]
    }
  },
  "overall_decision": "部分開示",
  "confidence": 0.78,
  "context": "公共事業入札に関する審査資料"
}
```

#### 外交文書の評価結果例

```json
{
  "document_id": "diplomatic_meeting_2024_012",
  "evaluation_results": {
    "article_5_3": {
      "score": 1,
      "reasoning": "国家安全保障規定に明確に該当し、不開示が適切。外交交渉の内容は国際関係に重大な影響を与える可能性がある",
      "steps": [
        {
          "step": "国の安全に関する情報か",
          "result": "YES",
          "reasoning": "外交交渉の詳細内容は国家安全保障に直接関わる情報である"
        },
        {
          "step": "開示により国際関係に影響するか",
          "result": "YES",
          "reasoning": "外交交渉の内容の開示は相手国との信頼関係に重大な影響を与える可能性が高い"
        }
      ]
    }
  },
  "overall_decision": "不開示",
  "confidence": 0.95,
  "context": "外交政策の検討資料"
}
```

### サマリー形式

#### 個人情報を含む文書の評価サマリー

```
=== 評価結果サマリー ===
文書ID: resident_record_2024_001
文書種別: 住民基本台帳に基づく人口統計資料
総合判断: 不開示
信頼度: 92%

【個人情報保護（第5条第1号）】
スコア: 2/5
判断: 不開示の可能性が高い
理由: 氏名、住所、生年月日等の個人識別情報が含まれており、
      開示例外要件に該当しない限り不開示が適切

【法人等情報保護（第5条第2号）】
スコア: 5/5
判断: 該当しない
理由: 個人情報であり、法人等の競争上の地位に関する情報ではない

【国家安全保障（第5条第3号）】
スコア: 5/5
判断: 該当しない
理由: 住民統計情報であり、国家安全保障に関する情報ではない
```

#### 企業情報を含む文書の評価サマリー

```
=== 評価結果サマリー ===
文書ID: public_works_bid_2024_003
文書種別: 公共事業入札に関する審査資料
総合判断: 部分開示
信頼度: 78%

【法人等情報保護（第5条第2号）】
スコア: 3/5
判断: 慎重な検討が必要
理由: 技術仕様の一部は競争上の地位に影響する可能性があるが、
      公共事業の透明性確保の観点も重要

【内部審議等（第5条第5号）】
スコア: 4/5
判断: 開示を検討可能
理由: 入札審査の透明性確保の観点から開示が求められる
```

#### 外交文書の評価サマリー

```
=== 評価結果サマリー ===
文書ID: diplomatic_meeting_2024_012
文書種別: 外交交渉の議事録
総合判断: 不開示
信頼度: 95%

【国家安全保障（第5条第3号）】
スコア: 1/5
判断: 明確に不開示事由に該当
理由: 外交交渉の内容は国際関係に重大な影響を与える可能性があり、
      国家安全保障上不開示が適切

【内部審議等（第5条第5号）】
スコア: 2/5
判断: 不開示の可能性が高い
理由: 外交交渉の詳細内容は率直な意見交換の保護が必要
```

### CSV 形式

#### 個人情報を含む文書の評価結果

```csv
document_id,document_type,criterion,score,decision,confidence,reasoning
resident_record_2024_001,住民基本台帳統計,article_5_1,2,non_disclosure,0.92,個人識別情報が含まれ開示例外に該当しない
resident_record_2024_001,住民基本台帳統計,article_5_2,5,disclosure,0.90,個人情報であり法人等情報ではない
resident_record_2024_001,住民基本台帳統計,article_5_3,5,disclosure,0.95,住民統計情報であり国家安全保障情報ではない
```

#### 企業情報を含む文書の評価結果

```csv
document_id,document_type,criterion,score,decision,confidence,reasoning
public_works_bid_2024_003,公共事業入札審査,article_5_2,3,partial_disclosure,0.78,技術仕様の一部は競争上重要だが透明性確保も重要
public_works_bid_2024_003,公共事業入札審査,article_5_5,4,disclosure,0.85,入札審査の透明性確保の観点から開示が求められる
```

#### 外交文書の評価結果

```csv
document_id,document_type,criterion,score,decision,confidence,reasoning
diplomatic_meeting_2024_012,外交交渉議事録,article_5_3,1,non_disclosure,0.95,外交交渉内容は国際関係に重大影響
diplomatic_meeting_2024_012,外交交渉議事録,article_5_5,2,non_disclosure,0.88,外交交渉の詳細は率直な意見交換保護が必要
```

## ファイル形式の対応

### 対応ファイル形式

**現在サポートされている形式**:

- **テキストファイル（.txt）**: 完全対応

**現在サポートされていない形式**:

- **PDF（.pdf）**: 対応していません
- **Word 文書（.docx）**: 対応していません
- **Excel（.xlsx）**: 対応していません
- **CSV（.csv）**: 対応していません

### 他のファイル形式を使用する場合

#### 1. 事前変換が必要

他のファイル形式を使用する場合は、事前にテキスト形式に変換してください：

```bash
# PDFをテキストに変換
pdftotext document.pdf document.txt

# Word文書をテキストに変換
# Microsoft Wordで「名前を付けて保存」→「テキスト形式」を選択

# ExcelをCSVに変換後、テキストエディタで開く
# Excelで「名前を付けて保存」→「CSV形式」を選択
```

#### 2. 変換ツールの使用例

```bash
# PDF変換ツール（pdftotext）のインストール
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# 変換実行
pdftotext input.pdf output.txt
python main.py --batch --documents output.txt
```

#### 3. 将来的な対応予定

システムの今後のバージョンでは以下の形式への対応を予定しています：

- PDF 文書の直接読み込み
- Word 文書の直接読み込み
- Excel 文書の直接読み込み
- CSV 文書の直接読み込み

## トラブルシューティング

### よくある問題

#### ファイル形式エラー

```
Error: Unsupported file format
```

**解決方法**: テキストファイル（.txt）のみがサポートされています。他の形式の場合は事前にテキスト形式に変換してください。

#### API キーの設定エラー

```
Error: API key not found
```

**解決方法**: `.env`ファイルに API キーが正しく設定されているか確認してください。

#### メモリ不足エラー

```
Error: Out of memory
```

**解決方法**: `config.json`の`max_workers`を減らすか、`file_size_limit`を小さくしてください。

#### タイムアウトエラー

```
Error: Request timeout
```

**解決方法**: `config.json`の`timeout`値を増やすか、文書サイズを小さくしてください。

### ログの確認

システムは詳細なログを出力します。ログレベルを調整するには`config.json`を編集してください：

```json
{
  "logging": {
    "level": "DEBUG",
    "format": "{\"timestamp\": \"%(asctime)s\", \"level\": \"%(levelname)s\", \"message\": \"%(message)s\"}"
  }
}
```

### エラーハンドリング

- **リトライ機能**: 失敗したリクエストは自動的に再試行されます（最大 3 回）
- **フォールバック**: LLM エラー時はデフォルトスコア（3）が適用されます
- **部分処理**: バッチ処理中に一部の文書でエラーが発生しても、他の文書の処理は継続されます

---

**注意**: このシステムは情報公開法の専門的判断を支援するツールです。最終的な開示・非開示判断は、法務専門家による検討が必要です。
