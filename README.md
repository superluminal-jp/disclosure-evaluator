# Disclosure Evaluator

高度な情報公開法評価システム - 日本の情報公開法に基づく段階的評価と LLM 活用による詳細分析

## 概要

このシステムは、日本の情報公開法（行政機関の保有する情報の公開に関する法律）に基づいて、情報の開示可能性を段階的に評価します。各不開示事由について詳細な評価を行い、ユーザーが適切な判断を行えるよう包括的な情報を提供します。

## 主な機能

- **段階的評価**: 各不開示事由について複数の段階で詳細評価
- **マルチプロバイダー対応**: OpenAI、Anthropic Claude、AWS Bedrock (Claude/Nova) をサポート
- **構造化出力**: JSON 形式と人間が読みやすいサマリー形式
- **詳細な理由**: 各評価段階の詳細な理由と法的根拠
- **ユーザー判断支援**: システムが判断を下すのではなく、判断に必要な情報を提供
- **構造化ログ**: 相関 ID による追跡可能なログ出力

## 評価対象の不開示事由

1. **個人情報保護** (第 5 条第 1 号)
2. **法人等情報保護** (第 5 条第 2 号)
3. **国家安全保障** (第 5 条第 3 号)
4. **公共の安全と秩序** (第 5 条第 4 号)
5. **内部審議等** (第 5 条第 5 号)
6. **行政運営等** (第 5 条第 6 号)

## 使用方法

### 基本的な使用方法

#### OpenAI を使用する場合

```bash
# OpenAI APIキーを設定
export OPENAI_API_KEY="your-api-key-here"

# 基本的な評価
python evaluator.py "評価対象の情報"

# コンテキスト付き評価
python evaluator.py "評価対象の情報" "追加のコンテキスト情報"

# 出力テキスト付き評価
python evaluator.py "評価対象の情報" "コンテキスト" "出力テキスト"
```

#### Anthropic Claude を使用する場合

```bash
# Anthropic APIキーを設定
export ANTHROPIC_API_KEY="your-api-key-here"

# 基本的な評価
python evaluator.py "評価対象の情報"

# プロバイダーを明示的に指定
python evaluator.py "評価対象の情報" --provider anthropic
```

#### AWS Bedrock を使用する場合

```bash
# AWS認証情報を設定
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_REGION="us-east-1"

# Anthropic Claude Sonnet 4 を使用
python evaluator.py "評価対象の情報" --provider bedrock

# Amazon Nova Premier を使用
python evaluator.py "評価対象の情報" --provider bedrock_nova
```

### 出力形式の選択

```bash
# JSON形式で出力
python evaluator.py "情報の内容" --format json

# 人間が読みやすいサマリー形式で出力（デフォルト）
python evaluator.py "情報の内容" --format summary
```

## 実行例

### 個人情報の評価（OpenAI 使用）

```bash
python evaluator.py "私の名前は、山田太郎です。私の住所は、東京都千代田区永田町1-7-1です。私の電話番号は、03-1234-5678です。" --provider openai
```

### 個人情報の評価（Anthropic Claude 使用）

```bash
python evaluator.py "私の名前は、山田太郎です。私の住所は、東京都千代田区永田町1-7-1です。私の電話番号は、03-1234-5678です。" --provider anthropic
```

### 法人情報の評価

```bash
python evaluator.py "当社の売上高は前年比120%増の50億円を記録しました。主要取引先はA社、B社、C社です。"
```

## 出力例

### サマリー形式の出力

```
# 情報公開法評価結果

## 評価対象
私の名前は、山田太郎です。私の住所は、東京都千代田区永田町1-7-1です。私の電話番号は、03-1234-5678です。

## 評価サマリー
- **評価criteria数**: 6
- **評価完了時刻**: 2025-09-28T20:41:12.377448

## 各不開示事由の評価

### ⚠️ 個人情報保護 (第5条第1号)
- **スコア**: 3/5
- **スコア理由**: 段階1から段階3までの評価では、提供された情報が特定の個人を識別できるものであり、個人のプライバシーや権利利益を害するおそれがあるため、情報公開法第5条第1号に基づく不開示事由に該当すると判断されました。

**段階的評価**:
- 段階1: 個人に関する情報（事業を営む個人の当該事業に関する情報を除く。）か: YES - 提供された情報は、特定の個人である山田太郎氏に関するものであり、氏名、住所、電話番号といった個人を識別できる情報が含まれています。
- 段階2: 特定の個人を識別することができるものか: YES - 提供された情報には、氏名（山田太郎）、住所（東京都千代田区永田町1-7-1）、電話番号（03-1234-5678）が含まれており、これらの情報は特定の個人を識別することが可能です。
...
```

## システム要件

- **Python**: 3.8 以上
- **LLM API**: 以下のいずれか
  - OpenAI API キー（GPT-4 以上推奨）
  - Anthropic API キー（Claude 3.5 Sonnet 以上推奨）
  - AWS 認証情報（Bedrock アクセス権限付き）
- **依存関係**: requirements.txt に記載

## インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd disclosure-evaluator

# 依存関係のインストール
pip install -r requirements.txt

# 環境変数の設定（.envファイルまたは直接設定）
# OpenAI を使用する場合
export OPENAI_API_KEY="your-api-key-here"

# Anthropic を使用する場合
export ANTHROPIC_API_KEY="your-api-key-here"

# AWS Bedrock を使用する場合
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_REGION="us-east-1"
```

### 環境変数の設定

認証情報は環境変数で管理されます。プロジェクトルートに `.env` ファイルを作成して設定することも可能です：

```bash
# .envファイルの作成（テンプレートからコピー）
cp .env.example .env

# .envファイルを編集して、使用するプロバイダーのAPIキーを設定してください
# 以下のいずれか1つ以上を設定する必要があります：

# OpenAI を使用する場合
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic を使用する場合
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# AWS Bedrock を使用する場合
AWS_ACCESS_KEY_ID=your_aws_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here
AWS_REGION=us-east-1
```

**注意**: `.env` ファイルには機密情報が含まれるため、バージョン管理には含めないでください。

## アーキテクチャ

### 主要コンポーネント

- **LLMProvider**: LLM プロバイダーの抽象基底クラス
  - `OpenAIProvider`: OpenAI API 実装
  - `AnthropicProvider`: Anthropic API 実装
  - `BedrockAnthropicProvider`: AWS Bedrock Claude 実装
  - `BedrockNovaProvider`: AWS Bedrock Nova 実装
- **StepEvaluator**: 個別評価ステップの実行
- **CriterionEvaluator**: 単一 criteria の評価管理
- **ResultAggregator**: 評価結果の処理
- **DisclosureEvaluator**: メイン評価オーケストレーター

### 設計原則

- **Provider Pattern**: LLM プロバイダーの切り替え可能な設計
- **Pydantic Models**: 型安全なデータ検証
- **Structured Logging**: 構造化ログによる追跡可能性
- **Error Handling**: 包括的なエラーハンドリングとフォールバック

## 技術仕様

### 評価フロー

1. **入力検証**: 評価対象情報の検証
2. **プロバイダー選択**: 指定された LLM プロバイダーの初期化
3. **段階的評価**: 各 criteria について複数段階で評価
4. **LLM 分析**: 選択されたモデルによる詳細な法的分析
5. **結果集約**: 評価結果の構造化
6. **出力生成**: JSON/サマリー形式での出力

### ログ機能

- **構造化ログ**: JSON 形式でのログ出力（`logs/` ディレクトリに保存）
- **相関 ID**: リクエスト追跡のための相関 ID
- **エラーハンドリング**: 包括的なエラー処理とフォールバック
- **タイムスタンプ付きログ**: 各評価実行ごとに独立したログファイル生成

### プロンプト管理

評価に使用される LLM プロンプトは `prompts/` ディレクトリで管理されています：

- **system_step_evaluation.md**: 各評価ステップの実行に使用するシステムプロンプト
- **system_score_reasoning.md**: 最終スコア算出の推論に使用するシステムプロンプト
- **user_step_template.md**: ユーザープロンプトのテンプレート

## プロジェクト構造

```
disclosure-evaluator/
├── criteria/                          # 評価基準定義ファイル
│   ├── disclosure_evaluation_criteria.json
│   ├── administrative_information_non_disclosure.json
│   └── multi-criteria_decision_making_framework.json
├── prompts/                          # LLM プロンプトテンプレート
│   ├── system_step_evaluation.md    # ステップ評価用システムプロンプト
│   ├── system_score_reasoning.md    # スコア推論用システムプロンプト
│   └── user_step_template.md        # ユーザープロンプトテンプレート
├── logs/                             # 評価ログファイル（自動生成）
├── outputs/                          # 評価結果出力ファイル（自動生成）
├── evaluator.py                      # メイン評価スクリプト
├── config.json                       # システム設定ファイル
├── requirements.txt                  # Python 依存関係
├── .env.example                      # 環境変数テンプレート
└── README.md                         # このファイル
```
