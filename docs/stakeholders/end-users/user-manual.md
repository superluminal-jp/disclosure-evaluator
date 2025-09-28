# ユーザーマニュアル

## 📋 文書情報

| 項目       | 内容                                    |
| ---------- | --------------------------------------- |
| 文書名     | Disclosure Evaluator ユーザーマニュアル |
| バージョン | 1.0                                     |
| 作成日     | 2025 年 9 月 28 日                      |
| 作成者     | AI 開発チーム                           |
| 承認者     | プロダクトオーナー                      |
| ステータス | 最終版                                  |

## 🎯 概要

Disclosure Evaluator は、情報公開法準拠の LLM 評価システムです。このマニュアルでは、システムの基本的な使用方法から高度な機能まで、段階的に説明します。

## 🚀 はじめに

### 1. システムの概要

Disclosure Evaluator は以下の機能を提供します：

- **単一評価**: 個別のプロンプト・応答ペアの評価
- **バッチ評価**: 複数の評価を一括実行
- **情報公開法評価**: 情報公開法第 5 条に基づく法的評価
- **分析・レポート**: 評価結果の分析とレポート生成

### 2. 前提条件

- **Python**: 3.11 以上がインストールされていること
- **インターネット接続**: LLM プロバイダーへの接続
- **API キー**: 使用する LLM プロバイダーの API キー

## 📥 インストール

### 1. システムのインストール

```bash
# パッケージのインストール
pip install disclosure-evaluator

# バージョンの確認
disclosure-evaluator --version
```

### 2. 初期設定

```bash
# 設定ディレクトリの作成
mkdir -p ~/.disclosure-evaluator

# 設定ファイルの作成
disclosure-evaluator config init
```

## ⚙️ 設定

### 1. プロバイダー設定

#### OpenAI 設定

```bash
# OpenAIプロバイダーの追加
disclosure-evaluator config provider add \
  --name openai \
  --api-key sk-your-api-key \
  --model gpt-4
```

#### Anthropic 設定

```bash
# Anthropicプロバイダーの追加
disclosure-evaluator config provider add \
  --name anthropic \
  --api-key sk-ant-your-api-key \
  --model claude-3-sonnet
```

### 2. 設定の確認

```bash
# 設定されたプロバイダーの一覧表示
disclosure-evaluator config provider list

# 特定プロバイダーの接続テスト
disclosure-evaluator config provider test --name openai
```

## 📝 基本的な使用方法

### 1. 単一評価の実行

#### 基本的な単一評価

```bash
# 基本的な評価
disclosure-evaluator evaluate single \
  "What is artificial intelligence?" \
  "Artificial intelligence is a branch of computer science..."
```

#### プロバイダーと基準の指定

```bash
# プロバイダーと評価基準を指定
disclosure-evaluator evaluate single \
  "What is artificial intelligence?" \
  "Artificial intelligence is a branch of computer science..." \
  --provider openai \
  --criteria standard
```

#### 結果の保存

```bash
# 結果をファイルに保存
disclosure-evaluator evaluate single \
  "What is artificial intelligence?" \
  "Artificial intelligence is a branch of computer science..." \
  --output results.json \
  --format json
```

### 2. バッチ評価の実行

#### CSV ファイルでのバッチ評価

```bash
# CSVファイルでのバッチ評価
disclosure-evaluator evaluate batch data.csv \
  --provider openai \
  --criteria standard \
  --output-dir ./results
```

#### CSV ファイルの形式

```csv
prompt,response
"What is AI?","AI is artificial intelligence."
"What is machine learning?","Machine learning is a subset of AI."
```

#### 並列実行の設定

```bash
# 並列実行数を指定
disclosure-evaluator evaluate batch data.csv \
  --parallel 10 \
  --timeout 600
```

### 3. 情報公開法評価

#### 基本的な情報公開法評価

```bash
# 行政情報の評価
disclosure-evaluator evaluate legal \
  "申請者: 田中太郎, 住所: 東京都新宿区1-1-1"
```

#### 特定条文の評価

```bash
# 特定の条文のみを評価
disclosure-evaluator evaluate legal \
  "法人の内部情報" \
  --articles 2 3
```

#### 詳細な結果の出力

```bash
# 詳細な結果をYAML形式で出力
disclosure-evaluator evaluate legal \
  "個人情報を含む文書" \
  --format yaml \
  --verbose
```

## 📊 分析・レポート機能

### 1. 統計分析

#### 基本的な統計分析

```bash
# 評価結果の統計分析
disclosure-evaluator analyze statistics ./results \
  --output statistics.json
```

#### 特定指標の分析

```bash
# 特定の評価指標のみを分析
disclosure-evaluator analyze statistics ./results \
  --metrics accuracy relevance \
  --include-visualizations
```

### 2. 比較分析

#### ベースラインとの比較

```bash
# ベースラインとの比較分析
disclosure-evaluator analyze compare \
  ./baseline_results \
  ./new_results \
  --output comparison.json
```

### 3. レポート生成

#### HTML レポートの生成

```bash
# HTMLレポートの生成
disclosure-evaluator report generate ./results \
  --output report.html \
  --include-visualizations
```

#### PDF レポートの生成

```bash
# PDFレポートの生成
disclosure-evaluator report generate ./results \
  --format pdf \
  --output report.pdf \
  --template detailed
```

## 🔧 高度な使用方法

### 1. カスタム評価基準

#### 評価基準の追加

```bash
# カスタム評価基準の追加
disclosure-evaluator config metrics add \
  --id custom_metric \
  --name "カスタム評価指標" \
  --description "独自の評価基準" \
  --weight 0.3
```

#### 評価基準の確認

```bash
# 設定された評価基準の一覧表示
disclosure-evaluator config metrics list
```

### 2. 環境変数の設定

#### API キーの環境変数設定

```bash
# 環境変数でのAPIキー設定
export OPENAI_API_KEY="sk-your-api-key"
export ANTHROPIC_API_KEY="sk-ant-your-api-key"

# 環境変数を使用した評価
disclosure-evaluator evaluate single "test" "response"
```

### 3. 設定ファイルの直接編集

#### 設定ファイルの場所

```bash
# 設定ファイルの場所
~/.disclosure-evaluator/settings.toml
```

#### 設定ファイルの例

```toml
[application]
name = "Disclosure Evaluator"
version = "1.0.0"
debug = false
log_level = "INFO"

[evaluation]
default_provider = "openai"
max_parallel_evaluations = 5
evaluation_timeout = 300

[providers.openai]
api_key = "sk-your-api-key"
model = "gpt-4"
base_url = "https://api.openai.com/v1"
timeout = 30
max_tokens = 4000
temperature = 0.1

[providers.anthropic]
api_key = "sk-ant-your-api-key"
model = "claude-3-sonnet"
timeout = 30
max_tokens = 4000
temperature = 0.1
```

## 🚨 トラブルシューティング

### 1. よくある問題

#### 接続エラー

```bash
# プロバイダー接続の確認
disclosure-evaluator config provider test --name openai

# 詳細なエラー情報の表示
disclosure-evaluator evaluate single "test" "response" --verbose
```

#### 認証エラー

```bash
# APIキーの確認
disclosure-evaluator config provider list

# APIキーの再設定
disclosure-evaluator config provider add --name openai --api-key sk-new-key
```

#### タイムアウトエラー

```bash
# タイムアウト時間の延長
disclosure-evaluator evaluate single "test" "response" --timeout 600
```

### 2. ログの確認

#### ログファイルの場所

```bash
# ログファイルの確認
tail -f ~/.disclosure-evaluator/logs/disclosure-evaluator.log
```

#### デバッグモードの有効化

```bash
# デバッグモードでの実行
disclosure-evaluator evaluate single "test" "response" --debug
```

### 3. パフォーマンスの問題

#### メモリ使用量の確認

```bash
# メモリ使用量の監視
disclosure-evaluator evaluate batch data.csv --monitor-memory
```

#### 並列実行数の調整

```bash
# 並列実行数の削減
disclosure-evaluator evaluate batch data.csv --parallel 2
```

## 📚 参考情報

### 1. コマンドリファレンス

#### 全コマンドの一覧

```bash
# 全コマンドの表示
disclosure-evaluator --help

# 特定コマンドのヘルプ
disclosure-evaluator evaluate --help
disclosure-evaluator analyze --help
disclosure-evaluator config --help
disclosure-evaluator report --help
```

### 2. 設定オプション

#### 全設定オプションの一覧

```bash
# 設定オプションの表示
disclosure-evaluator config --help
```

### 3. 出力形式

#### サポートされている出力形式

- **JSON**: 構造化データ（デフォルト）
- **YAML**: 人間が読みやすい形式
- **CSV**: 表計算ソフトで開ける形式
- **HTML**: ブラウザで表示可能な形式
- **PDF**: 印刷可能な形式

## 📞 サポート

### 1. ヘルプの取得

```bash
# ヘルプの表示
disclosure-evaluator --help
disclosure-evaluator evaluate --help
disclosure-evaluator analyze --help
```

### 2. ドキュメント

- **オンラインドキュメント**: https://docs.disclosure-evaluator.com
- **API 仕様書**: https://api.disclosure-evaluator.com/docs
- **サンプルコード**: https://github.com/disclosure-evaluator/examples

### 3. コミュニティ

- **GitHub Issues**: バグ報告・機能要求
- **Discord**: ユーザーコミュニティ
- **メール**: support@disclosure-evaluator.com

## 🔄 アップデート

### 1. バージョン確認

```bash
# 現在のバージョン確認
disclosure-evaluator --version
```

### 2. アップデート

```bash
# 最新版へのアップデート
pip install --upgrade disclosure-evaluator
```

### 3. 設定の移行

```bash
# 設定の移行
disclosure-evaluator config migrate
```

---

_このユーザーマニュアルにより、Disclosure Evaluator を効果的に活用できます。_
