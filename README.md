# 情報公開法評価システム (Disclosure Evaluator)

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

- **OpenAI**: GPT-4、GPT-5-nano
- **Anthropic**: Claude Sonnet 4
- **AWS Bedrock**: Claude Sonnet 4、Amazon Nova Premier

### バッチ処理機能

- フォルダ単位での一括処理
- 特定ファイルの選択的処理
- 処理状況の追跡と管理
- 失敗した文書の再処理

### 並列処理サポート

- 最大 30 の並列ワーカー
- タイムアウト設定（デフォルト 300 秒）
- リトライ機能（最大 3 回）
- ファイルサイズ制限（デフォルト 50MB）

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
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows
```

3. **依存関係のインストール**

```bash
pip install -r requirements.txt
```

### 必要な依存関係

- pydantic>=2.0.0
- python-dotenv>=1.0.0
- openai>=1.0.0
- anthropic>=0.60.0
- boto3>=1.39.0
- pytest>=7.0.0（テスト用）

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

## 使用方法

### 単一文書の評価

```bash
# 基本的な使用方法
python main.py "評価対象の文書内容" "追加の文脈情報" "出力ファイル名"

# 出力形式を指定
python main.py "文書内容" "文脈" "出力" --format json

# LLMプロバイダーを指定
python main.py "文書内容" "文脈" "出力" --provider anthropic
```

### バッチ処理

```bash
# フォルダ内の全文書を処理
python main.py --batch --folder /path/to/documents

# 特定のファイルを処理
python main.py --batch --documents file1.txt,file2.pdf,file3.docx

# 並列ワーカー数を指定
python main.py --batch --folder /path/to/documents --max-workers 10

# ファイルサイズ制限を設定
python main.py --batch --folder /path/to/documents --file-size-limit 10000000
```

### バッチ処理の管理

```bash
# バッチ処理の状況確認
python main.py --batch-status <batch_id>

# バッチ処理の結果取得
python main.py --batch-results <batch_id> --format json

# 失敗した文書の再処理
python main.py --retry-documents <batch_id> <document_id1,document_id2>
```

## 評価基準

### 情報公開法第 5 条の 6 つの不開示事由

1. **個人情報保護**（第 5 条第 1 号）

   - 個人に関する情報の識別可能性
   - 個人の権利利益への影響
   - 開示例外要件の適用

2. **法人等情報保護**（第 5 条第 2 号）

   - 法人等の競争上の地位への影響
   - 正当な利益の保護
   - 任意提供情報の条件

3. **国家安全保障**（第 5 条第 3 号）

   - 国の安全への影響
   - 国際関係への影響
   - 外交交渉への影響

4. **公共の安全と秩序**（第 5 条第 4 号）

   - 犯罪の予防・鎮圧・捜査への影響
   - 公訴の維持・刑の執行への影響
   - 公共の安全と秩序の維持

5. **内部審議等**（第 5 条第 5 号）

   - 率直な意見交換の保護
   - 意思決定の中立性
   - 国民の混乱防止

6. **行政運営等**（第 5 条第 6 号）
   - 監査・検査・取締り等への影響
   - 契約・交渉・争訟への影響
   - 調査研究・人事管理への影響

### スコアリング基準（1-5 スケール）

- **1**: 明確に不開示事由に該当し、不開示が必要
- **2**: 不開示事由に該当し、不開示が適切
- **3**: 不開示事由の該当性があり、慎重な検討が必要
- **4**: 不開示事由の該当性は低く、開示を検討可能
- **5**: 不開示事由に該当せず、開示が適切

## 開発者向け情報

### プロジェクト構造

```
disclosure-evaluator/
├── main.py                   # メイン評価システム
├── config.json              # システム設定
├── requirements.txt          # 依存関係
├── criteria/
│   └── disclosure_evaluation_criteria.json  # 評価基準
└── venv/                    # 仮想環境
```

### アーキテクチャ概要

- **ConfigManager**: 設定管理
- **LLMProvider**: LLM プロバイダーの抽象化
- **DisclosureEvaluator**: メイン評価ロジック
- **BatchProcessor**: バッチ処理管理
- **EvaluationCriteria**: 評価基準の管理

### 主要コンポーネント

#### LLMProvider クラス

```python
class LLMProvider:
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """LLMからの応答を生成"""
        raise NotImplementedError
```

#### 評価プロセス

1. 文書の前処理と分類
2. 各不開示事由の段階的評価
3. LLM による専門的判断
4. スコアの算出と根拠の生成
5. 総合的な開示判断

### カスタマイズ方法

#### 新しい LLM プロバイダーの追加

```python
class CustomProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # カスタムプロバイダーの初期化

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        # カスタムプロバイダーの実装
        pass
```

#### 評価基準のカスタマイズ

`criteria/disclosure_evaluation_criteria.json`を編集して、評価基準をカスタマイズできます。

## 出力形式

### JSON 形式

```json
{
  "document_id": "doc_001",
  "evaluation_results": {
    "article_5_1": {
      "score": 2,
      "reasoning": "個人情報保護規定に該当する可能性が高い",
      "steps": [
        {
          "step": "個人に関する情報か",
          "result": "YES",
          "reasoning": "氏名、住所等の個人識別情報が含まれている"
        }
      ]
    }
  },
  "overall_decision": "不開示",
  "confidence": 0.85
}
```

### サマリー形式

```
=== 評価結果サマリー ===
文書ID: doc_001
総合判断: 不開示
信頼度: 85%

【個人情報保護】
スコア: 2/5
判断: 不開示の可能性が高い
理由: 個人識別情報が含まれ、開示例外に該当しない
```

### CSV 形式

```csv
document_id,criterion,score,decision,confidence
doc_001,article_5_1,2,non_disclosure,0.85
doc_001,article_5_2,5,disclosure,0.90
```

## トラブルシューティング

### よくある問題

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
