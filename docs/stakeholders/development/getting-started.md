# はじめに - 開発者向けガイド

## 📋 文書情報

| 項目       | 内容                          |
| ---------- | ----------------------------- |
| 文書名     | Disclosure Evaluator はじめに |
| バージョン | 1.0                           |
| 作成日     | 2025 年 9 月 28 日            |
| 作成者     | AI 開発チーム                 |
| 承認者     | 技術責任者                    |
| ステータス | 草案                          |

## 🎯 概要

Disclosure Evaluator は、情報公開法準拠の LLM 評価システムです。このガイドでは、開発者がプロジェクトに参加し、効果的に貢献するための手順を説明します。

## 🚀 クイックスタート

### 1. 前提条件

- **Python**: 3.11 以上
- **Git**: 2.30 以上
- **Docker**: 20.10 以上（オプション）
- **IDE**: VS Code、PyCharm、または同等の IDE

### 2. 環境セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/your-org/disclosure-evaluator.git
cd disclosure-evaluator

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements-dev.txt

# 設定ファイルのコピー
cp .env.example .env
cp settings.example.toml settings.toml
```

### 3. 初回実行

```bash
# システムの動作確認
disclosure-evaluator --version

# ヘルプの表示
disclosure-evaluator --help

# 設定の確認
disclosure-evaluator config provider list
```

## 🏗️ プロジェクト構造

```
disclosure-evaluator/
├── src/                          # ソースコード
│   ├── disclosure_evaluator/         # メインパッケージ
│   │   ├── domain/              # ドメイン層
│   │   │   ├── entities/        # エンティティ
│   │   │   ├── value_objects/   # 値オブジェクト
│   │   │   ├── services/        # ドメインサービス
│   │   │   └── events/          # ドメインイベント
│   │   ├── application/         # アプリケーション層
│   │   │   ├── services/        # アプリケーションサービス
│   │   │   ├── use_cases/       # ユースケース
│   │   │   └── dto/             # データ転送オブジェクト
│   │   ├── infrastructure/      # インフラストラクチャ層
│   │   │   ├── adapters/        # アダプター
│   │   │   ├── repositories/     # リポジトリ実装
│   │   │   └── external/        # 外部システム統合
│   │   └── presentation/        # プレゼンテーション層
│   │       ├── cli/             # CLI実装
│   │       └── api/             # API実装
│   └── tests/                   # テストコード
│       ├── unit/                # 単体テスト
│       ├── integration/         # 統合テスト
│       └── e2e/                 # E2Eテスト
├── docs/                        # ドキュメント
├── scripts/                     # スクリプト
├── requirements.txt             # 本番依存関係
├── requirements-dev.txt         # 開発依存関係
├── pyproject.toml               # プロジェクト設定
└── README.md                    # プロジェクト概要
```

## 🔧 開発環境設定

### 1. IDE 設定

#### VS Code 設定

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  }
}
```

#### PyCharm 設定

1. **プロジェクト設定**: Python 3.11 を選択
2. **仮想環境**: プロジェクトの venv を選択
3. **コードスタイル**: Black フォーマッターを設定
4. **テスト**: pytest をテストランナーに設定

### 2. 開発ツール設定

```bash
# プリコミットフックの設定
pre-commit install

# コードフォーマッターの設定
black --line-length 88 src/ tests/

# リンターの設定
ruff check src/ tests/

# 型チェッカーの設定
mypy src/
```

## 🧪 テスト実行

### 1. 単体テスト

```bash
# 全テストの実行
pytest tests/unit/

# 特定のテストファイルの実行
pytest tests/unit/test_evaluation.py

# 詳細出力での実行
pytest tests/unit/ -v

# カバレッジ付きでの実行
pytest tests/unit/ --cov=src --cov-report=html
```

### 2. 統合テスト

```bash
# 統合テストの実行
pytest tests/integration/

# 特定のプロバイダーでのテスト
pytest tests/integration/ -k "openai"

# 並列実行
pytest tests/integration/ -n auto
```

### 3. E2E テスト

```bash
# E2Eテストの実行
pytest tests/e2e/

# 特定のシナリオの実行
pytest tests/e2e/test_cli_evaluation.py
```

## 🔨 開発ワークフロー

### 1. ブランチ戦略

```bash
# 機能開発ブランチの作成
git checkout -b feature/evaluation-engine

# バグ修正ブランチの作成
git checkout -b bugfix/score-calculation

# ホットフィックスブランチの作成
git checkout -b hotfix/security-patch
```

### 2. コミット規約

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 例**:

- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメント更新
- `style`: コードスタイル修正
- `refactor`: リファクタリング
- `test`: テスト追加・修正
- `chore`: その他の変更

**例**:

```bash
git commit -m "feat(evaluation): add legal evaluation engine

- Implement LegalEvaluationEngine class
- Add LegalArticle enum for legal articles
- Add comprehensive test coverage

Closes #123"
```

### 3. プルリクエスト

```bash
# プルリクエストの作成
git push origin feature/evaluation-engine

# プルリクエストテンプレートに従って記入
# - 変更内容の説明
# - テストの実行結果
# - レビュアーの指定
```

## 📚 コーディング標準

### 1. Python コーディング標準

```python
# 型ヒントの使用
def evaluate_response(
    prompt: str,
    response: str,
    criteria: EvaluationCriteria
) -> EvaluationResult:
    """評価を実行する"""
    pass

# ドキュメント文字列
def calculate_score(
    self,
    metric_scores: Dict[str, Score]
) -> float:
    """
    重み付けされた総合スコアを計算する

    Args:
        metric_scores: 評価指標別スコア

    Returns:
        総合スコア

    Raises:
        ValueError: スコアが無効な場合
    """
    pass
```

### 2. エラーハンドリング

```python
# カスタム例外の使用
try:
    result = evaluation_engine.evaluate(prompt, response)
except EvaluationError as e:
    logger.error(f"評価エラー: {e}")
    raise
except Exception as e:
    logger.error(f"予期しないエラー: {e}")
    raise EvaluationError(f"評価に失敗しました: {e}") from e
```

### 3. ログ記録

```python
import logging

logger = logging.getLogger(__name__)

def evaluate_information(self, information: str) -> LegalEvaluationResult:
    """行政情報を評価する"""
    logger.info(
        "情報公開法評価を開始",
        extra={
            "information_length": len(information),
            "evaluation_type": "legal"
        }
    )

    try:
        result = self._execute_evaluation(information)
        logger.info("情報公開法評価完了", extra={"result": result.id})
        return result
    except Exception as e:
        logger.error("情報公開法評価エラー", extra={"error": str(e)})
        raise
```

## 🔍 デバッグ

### 1. ログレベル設定

```python
# 開発環境でのログ設定
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 2. デバッグツール

```python
# デバッガーの使用
import pdb

def evaluate_response(self, prompt: str, response: str) -> Score:
    """評価を実行する"""
    pdb.set_trace()  # ブレークポイント
    # デバッグコード
    pass
```

### 3. プロファイリング

```bash
# プロファイリングの実行
python -m cProfile -o profile.stats src/disclosure_evaluator/cli/main.py evaluate single "test" "response"

# プロファイル結果の表示
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

## 📦 パッケージ管理

### 1. 依存関係の管理

```bash
# 新しい依存関係の追加
pip install new-package
pip freeze > requirements.txt

# 開発依存関係の追加
pip install -r requirements-dev.txt
```

### 2. バージョン管理

```bash
# バージョンの更新
bump2version patch  # パッチバージョン
bump2version minor  # マイナーバージョン
bump2version major  # メジャーバージョン
```

## 🚀 デプロイメント

### 1. ローカル環境

```bash
# ローカルでの実行
disclosure-evaluator evaluate single "test prompt" "test response"
```

### 2. 開発環境

```bash
# 開発環境へのデプロイ
docker build -t disclosure-evaluator:dev .
docker run -p 8000:8000 disclosure-evaluator:dev
```

### 3. 本番環境

```bash
# 本番環境へのデプロイ
docker build -t disclosure-evaluator:latest .
docker push disclosure-evaluator:latest
kubectl apply -f k8s/
```

## 📞 サポート・ヘルプ

### 1. ドキュメント

- **アーキテクチャ**: `docs/architecture/`
- **ドメインモデル**: `docs/domain/`
- **技術仕様**: `docs/technical/`
- **API 仕様**: `docs/api/`

### 2. コミュニティ

- **Slack**: #disclosure-evaluator-dev
- **GitHub Issues**: バグ報告・機能要求
- **Wiki**: 開発者向け情報

### 3. メンタリング

- **ペアプログラミング**: 経験豊富な開発者とのペア
- **コードレビュー**: 定期的なコードレビュー
- **技術相談**: 技術的な質問・相談

## 🎯 次のステップ

### 1. 初回タスク

- [ ] **環境セットアップ**: 開発環境の構築
- [ ] **テスト実行**: 全テストの実行と確認
- [ ] **サンプル実行**: 基本的なコマンドの実行
- [ ] **ドキュメント読了**: 関連ドキュメントの読了

### 2. 学習リソース

- [ ] **DDD 学習**: ドメイン駆動設計の理解
- [ ] **Clean Architecture**: クリーンアーキテクチャの学習
- [ ] **Python 型ヒント**: 型ヒントの活用
- [ ] **テスト駆動開発**: TDD の実践

### 3. 貢献方法

- [ ] **バグ修正**: 既知のバグの修正
- [ ] **機能追加**: 新機能の実装
- [ ] **ドキュメント**: ドキュメントの改善
- [ ] **テスト**: テストカバレッジの向上

---

_このはじめにガイドにより、開発者は Disclosure Evaluator プロジェクトに効果的に参加できます。_
