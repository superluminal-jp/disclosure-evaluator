# CLI コマンド仕様

## 📋 文書情報

| 項目       | 内容               |
| ---------- | ------------------ |
| 文書名     | CLI コマンド仕様   |
| バージョン | 1.0                |
| 作成日     | 2025 年 9 月 28 日 |
| 作成者     | AI 開発チーム      |
| 承認者     | 技術責任者         |
| ステータス | 草案               |

## 🎯 概要

Disclosure Evaluator の CLI インターフェースは、Typer フレームワークを使用して型安全で使いやすいコマンドラインインターフェースを提供します。更新された情報公開法評価基準（詳細な評価プロンプト、段階的評価手順、豊富な具体例）を活用した高度な評価機能を提供します。

## 🏗️ CLI アーキテクチャ

### 1. 基本構造

```python
import typer
from typing import Optional, List
from pathlib import Path

app = typer.Typer(
    name="disclosure-evaluator",
    help="構造化マルチプロバイダー LLM 評価システム",
    add_completion=False
)

# サブコマンドの登録
app.add_typer(evaluate_app, name="evaluate", help="評価コマンド")
app.add_typer(analyze_app, name="analyze", help="分析コマンド")
app.add_typer(config_app, name="config", help="設定コマンド")
app.add_typer(report_app, name="report", help="レポートコマンド")
```

### 2. コマンド階層

```
disclosure-evaluator
├── evaluate
│   ├── single          # 単一評価
│   ├── batch           # バッチ評価
│   └── legal            # 法的評価
├── analyze
│   ├── statistics      # 統計分析
│   ├── compare         # 比較分析
│   └── trends          # トレンド分析
├── config
│   ├── provider        # プロバイダー設定
│   ├── metrics         # 評価指標設定
│   └── security        # セキュリティ設定
└── report
    ├── generate        # レポート生成
    └── export          # データエクスポート
```

## 📝 評価コマンド

### 1. 単一評価コマンド

```python
@app.command("single")
def evaluate_single(
    prompt: str = typer.Argument(..., help="評価対象のプロンプト"),
    response: str = typer.Argument(..., help="評価対象の応答"),
    provider: str = typer.Option("openai", "--provider", "-p", help="使用するLLMプロバイダー"),
    criteria: str = typer.Option("standard", "--criteria", "-c", help="評価基準"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="出力ファイルパス"),
    format: str = typer.Option("json", "--format", "-f", help="出力形式 (json, yaml, csv)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="詳細出力"),
    config_dir: Optional[Path] = typer.Option(None, "--config-dir", help="設定ディレクトリ")
):
    """単一のプロンプト・応答ペアを評価する"""
    try:
        # 設定の読み込み
        settings = load_settings(config_dir)

        # 評価の実行
        result = execute_single_evaluation(
            prompt=prompt,
            response=response,
            provider=provider,
            criteria=criteria,
            settings=settings
        )

        # 結果の出力
        output_result(result, output, format, verbose)

    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)
```

**使用例**:

```bash
# 基本的な単一評価
disclosure-evaluator evaluate single "What is AI?" "AI is artificial intelligence."

# プロバイダーと基準を指定
disclosure-evaluator evaluate single "What is AI?" "AI is artificial intelligence." \
  --provider anthropic --criteria legal

# 結果をファイルに出力
disclosure-evaluator evaluate single "What is AI?" "AI is artificial intelligence." \
  --output results.json --format json
```

### 2. バッチ評価コマンド

```python
@app.command("batch")
def evaluate_batch(
    input_file: Path = typer.Argument(..., help="入力ファイル (CSV, JSON, Excel)"),
    provider: str = typer.Option("openai", "--provider", "-p", help="使用するLLMプロバイダー"),
    criteria: str = typer.Option("standard", "--criteria", "-c", help="評価基準"),
    output_dir: Path = typer.Option(Path("./output"), "--output-dir", "-o", help="出力ディレクトリ"),
    parallel: int = typer.Option(5, "--parallel", help="並列実行数"),
    timeout: int = typer.Option(300, "--timeout", help="タイムアウト時間（秒）"),
    resume: bool = typer.Option(False, "--resume", help="中断された評価を再開"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="詳細出力")
):
    """複数のプロンプト・応答ペアをバッチで評価する"""
    try:
        # 入力ファイルの検証
        validate_input_file(input_file)

        # バッチ評価の実行
        results = execute_batch_evaluation(
            input_file=input_file,
            provider=provider,
            criteria=criteria,
            parallel=parallel,
            timeout=timeout,
            resume=resume,
            settings=load_settings()
        )

        # 結果の保存
        save_batch_results(results, output_dir)

        typer.echo(f"バッチ評価完了: {len(results)}件の評価を実行")

    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)
```

**使用例**:

```bash
# CSVファイルでのバッチ評価
disclosure-evaluator evaluate batch data.csv --provider openai --criteria standard

# 並列実行数を指定
disclosure-evaluator evaluate batch data.csv --parallel 10 --timeout 600

# 中断された評価を再開
disclosure-evaluator evaluate batch data.csv --resume
```

### 3. 情報公開法評価コマンド

```python
@app.command("legal")
def evaluate_legal(
    information: str = typer.Argument(..., help="評価対象の行政情報"),
    context: Optional[str] = typer.Option(None, "--context", help="追加のコンテキスト情報"),
    provider: str = typer.Option("openai", "--provider", "-p", help="使用するLLMプロバイダー"),
    articles: Optional[List[str]] = typer.Option(None, "--articles", help="評価する条文（第1号〜第6号）"),
    detailed: bool = typer.Option(False, "--detailed", help="詳細な段階的評価プロセスを実行"),
    partial_disclosure: bool = typer.Option(False, "--partial", help="部分開示の可能性を検討"),
    expert_review: bool = typer.Option(False, "--expert", help="専門機関との協議を考慮"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="出力ファイルパス"),
    format: str = typer.Option("json", "--format", "-f", help="出力形式"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="詳細出力")
):
    """情報公開法に基づく行政情報の開示判断を実行（詳細な段階的評価プロセス対応）"""
    try:
        # 情報公開法評価の実行（更新された詳細基準を使用）
        result = execute_legal_evaluation(
            information=information,
            context=context,
            provider=provider,
            articles=articles or ["1", "2", "3", "4", "5", "6"],
            detailed_evaluation=detailed,
            consider_partial_disclosure=partial_disclosure,
            expert_review_required=expert_review,
            settings=load_settings()
        )

        # 結果の出力
        output_legal_result(result, output, format, verbose)

    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)
```

**使用例**:

```bash
# 基本的な情報公開法評価
disclosure-evaluator evaluate legal "申請者: 田中太郎, 住所: 東京都新宿区1-1-1"

# 詳細な段階的評価プロセスを実行
disclosure-evaluator evaluate legal "個人情報を含む文書" --detailed --verbose

# 部分開示の可能性を検討
disclosure-evaluator evaluate legal "法人の内部情報" --partial --articles 2 3

# 専門機関との協議を考慮した評価
disclosure-evaluator evaluate legal "国家安全保障関連情報" --expert --detailed

# 特定の条文のみを詳細評価
disclosure-evaluator evaluate legal "内部審議記録" --articles 5 --detailed --partial
```

## 📊 分析コマンド

### 1. 統計分析コマンド

```python
@app.command("statistics")
def analyze_statistics(
    input_dir: Path = typer.Argument(..., help="評価結果ディレクトリ"),
    metrics: Optional[List[str]] = typer.Option(None, "--metrics", help="分析する評価指標"),
    provider: Optional[str] = typer.Option(None, "--provider", help="特定プロバイダーの結果のみ"),
    time_range: Optional[str] = typer.Option(None, "--time-range", help="時間範囲 (例: 2024-01-01:2024-12-31)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="出力ファイルパス"),
    format: str = typer.Option("json", "--format", "-f", help="出力形式"),
    include_visualizations: bool = typer.Option(False, "--include-visualizations", help="可視化を含める")
):
    """評価結果の統計分析を実行"""
    try:
        # 統計分析の実行
        statistics = execute_statistical_analysis(
            input_dir=input_dir,
            metrics=metrics,
            provider=provider,
            time_range=time_range,
            settings=load_settings()
        )

        # 結果の保存
        save_statistics(statistics, output, format, include_visualizations)

        typer.echo(f"統計分析完了: {statistics.summary}")

    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)
```

### 2. 比較分析コマンド

```python
@app.command("compare")
def analyze_compare(
    baseline_dir: Path = typer.Argument(..., help="ベースライン評価結果ディレクトリ"),
    comparison_dir: Path = typer.Argument(..., help="比較対象評価結果ディレクトリ"),
    metrics: Optional[List[str]] = typer.Option(None, "--metrics", help="比較する評価指標"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="出力ファイルパス"),
    format: str = typer.Option("json", "--format", "-f", help="出力形式"),
    include_visualizations: bool = typer.Option(False, "--include-visualizations", help="可視化を含める")
):
    """評価結果の比較分析を実行"""
    try:
        # 比較分析の実行
        comparison = execute_comparison_analysis(
            baseline_dir=baseline_dir,
            comparison_dir=comparison_dir,
            metrics=metrics,
            settings=load_settings()
        )

        # 結果の保存
        save_comparison(comparison, output, format, include_visualizations)

        typer.echo(f"比較分析完了: {comparison.summary}")

    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)
```

## ⚙️ 設定コマンド

### 1. プロバイダー設定コマンド

```python
@app.command("provider")
def config_provider(
    action: str = typer.Argument(..., help="アクション (add, remove, list, test)"),
    name: Optional[str] = typer.Option(None, "--name", help="プロバイダー名"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="APIキー"),
    model: Optional[str] = typer.Option(None, "--model", help="モデル名"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="ベースURL"),
    config_file: Optional[Path] = typer.Option(None, "--config-file", help="設定ファイルパス")
):
    """LLMプロバイダーの設定を管理"""
    try:
        if action == "add":
            add_provider_config(name, api_key, model, base_url, config_file)
            typer.echo(f"プロバイダー '{name}' を追加しました")
        elif action == "remove":
            remove_provider_config(name, config_file)
            typer.echo(f"プロバイダー '{name}' を削除しました")
        elif action == "list":
            list_provider_configs(config_file)
        elif action == "test":
            test_provider_connection(name, config_file)
        else:
            typer.echo(f"不明なアクション: {action}", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)
```

**使用例**:

```bash
# プロバイダーを追加
disclosure-evaluator config provider add --name openai --api-key sk-xxx --model gpt-4

# プロバイダー一覧を表示
disclosure-evaluator config provider list

# プロバイダー接続をテスト
disclosure-evaluator config provider test --name openai
```

### 2. 評価指標設定コマンド

```python
@app.command("metrics")
def config_metrics(
    action: str = typer.Argument(..., help="アクション (add, remove, list, update)"),
    metric_id: Optional[str] = typer.Option(None, "--id", help="評価指標ID"),
    name: Optional[str] = typer.Option(None, "--name", help="評価指標名"),
    description: Optional[str] = typer.Option(None, "--description", help="説明"),
    weight: Optional[float] = typer.Option(None, "--weight", help="重み"),
    config_file: Optional[Path] = typer.Option(None, "--config-file", help="設定ファイルパス")
):
    """評価指標の設定を管理"""
    try:
        if action == "add":
            add_metric_config(metric_id, name, description, weight, config_file)
            typer.echo(f"評価指標 '{name}' を追加しました")
        elif action == "remove":
            remove_metric_config(metric_id, config_file)
            typer.echo(f"評価指標 '{metric_id}' を削除しました")
        elif action == "list":
            list_metric_configs(config_file)
        elif action == "update":
            update_metric_config(metric_id, name, description, weight, config_file)
            typer.echo(f"評価指標 '{metric_id}' を更新しました")
        else:
            typer.echo(f"不明なアクション: {action}", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)
```

## 📊 レポートコマンド

### 1. レポート生成コマンド

```python
@app.command("generate")
def report_generate(
    input_dir: Path = typer.Argument(..., help="評価結果ディレクトリ"),
    template: str = typer.Option("standard", "--template", "-t", help="レポートテンプレート"),
    output: Path = typer.Option(Path("./report.html"), "--output", "-o", help="出力ファイルパス"),
    format: str = typer.Option("html", "--format", "-f", help="出力形式 (html, pdf, markdown)"),
    include_charts: bool = typer.Option(True, "--include-charts", help="チャートを含める"),
    include_details: bool = typer.Option(True, "--include-details", help="詳細情報を含める"),
    time_range: Optional[str] = typer.Option(None, "--time-range", help="時間範囲")
):
    """評価結果からレポートを生成"""
    try:
        # レポート生成の実行
        report = generate_report(
            input_dir=input_dir,
            template=template,
            include_charts=include_charts,
            include_details=include_details,
            time_range=time_range,
            settings=load_settings()
        )

        # レポートの保存
        save_report(report, output, format)

        typer.echo(f"レポート生成完了: {output}")

    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)
```

**使用例**:

```bash
# 基本的なレポート生成
disclosure-evaluator report generate ./results --output report.html

# PDF形式でレポート生成
disclosure-evaluator report generate ./results --format pdf --output report.pdf

# チャート付きレポート生成
disclosure-evaluator report generate ./results --include-charts --template detailed
```

## 🔧 実装ガイドライン

### 1. エラーハンドリング

```python
def handle_cli_error(func):
    """CLIエラーハンドリングデコレーター"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            typer.echo(f"検証エラー: {e}", err=True)
            raise typer.Exit(1)
        except ProviderConnectionError as e:
            typer.echo(f"プロバイダー接続エラー: {e}", err=True)
            raise typer.Exit(2)
        except EvaluationError as e:
            typer.echo(f"評価エラー: {e}", err=True)
            raise typer.Exit(3)
        except Exception as e:
            typer.echo(f"予期しないエラー: {e}", err=True)
            raise typer.Exit(99)
    return wrapper
```

### 2. 設定管理

```python
def load_settings(config_dir: Optional[Path] = None) -> Settings:
    """設定を読み込み"""
    try:
        if config_dir:
            settings = Settings(config_dir=config_dir)
        else:
            settings = Settings()
        return settings
    except Exception as e:
        typer.echo(f"設定読み込みエラー: {e}", err=True)
        raise typer.Exit(1)
```

### 3. 出力フォーマット

```python
def output_result(result: EvaluationResult, output: Optional[Path], format: str, verbose: bool):
    """結果を出力"""
    if format == "json":
        content = result.to_json(indent=2)
    elif format == "yaml":
        content = result.to_yaml()
    elif format == "csv":
        content = result.to_csv()
    else:
        content = result.to_text(verbose=verbose)

    if output:
        output.write_text(content, encoding='utf-8')
        typer.echo(f"結果を {output} に保存しました")
    else:
        typer.echo(content)
```

## 📊 パフォーマンス考慮事項

### 1. 並列処理

- **バッチ評価**: 並列実行による高速化
- **プロセス分離**: 各評価の独立実行
- **リソース管理**: CPU・メモリ使用量の制御

### 2. 進捗表示

```python
def show_progress_bar(total: int, current: int, description: str = ""):
    """進捗バーを表示"""
    with typer.progressbar(length=total, label=description) as progress:
        progress.update(current)
```

### 3. ログ出力

```python
def setup_logging(verbose: bool, log_file: Optional[Path] = None):
    """ログ設定"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
```

---

_この CLI コマンド仕様により、Disclosure Evaluator は使いやすく強力なコマンドラインインターフェースを提供します。_
