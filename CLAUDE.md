## プロジェクト概要

CareSummaryGen — 医療記録（看護記録・退院サマリー等）から AI を用いて看護サマリーを自動生成するツール。
RAG（検索拡張生成）、ファインチューニング済みモデル、Ollama ローカル LLM の3パターンに対応する。

技術スタック: Python 3.12 + Flask + LangChain + ChromaDB + HuggingFace Transformers / uv 依存管理

## 開発コマンド

```bash
uv sync                                 # 依存インストール
uv run python app.py                    # RAG サーバー起動 (port 5000)
uv run python app_custom.py             # カスタムモデルサーバー起動 (port 5000)
uv run python app_ollama.py             # Ollama サーバー起動 (port 5000)
uv run python request.py                # RAG クライアント実行
uv run python request_custom.py         # カスタムモデルクライアント実行
uv run python run_ollama.py             # Ollama クライアント実行
uv run python store_vector.py           # ChromaDB ベクトル更新
chroma run --path ./vectorDB            # ChromaDB 起動 (port 8000)
```

## アーキテクチャ

```
医療記録（XML/TXT）
  → utils/extract_data.py       # データ抽出・前処理
  → utils/extract_summary.py    # サマリー抽出メイン処理
  → Flask サーバー (app.py)      # LLM リクエスト受信
  → LangChain + ChromaDB         # RAG 検索（ベクトル埋め込み）
  → HuggingFace LLM              # サマリー生成
  → request.py                   # 結果取得
```

## ディレクトリ構造

| ディレクトリ/ファイル | 役割 |
|---|---|
| `app.py` | RAG 対応 Flask サーバー（メイン） |
| `app_custom.py` | カスタムファインチューニングモデル用サーバー |
| `app_ollama.py` | Ollama 統合サーバー |
| `request.py` / `request_custom.py` / `request_openai.py` | 各種クライアント |
| `store_vector.py` | ChromaDB ベクトル保存 |
| `utils/` | データ抽出・前処理・サマリー生成ユーティリティ |
| `data/` | サンプル医療記録データ |
| `instructions_inputs.json` | 質問テンプレート設定 |

## 環境変数

`.env` で管理（`.gitignore` 対象）。RAG 実行には ChromaDB サーバー（port 8000）の起動が必要。

## 主要依存ライブラリ

- **LLM**: HuggingFace Transformers, Sentence Transformers, Accelerate, BitsAndBytes (4bit量子化), PEFT
- **RAG**: LangChain, ChromaDB, LangChain Text Splitters
- **Web**: Flask
- **計算**: PyTorch (CUDA 12.4), TensorFlow
- **データ**: Pandas, Unstructured, Pydantic, Tiktoken

## 開発ツール

- **パッケージ管理**: uv
- **フォーマッター**: Black (line-length=79)
- **リンター**: Flake8
- **型チェック**: MyPy
- **インポート整理**: Isort

## Definition of Done

- [ ] 既存の動作が壊れていないこと（手動確認）
- [ ] コードレビュー完了（Critical ゼロ）
- [ ] 新規・変更コードに適切な docstring・型アノテーションがあること

## Claude Code エージェント & スキル

スキル（`.claude/skills/`）はメイン会話内で `/` コマンドとして実行。CLAUDE.md のルールが適用される。
エージェント（`.claude/agents/`）は独立コンテキストで Task ツール経由実行。CLAUDE.md は見えない。

- スキル一覧: `/feature`, `/bugfix`, `/review`, `/spec`, `/git-manager`, `/security-audit`, `/ai-research`, `/postmortem`, `/refactor`, `/pr`
- エージェント一覧: `dev-orchestrator` が開発チームを統括
- **エージェントは CLAUDE.md を読めない**: 従うべきルールは各エージェント `.md` に直接記載が必要

## ルールファイル（`.claude/rules/`）

| ファイル | 内容 | ロード条件 |
|---|---|---|
| `workflow.md` | 実装ワークフロー・基本原則 | 常時 |
| `coding-style.md` | Python Docstring 規約（Google Style） | 常時 |
