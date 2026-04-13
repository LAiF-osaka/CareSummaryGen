## プロジェクト概要

CareSummaryGen — 医療記録（看護記録・退院サマリー等）から AI を用いて看護サマリーを自動生成するツール。
LLM には gpt-oss-120b（Ollama 経由）を使用する。

技術スタック: Python 3.12 + Flask + LangChain + Ollama (gpt-oss-120b) / uv 依存管理

## 開発コマンド

```bash
uv sync                                 # 依存インストール
uv run python app.py                    # Flask サーバー起動 (port 5000)
uv run python client.py                 # クライアント実行
```

## アーキテクチャ

```
医療記録（XML/TXT）
  → utils/extract_data.py       # データ抽出・前処理
  → Flask サーバー (app.py)      # LLM リクエスト受信
  → Ollama (gpt-oss-120b)        # サマリー生成
  → client.py                    # 結果取得
```

## ディレクトリ構造

| ディレクトリ/ファイル | 役割 |
|---|---|
| `app.py` | Flask サーバー（Ollama gpt-oss-120b 経由でサマリー生成） |
| `client.py` | クライアント（サーバーにリクエスト送信・結果保存） |
| `instructions_inputs.json` | 質問テンプレート設定 |
| `utils/` | 汎用ユーティリティ（データ抽出・前処理） |
| `data/` | サンプル医療記録データ |
| `old/` | 旧バックエンド（RAG, カスタムモデル, OpenAI）のバックアップ（.gitignore 対象） |

## 環境変数

`.env` で管理（`.gitignore` 対象）。

```
HF_TOKEN=<HuggingFace トークン>
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_BASE_URL=http://localhost:11434
HOSPITAL=hanwa
```

## 主要依存ライブラリ

- **LLM**: LangChain + Ollama (gpt-oss-120b)
- **Web**: Flask
- **計算**: PyTorch (CUDA 12.4)
- **データ**: Tiktoken, Pydantic

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
