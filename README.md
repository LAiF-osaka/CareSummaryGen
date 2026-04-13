# CareSummaryGen

医療記録から AI を用いて看護サマリーを自動生成するツール。

## セットアップ

```bash
uv sync
```

## 環境変数

`.env` ファイルを作成し、以下を設定:

```
HF_TOKEN=<HuggingFace トークン>
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_BASE_URL=http://localhost:11434
HOSPITAL=hanwa
```

## 実行手順

### 1. Ollama でモデルを起動

```bash
ollama serve
```

### 2. Flask サーバーを起動

```bash
uv run python app.py
```

### 3. クライアントを実行

```bash
uv run python client.py
```

## ディレクトリ構成

```
CareSummaryGen/
├── app.py                      # Flask サーバー (Ollama gpt-oss-120b)
├── client.py                   # クライアント
├── instructions_inputs.json    # 質問テンプレート
├── utils/                      # データ抽出・前処理ユーティリティ
├── data/                       # サンプルデータ
└── old/                        # 旧バックエンド（RAG, カスタムモデル等）のバックアップ
```

## 旧バックエンド

`old/` ディレクトリに以下のバックアップを格納:
- RAG (ChromaDB + bakeneko-32b)
- カスタムファインチューニングモデル
- OpenAI API クライアント
