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

## モデル切替

`.env` のモデル指定を差し替えるだけで使用モデルが切り替わる。サンプリング
パラメータ（temperature 等）は `config/model_profiles.py` がモデル名から
自動解決するため、コードの変更は不要。

```
# ローカル Ollama で qwen3.8 27B を使う場合
ENV=production
OLLAMA_MODEL_PRODUCTION=qwen3.8:27b
OLLAMA_BASE_URL_PRODUCTION=http://localhost:11434
```

| モデルファミリ | temperature | top_p | top_k | repeat_penalty | 構造化出力の温度 |
|---|---|---|---|---|---|
| `gpt-oss` | 0.1 | 0.92 | — | 1.2 | 0.0 |
| `qwen3` 系（3.5 / 3.6 / 3.8） | 0.7 | 0.8 | 20 | 1.0 | 0.1 |
| `gemma` 系（3 / 4） | 1.0 | 0.95 | 64 | 1.0 | 0.1 |
| 未登録モデル | 0.2 | 0.9 | — | — | 0.1 |

`qwen3` 系は `presence_penalty=1.5` も併用する（公式の非思考モード推奨）。

Qwen3 系は公式が「repetition_penalty の引き上げ」と「greedy decoding」を
非推奨としているため、反復抑制は `presence_penalty` に委ね、構造化出力でも
温度 0 を使わない設定になっている。
コンテキスト長・生成長はモデル比較の公平性のため全モデル共通。

### モデル間の生成結果を比較する

同一の医療記録・テンプレートに対してモデルだけを差し替えて生成し、
生成本文・所要時間・レビューフラグ・セクション別の充足状況を並べた
Markdown レポートを出力する。

```bash
uv run python -m scripts.compare_models \
  --model gpt-oss:120b-cloud@https://ollama.com \
  --model qwen3.8:27b@http://localhost:11434
```

主なオプション:

| オプション | 既定 | 説明 |
|---|---|---|
| `--model MODEL[@BASE_URL]` | （必須） | 比較対象。複数指定可 |
| `--input PATH` | テスト用サンプル | 医療記録テキスト |
| `--template ID` | `HOSPITAL` | 使用テンプレート |
| `--output-dir DIR` | `output/model-comparison` | 出力先 |
| `--timeout SEC` | 3600 | 1 モデルあたりの上限時間 |

出力先に `report.md`（比較レポート）、`<model>.json`（生の結果）、
`input.md`（使用した入力）が保存される。

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
