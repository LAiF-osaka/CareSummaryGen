# transformerを用いてファインチューニング済みのカスタムモデルを使用する場合
1. flaskを使用して、localhostのポート 5000 で実行されているLLMアプリケーションサーバーに接続します。
    - flask LLMアプリケーションサーバーを起動するには、別ターミナルで以下を実行してください。
```
python app_custom.py
```
2. LLMアプリケーションサーバーに対して、リクエストを送り、LLMの回答を受け取ります。
    - リクエストの送るには、別ターミナルで以下を実行してください。
```
python request.py
```

# RAGを用いる場合
## 実行手順
1. (RAGの場合)chromadb.HttpClient を使用して、localhost のポート 8000 で実行されている ChromaDB サーバーに接続します。
    - ChromaDB サーバーを起動するには、別のターミナルで以下を実行してください。
```
chroma run --path ./vectorDB
```
2. flaskを使用して、localhostのポート 5000 で実行されているLLMアプリケーションサーバーに接続します。
    - flask LLMアプリケーションサーバーを起動するには、別ターミナルで以下を実行してください。
```
uv run python app.py
```
3. LLMアプリケーションサーバーに対して、リクエストを送り、LLMの回答を受け取ります。
    - リクエストの送るには、別ターミナルで以下を実行してください。
```
uv run python request.py
```

## (RAGの場合)vectorDB(ChromaDB)の更新方法 (増分更新)
1. dataフォルダにデータを格納してください。
2. 以下のコマンドを別ターミナルで実行してください。
```
python store_vector.py
```

# ollamaを用いる場合
## ollama モデルのインストール
1. huggingfaceでモデル（quantization済みのggufファイル）を見つける
2. use this model を選択し、ollamaでrunをする（これでモデルがローカルにインストールされる）
```例
ollama run hf.co/rinna/qwen2.5-bakeneko-32b-instruct-v2-gguf:Q8_0
```
3. 以下コマンドでモデルがインストールされていることを確認する
```
ollama list
```

## 実行手順(ollamaを用いたモデルの場合)
1. flaskを使用して、localhostのポート 5000 で実行されているLLMアプリケーションサーバーに接続します。
    - flask LLMアプリケーションサーバーを起動するには、別ターミナルで以下を実行してください。
    - 同時にlocalhostのポート11434でollamaサーバが起動する。
```
uv run python app_ollama.py
```
2. LLMアプリケーションサーバーに対して、リクエストを送り、LLMの回答を受け取ります。
```
uv run python run_ollama.py
```

## 補足
- transformerモデルのパラメータは~/.cache/huggingface/hub 配下に格納される。
    - huggingface-cli login
    ```
    The token `kinshukai-ps` has been saved to C:\Users\Administrator\.cache\huggingface\stored_tokens
    Your token has been saved in your configured git credential helpers (manager).
    Your token has been saved to C:\Users\Administrator\.cache\huggingface\token
    Login successful.
    The current active token is: `kinshukai-ps`
    ```

# PowerShellでのバックグラウンド実行
## 実行
$job1 = Start-Job {
    Set-Location C:/Users/Administrator/CareSummaryGen
    python app_custom.py
    }
$job2 = Start-Job {
    Set-Location C:/Users/Administrator/CareSummaryGen
    python request_custom.py
    }

$job1 = Start-Job {
    Set-Location C:/Users/Administrator/CareSummaryGen
    python app_ollama.py
    }
$job2 = Start-Job {
    Set-Location C:/Users/Administrator/CareSummaryGen
    python run_ollama.py
    }

## 標準出力確認
Receive-Job -Job $job1
Receive-Job -Job $job2

## バックグラウンドジョブの確認
Get-Job

## バックグラウンドジョブの削除
Remove-Job -Job $job1
Remove-Job -Job $job2
Remove-Job -Id <int[]>