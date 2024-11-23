# 実行手順
1. chromadb.HttpClient を使用して、localhost のポート 8000 で実行されている ChromaDB サーバーに接続します。
    - ChromaDB サーバーを起動するには、別のターミナルで以下を実行してください。
    - chroma run --path ./vectorDB
2. flaskを使用して、localhostのポート 5000 で実行されているLLMアプリケーションサーバーに接続します。
    - flask LLMアプリケーションサーバーを起動するには、別ターミナルで以下を実行してください。
    - python app.py
3. LLMアプリケーションサーバーに対して、リクエストを送り、LLMの回答を受け取ります。
    - リクエストの送るには、別ターミナルで以下を実行してください。
    - python request.py

# vectorDB(ChromaDB)の更新方法 (増分更新)
1. dataフォルダにデータを格納してください。
2. 以下のコマンドを別ターミナルで実行してください。
    - python store_vector.py
