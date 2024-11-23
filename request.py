import json

import pandas as pd
import requests

url = "http://localhost:5000/ask"
headers = {"Content-Type": "application/json"}
CSV_PATH = "instructions_inputs.csv"
# データを順番に処理
data = pd.read_csv(CSV_PATH)
for index, row in data.iterrows():
    input_text = row["input"]
    data = {"question": input_text}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        answer = response.json().get("answer")
        print(answer)
    else:
        print("エラー:", response.json())
print("すべての質問への回答が完了しました。")
