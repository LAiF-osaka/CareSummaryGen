"""テスト共通フィクスチャ。"""

import pytest


# Ollama 接続チェック用マーカー
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "ollama: Ollama サーバー稼働時のみ実行するテスト"
    )


def pytest_collection_modifyitems(config, items):
    """Ollama 利用不可時は ollama マーカー付きテストをスキップする。

    接続確認だけでなく、設定されたモデルが利用可能かもチェックする。
    """
    try:
        from ollama import Client
        from config.settings import MODEL_NAME, OLLAMA_BASE_URL

        client = Client(host=OLLAMA_BASE_URL)
        models = client.list()
        # ローカルモデルの場合、モデルが存在するか確認
        model_names = [m.model for m in models.models] if models.models else []
        has_model = any(MODEL_NAME in name for name in model_names)
        # :cloud モデルの場合はリストに載らないので接続成功で OK
        is_cloud = "-cloud" in MODEL_NAME or ":cloud" in MODEL_NAME
        if not has_model and not is_cloud:
            raise RuntimeError(f"モデル '{MODEL_NAME}' が利用できません")
    except Exception as e:
        skip_ollama = pytest.mark.skip(reason=f"Ollama 利用不可: {e}")
        for item in items:
            if "ollama" in item.keywords:
                item.add_marker(skip_ollama)


SAMPLE_MEDICAL_RECORD = """# 患者ID: TEST001

- 20230209
  - カルテ#1
    入院時記録
    主訴: 右下肢の疼痛と腫脹
    現病歴: 2023年2月8日に自宅で転倒し、右大腿骨頸部骨折と診断。
    バイタルサイン: BP 138/82 mmHg、HR 78/分、BT 36.8℃、SpO2 97%
    意識レベル: JCS 0、GCS 15
    ADL: ベッド上安静、トイレ介助必要
    アレルギー: なし
    内服薬: アムロジピン 5mg 1日1回、メトホルミン 500mg 1日2回

- 20230210
  - カルテ#1
    手術記録
    右大腿骨頸部骨折に対し、人工骨頭置換術を施行
    麻酔: 全身麻酔
    手術時間: 2時間15分
    出血量: 350mL
    術後バイタル: BP 128/76 mmHg、HR 82/分、SpO2 98%
    術後指示: セフトリアキソン 1g×2/日 静注、ロキソプロフェン 60mg×3/日 内服
    ドレーン: JP ドレーン 1本留置
  - カルテ#2
    看護記録
    術後の疼痛管理: NRS 5/10 → フェンタニル持続投与開始後 NRS 2/10 に改善
    創部: 出血なし、ガーゼ汚染なし
    排尿: バルーンカテーテル留置中、尿量 1200mL/日

- 20230212
  - カルテ#1
    リハビリ開始
    理学療法士による評価: 右下肢の筋力 MMT 2/5
    車椅子移乗訓練開始
    バイタル: BP 132/78 mmHg、HR 74/分、SpO2 98%
    疼痛: NRS 3/10（安静時）、NRS 5/10（動作時）
    ドレーン抜去、バルーンカテーテル抜去
    排尿: 自排尿あり、残尿なし
  - カルテ#2
    看護記録
    患者への退院後生活指導実施
    家族（長女）に介護方法の指導を実施
    転倒予防のための環境整備について説明

- 20230215
  - カルテ#1
    経過記録
    リハビリ進捗: 平行棒歩行訓練開始、歩行器での歩行訓練へ移行
    ADL: 食事自立、更衣一部介助、トイレ見守り
    バイタル: BP 126/74 mmHg、HR 72/分、SpO2 99%
    採血結果: Hb 10.2 g/dL、CRP 0.8 mg/dL、WBC 6800/μL
    創部: 良好、抜糸済み
    セフトリアキソン終了、内服抗菌薬に変更（セファレキシン 500mg×3/日）

- 20230220
  - カルテ#1
    退院前カンファレンス
    退院予定日: 2023年2月22日
    退院先: 自宅（長女と同居）
    継続問題: 転倒リスク、糖尿病管理、疼痛管理
    退院後フォロー: 整形外科外来 2週間後、訪問リハビリ週2回
    処方: ロキソプロフェン 60mg 頓服、アムロジピン 5mg 1日1回、メトホルミン 500mg 1日2回
    ケアマネージャーと連携済み
"""
