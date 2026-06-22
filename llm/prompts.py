"""プロンプトテンプレート（v2 agentic search）。

single-pass（全セクション1回生成）と section_worker（セクション単位抽出）、
consistency（claim 分解）で使用するプロンプトを定義する。
gpt-oss の構造化出力安定化のため、出力は JSON のみを要求し、
本文に reasoning を吸収するフィールドを設ける。
"""

# --- single_pass: 全セクションを1回で生成 ---
SINGLE_PASS_PROMPT = """あなたは熟練した看護師で、看護サマリーを作成する専門家です。
以下の患者横断情報と日付ごとの医療記録から、指定された全セクションを作成してください。

## 患者横断情報（全期間共通）
{summary_header}

## 医療記録（日付順）
{records}

## 作成するセクション
{section_specs}

## 厳守事項
- 各セクションを上記「作成するセクション」の section_key ごとに作成する。
- バイタルサイン・投薬量等の具体的数値を保持する。
- 日付は YYYY年MM月DD日 形式。時系列順に整理する。
- 記録にない情報は「記録なし」と明記し、推測しない。
- 医学的略語は初出時に正式名称を併記する。

## 出力形式
必ず次の JSON のみを出力する。説明文・コードフェンスは含めない。
{{"sections": [{{"section_key": "...", "body": "...",
"cited_dates": ["YYYYMMDD"]}}]}}
"""

# --- section_worker: 1セクションを抽出生成 ---
SECTION_EXTRACT_PROMPT = """あなたは熟練した看護師です。
以下の収集済みの医療記録から、指定セクションの本文を作成してください。

## 対象セクション
名前: {section_name}
説明: {section_description}

## 患者横断情報（全期間共通）
{summary_header}

## 収集済みの医療記録
{evidence}

## 厳守事項
- 具体的数値（バイタル・投薬量等）を保持する。
- 日付は YYYY年MM月DD日 形式・時系列順。
- 記録にない情報は「記録なし」と明記し、推測しない。
- 医学的略語は初出時に正式名称を併記する。

## 出力形式
必ず次の JSON のみを出力する。説明文・コードフェンスは含めない。
reasoning には判断根拠、body にはセクション本文、cited_dates には根拠日付を入れる。
{{"reasoning": "...", "body": "...", "cited_dates": ["YYYYMMDD"]}}
"""

# --- section_worker: LLM補完検索（ハイブリッドの agentic 部分） ---
SUPPLEMENT_PROMPT = """あなたは医療記録の検索エージェントです。
あるセクションの本文作成に必要な情報を、決定論的収集で既に集めました。
不足があれば追加で検索してください。

## 対象セクション
名前: {section_name}
説明: {section_description}

## 既に収集済みの情報（カテゴリと日付）
{collected_summary}

## 医療記録に存在する全カテゴリ・全日付（検索可能な範囲）
カテゴリ: {available_categories}
日付: {available_dates}

## 利用可能な検索ツール
- keyword: 指定キーワードを含む記録を追加取得（例: 「酸素」「点滴」「転倒」）
- date_range: 指定日付範囲の記録を追加取得（YYYYMMDD形式）

## 指示
このセクションに不足情報があるか判断してください。
- 十分なら need_more=false。
- 不足なら need_more=true とし、tool（keyword/date_range）と引数を指定してください。
- 既に収集済みの情報で足りる場合は無理に検索しないでください。

## 出力形式
必ず次の JSON のみを出力する。説明文・コードフェンスは含めない。
{{"need_more": true, "tool": "keyword", "keyword": "...", "reason": "..."}}
または
{{"need_more": false}}
"""


# --- consistency: ドラフトの主張を分解 ---
CONSISTENCY_PROMPT = """以下の看護サマリーから、検証可能な事実主張（atomic claim）を抽出してください。
バイタル数値・投薬・日付・処置など、医療記録と照合できる具体的事実のみを対象とします。

## 看護サマリー
{draft}

## 出力形式
必ず次の JSON のみを出力する。説明文・コードフェンスは含めない。
{{"claims": ["主張1", "主張2"]}}
"""
