import os
import re
import glob
import sys
import time
import gc
import traceback
import warnings
from typing import List, Dict, Optional
from datetime import datetime

import dotenv

dotenv.load_dotenv()

# 警告を抑制
warnings.filterwarnings("ignore", ".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", ".*Starting from v4.46, the `logits` model output will have the same type as the model.*")
warnings.filterwarnings("ignore", ".*This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length.*")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorchとTransformersがインストールされていません。抽出モードのみ使用できます。")

# 設定
MODEL_ID = "tokyotech-llm/Swallow-70b-instruct-hf"  # 使用するモデル
TOKEN = os.environ.get("HF_TOKEN", "")  # HuggingFaceのトークン（.envで管理）

# スクリプトがある場所から相対パスで指定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "1_preprocess/hanwa_kinen/input")
EXTRACTED_DIR = os.path.join(BASE_DIR, "2_summarize")
OUTPUT_DIR = os.path.join(BASE_DIR, "2_llm")

# バッチサイズ
BATCH_SIZE = 5

# 要約に必要な情報を抽出するための正規表現パターン
DATE_PATTERN = r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?'
VITAL_PATTERN = r'(体温|血圧|BP|脈拍|PR|心拍数|HR|SpO2|酸素飽和度|呼吸数|RR)[:：]?\s*(\d+\.?\d*)'
TREATMENT_PATTERN = r'(点滴|注射|内服薬|手術|処置|リハビリ|投薬|薬剤|透析|化学療法|放射線|輸血|胃管|チューブ|ドレーン|カテーテル|気管挿管|人工呼吸器)'
STATUS_PATTERN = r'(状態|症状|改善|悪化|安定|意識|疼痛|痛み|発熱|呼吸|循環|食欲|嘔気|嘔吐|下痢|便秘|排尿|浮腫|むくみ|睡眠|不眠|倦怠感|だるさ)'

# 重要なセクションを示すキーワード
IMPORTANT_SECTIONS = [
    '入院', '診断', '既往歴', '現病歴', '治療', '処置', 
    '手術', 'バイタル', '状態', '看護', '退院', '指導',
    '記録', '情報', '所見', '評価', 'アセスメント', '薬剤',
    '通院', '転院', '経過', '検査', '排泄', '食事', '栄養',
    '移動', '清潔', '衛生', 'ADL', '合併症', '併存症', 
    '家族', '服薬', '退院指導', '在宅', '介護', '訪問', 
    'リハビリ', '理学療法', '作業療法', '言語療法', '投薬'
]

# 追加の重要パターン
MEDICAL_PATTERN = r'(疾患|病名|診断名|検査|投薬|手術|摂食|栄養|排泄|清潔|移動|睡眠|ADL|意識|酸素|CT|MRI|エコー|X線|レントゲン|心電図|ECG|脳波|EEG|血液検査|尿検査|培養|病理|アレルギー|副作用|感染症|褥瘡|抑制|転倒|誤嚥|肺炎|敗血症|血栓|PICC|PEG|CVC|CVカテーテル|CHDF|IVH|TPN|PCA|PPN)'
NUMERICAL_PATTERN = r'(\d+\.?\d*)\s*(mg|ml|kg|cm|mm|mmHg|g/dl|%|℃|度|回|分|時間|日|週間|ヶ月|mmol/L|U/L|mEq/L|ng/ml|μg/dl)'

# CPU実行モードフラグ
USE_CPU = False

# モデルとトークナイザーの変数を初期化
model = None
tokenizer = None
pipe = None


def read_markdown(context_file_path: str) -> str:
    """Markdownファイルからテキストを読み込む"""
    if not os.path.exists(context_file_path):
        print(f"ファイルが存在しません: {context_file_path}")
        return ""
    try:
        with open(context_file_path, encoding="utf-8") as f:
            # ファイル内容全体を文字列にする
            content = f.read()
        return content
    except Exception as e:
        print(f"Markdownファイル {context_file_path} の読み込みに失敗しました: {e}")
        return ""


def extract_important_info(text: str) -> str:
    """
    テキストから看護サマリに必要な重要情報を抽出する
    抽出アルゴリズムを改善し、より精度の高い抽出を行う
    """
    lines = text.split('\n')
    important_lines = []
    
    # 1. セクション単位での抽出
    in_important_section = False
    section_header = ""
    section_depth = 0  # セクションの階層深さを追跡
    
    # 前後行のコンテキストを考慮した抽出
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            # 空行の場合、重要なセクション内なら保持
            if in_important_section and i > 0 and i < len(lines) - 1:
                if lines[i-1].strip() and lines[i+1].strip():  # 前後に内容がある場合は空行も保持
                    important_lines.append("")
            continue
        
        # 見出しっぽい行を検出 (複数の見出しスタイルに対応)
        is_header = False
        if re.match(r'^#+\s+', line):  # Markdown見出し
            is_header = True
            section_depth = len(re.match(r'^(#+)\s+', line).group(1))
        elif line.endswith(':'):  # コロン終わりの見出し
            is_header = True
        elif re.match(r'^[■◆●◎○※【】「」『』≪≫\[\]()（）<>〈〉]', line):  # 様々な記号による見出し
            is_header = True
        elif re.match(r'^\d+[\.、\s]', line) and len(line) < 50:  # 数字で始まる短い行
            is_header = True
        
        if is_header:
            section_header = line
            # セクション名に重要キーワードが含まれるか確認
            in_important_section = any(keyword in line.lower() for keyword in IMPORTANT_SECTIONS)
            if in_important_section:
                important_lines.append(line)
            continue
        
        # 重要なセクション内の行を保持
        if in_important_section:
            important_lines.append(line)
            continue
        
        # 日付を含む行は重要
        if re.search(DATE_PATTERN, line):
            # 日付の前後のコンテキストも追加
            context_added = False
            if i > 0 and lines[i-1].strip() and lines[i-1].strip() not in important_lines:
                important_lines.append(lines[i-1].strip())
                context_added = True
            
            important_lines.append(line)
            
            if i < len(lines) - 1 and lines[i+1].strip() and lines[i+1].strip() not in important_lines:
                important_lines.append(lines[i+1].strip())
                context_added = True
            
            if context_added:
                continue
        
        # バイタルサインを含む行は重要
        if re.search(VITAL_PATTERN, line):
            important_lines.append(line)
            continue
            
        # 治療内容を含む行は重要
        if re.search(TREATMENT_PATTERN, line):
            important_lines.append(line)
            continue
            
        # 状態変化を示す行は重要
        if re.search(STATUS_PATTERN, line):
            important_lines.append(line)
            continue
            
        # 医療関連の情報を含む行は重要
        if re.search(MEDICAL_PATTERN, line):
            important_lines.append(line)
            continue
            
        # 数値データを含む行は重要
        if re.search(NUMERICAL_PATTERN, line):
            important_lines.append(line)
            continue
            
        # その他の重要な特徴
        # 箇条書きの可能性が高い短い行
        if (line.startswith('-') or line.startswith('・') or re.match(r'^\d+[\.、]', line)) and len(line) < 100:
            important_lines.append(line)
            continue
    
    # 重複行を除去
    unique_lines = []
    for line in important_lines:
        if line not in unique_lines:
            unique_lines.append(line)
    
    # 結果の文字列を生成
    result = '\n'.join(unique_lines)
            
    # テキストが極端に短くなりすぎた場合は、元のテキストを返す
    if len(result) < len(text) * 0.05:
        print("  抽出した情報が少なすぎるため、元のテキストを使用します")
        return text
        
    return result


def load_model() -> bool:
    """
    モデルとトークナイザーをロードする
    """
    global model, tokenizer, pipe
    
    if not HAS_TORCH:
        print("PyTorchとTransformersがインストールされていないため、モデルをロードできません")
        return False
    
    try:
        print("モデルをロード中...")
        
        # GPUが利用可能かチェック
        if torch.cuda.is_available() and not USE_CPU:
            device = "cuda"
            torch_dtype = torch.bfloat16
            print(f"GPUを使用してモデルをロードします: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("CPUを使用してモデルをロードします")
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            token=TOKEN,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device if device == "cpu" else "auto",
        )
        
        print("トークナイザーをロード中...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
        
        print("パイプラインをセットアップ中...")
        pipe = pipeline(
            trust_remote_code=True,
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=4096,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.15,
            top_p=0.92,
            return_full_text=False,
        )
        
        print("モデルのロードが完了しました")
        return True
        
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return False


def create_system_prompt() -> str:
    """
    看護サマリ生成のためのシステムプロンプトを作成
    """
    return """
あなたは熟練した看護師で、患者の看護サマリを作成するエキスパートです。
看護サマリとは、患者の病歴や治療内容、看護経過などの情報をまとめ、他の医療機関や関係者に提供する重要な医療文書です。

以下の形式に従って、患者の看護サマリを作成してください：

--- 入院中の経過及び看護上の問題経過 ---
[入院日] 入院時の状態（バイタルサイン、意識レベル、主訴、ADL状況など）
[日付] 実施した治療内容とその効果（治療名、薬剤名、投与量、患者の反応など）
[日付] 状態の変化や症状の推移（具体的な数値やスケールで記載）
[日付] 看護上の問題とケア内容（実施した看護ケア、患者のセルフケア能力など）
[日付] 退院に向けた準備状況（リハビリの進捗、自宅環境の調整、指導内容など）

--- 備考 ---
[退院後の注意点、外来通院予定、利用するサービスなど]

作成にあたり、以下のポイントを必ず守ってください：
1. バイタルサインやラボデータは必ず実際の数値で記載すること（BP 120/80 mmHg、HR 72/分、SpO2 98%など）
2. 全ての治療や処置は具体的な名称と用量を記載すること（セフトリアキソン 1g×2/日など）
3. 日付は「2023年10月20日」という明確な形式で記載し、時系列順に整理すること
4. 医学的略語は初出時にフルスペルを併記すること（CVA（脳血管障害）など）
5. 患者の状態変化は客観的な指標や観察事実に基づいて記載すること
6. 情報が不足している場合は「記録なし」と明記し、推測や憶測は避けること
7. 重要な処置や手術については詳細な説明を含めること
8. 退院後のフォローアップ計画や注意事項を具体的に記載すること

看護サマリは医療専門家間の情報共有のための文書であり、専門用語を適切に使用し、簡潔かつ具体的に記載してください。
"""


def generate_summary(context: str) -> Optional[str]:
    """
    コンテキストから看護サマリを生成する
    """
    if not model or not tokenizer or not pipe:
        print("モデルがロードされていません")
        return None
    
    try:
        # チャンクに分割してメモリ効率を上げる
        if len(context) > 8000:
            print(f"コンテキストが長いため ({len(context)} 文字)、重要情報抽出を実行します")
            context = extract_important_info(context)
            print(f"抽出後のコンテキストサイズ: {len(context)} 文字")
        
        # さらにコンテキストが長い場合はチャンクに分割
        if len(context) > 4000:
            print("コンテキストが長いため、チャンクに分割して処理します")
            # チャンクに分割
            chunks = []
            # 単純な改行による分割
            paragraphs = context.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < 4000:
                    current_chunk += para + "\n\n"
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 各チャンクに位置情報を追加
            for i in range(len(chunks)):
                chunks[i] = f"[チャンク {i+1}/{len(chunks)}]\n" + chunks[i]
            
            print(f"チャンク数: {len(chunks)}")
            
            all_outputs = []
            for i, chunk in enumerate(chunks):
                print(f"チャンク {i+1}/{len(chunks)} を処理中...")
                
                # システムプロンプトを含む完全なプロンプトを作成
                prompt = f"{create_system_prompt()}\n\n以下の医療記録から看護サマリを作成してください：\n\n{chunk}"
                
                # プロンプトの内容をログに出力
                print(f"チャンク {i+1} のプロンプト（先頭500文字）：\n{prompt[:500]}...\n（後略）")
                
                try:
                    # チャンクごとにGPUメモリをクリア
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # 出力を生成
                    output = pipe(prompt, max_length=8192)[0]["generated_text"]
                    
                    # 出力からプロンプト部分を削除
                    if "以下の医療記録から" in output:
                        output = output.split("以下の医療記録から")[0]
                    
                    all_outputs.append(output)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"  GPUメモリ不足エラー: {e}")
                        print("  GPUメモリ不足のため処理を中止します。")
                        return "GPUメモリ不足エラーが発生しました。処理を中止します。"
                    else:
                        print(f"  要約生成中にエラーが発生しました: {e}")
                        all_outputs.append("要約生成中にエラーが発生しました。")
                except Exception as e:
                    print(f"  要約生成中にエラーが発生しました: {e}")
                    all_outputs.append("要約生成中にエラーが発生しました。")
            
            # 全てのチャンクの出力を結合して整形
            combined_output = "\n\n".join(all_outputs)
            
            # 重複を排除して整形
            return clean_and_format_output(combined_output)
        else:
            # 単一のプロンプトで処理
            print("単一プロンプトで処理します")
            prompt = f"{create_system_prompt()}\n\n以下の医療記録から看護サマリを作成してください：\n\n{context}"
            
            # プロンプトの内容をログに出力
            print(f"単一プロンプト（先頭500文字）：\n{prompt[:500]}...\n（後略）")
            
            # GPUメモリをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            output = pipe(prompt, max_length=8192)[0]["generated_text"]
            return clean_and_format_output(output)
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"GPUメモリ不足エラー: {e}")
            print("GPUメモリ不足のため処理を中止します。")
            return "GPUメモリ不足エラーが発生しました。処理を中止します。"
        
        print(f"要約生成中にエラーが発生しました: {e}")
        print(traceback.format_exc())
        return None
    except Exception as e:
        print(f"要約生成中にエラーが発生しました: {e}")
        print(traceback.format_exc())
        return None


def clean_and_format_output(text: str) -> str:
    """
    LLMの出力をクリーンアップして整形する
    - 余分な空行を削除
    - 重複する情報を削除
    - ヘッダーとセクションを統一
    - 日付形式を統一
    """
    # チャンク情報を削除
    text = re.sub(r'\[チャンク \d+/\d+\]\n', '', text)
    
    # システムの指示やプロンプトが含まれていたら削除
    text = re.sub(r'以下の形式で看護サマリを作成してください.*?---', '---', text, flags=re.DOTALL)
    
    # 看護サマリの開始を統一
    if not text.startswith("---"):
        for pattern in ["入院中の経過", "看護上の問題", "経過及び", "入院経過", "看護経過"]:
            if pattern in text[:200]:
                parts = text.split(pattern, 1)
                if len(parts) > 1:
                    text = f"--- 入院中の経過及び看護上の問題経過 ---\n{parts[1].lstrip()}"
                    break
    
    # 備考セクションの形式を統一
    if "備考" not in text and "退院後" in text:
        # 退院後の情報を備考セクションに移動
        match = re.search(r'(退院後.*?)($|---)', text, re.DOTALL)
        if match:
            backup_text = text
            try:
                before_discharge = text.split(match.group(1))[0]
                text = f"{before_discharge.rstrip()}\n\n--- 備考 ---\n{match.group(1).strip()}"
            except:
                text = backup_text  # エラーが発生した場合は元のテキストを使用
    
    # 日付形式を統一 (YYYY年MM月DD日)
    def date_replacer(match):
        date_str = match.group(0)
        year, month, day = None, None, None
        
        # YYYY/MM/DD 形式
        slash_match = re.match(r'(\d{4})/(\d{1,2})/(\d{1,2})', date_str)
        if slash_match:
            year, month, day = slash_match.groups()
        
        # YYYY-MM-DD 形式
        hyphen_match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_str)
        if hyphen_match:
            year, month, day = hyphen_match.groups()
        
        if year and month and day:
            return f"{year}年{month}月{day}日"
        return date_str
    
    text = re.sub(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', date_replacer, text)
    
    # 余分な空行を削減（2行以上の空行を2行に）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # セクションが重複している場合は最初のセクションのみ残す
    if text.count("--- 入院中の経過") > 1:
        first_section_end = text.find("--- 入院中の経過", 1)
        if first_section_end > 0:
            second_section_start = text.find("---", first_section_end)
            if second_section_start > 0:
                text = text[:first_section_end] + text[second_section_start:]
    
    # 最後にセクションヘッダーを確認し、標準的な形式に統一
    text = text.replace("--- 入院経過 ---", "--- 入院中の経過及び看護上の問題経過 ---")
    text = text.replace("--- 看護経過 ---", "--- 入院中の経過及び看護上の問題経過 ---")
    text = text.replace("--- 退院時の注意点 ---", "--- 備考 ---")
    text = text.replace("--- 退院指導 ---", "--- 備考 ---")
    
    # すべてのセクションが存在することを確認
    if "--- 入院中の経過" not in text:
        text = "--- 入院中の経過及び看護上の問題経過 ---\n" + text
    
    if "--- 備考 ---" not in text:
        text = text.rstrip() + "\n\n--- 備考 ---\n情報なし"
    
    return text.strip()


def extract_patient_id(file_path: str) -> str:
    """ファイル名から患者IDを抽出する"""
    # ファイル名を取得してパスと拡張子を除去
    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    
    # 抽出済みファイルの場合は "_extracted" を取り除く
    if file_name_without_ext.endswith("_extracted"):
        return file_name_without_ext[:-10]  # "_extracted" の長さは 10
    
    return file_name_without_ext


def ensure_directory_exists(directory: str) -> None:
    """ディレクトリが存在しない場合は作成する"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_progress_log(message: str) -> None:
    """進捗ログを記録する"""
    log_file = os.path.join(OUTPUT_DIR, "progress.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def process_files(extract_only: bool = False, generate_only: bool = False) -> None:
    """
    ファイルを処理する
    extract_only=True: 抽出のみ実行
    generate_only=True: 既に抽出されたファイルを使用してサマリー生成のみ実行
    両方False: 抽出してからサマリー生成まで実行
    """
    # 処理するファイルを取得
    if generate_only:
        # 抽出済みファイルを使用
        md_files = glob.glob(os.path.join(EXTRACTED_DIR, "*_extracted.md"))
        if not md_files:
            print(f"指定されたディレクトリ {EXTRACTED_DIR} に抽出済みMarkdownファイルが見つかりませんでした。")
            print("まず '--extract' オプションでファイルを抽出してください。")
            return
    else:
        # 元のファイルを使用
        md_files = glob.glob(os.path.join(INPUT_DIR, "*.md"))
        if not md_files:
            print(f"指定されたディレクトリ {INPUT_DIR} にMarkdownファイルが見つかりませんでした。")
            return
    
    print(f"処理するファイル数: {len(md_files)}")
    
    # 対象のファイルを限定（テスト用）
    test_mode = input("テストモードで実行しますか？一部のファイルのみ処理します (y/n): ").lower() == 'y'
    if test_mode:
        md_files = md_files[:3]  # 最初の3ファイルのみ処理
        print(f"テストモード: 処理するファイル数を {len(md_files)} に制限しました")
    
    # ディレクトリの準備
    if extract_only or not generate_only:
        ensure_directory_exists(EXTRACTED_DIR)
    
    if not extract_only:
        ensure_directory_exists(OUTPUT_DIR)
        # サマリー生成が必要な場合はモデルをロード
        if not load_model():
            print("モデルのロードに失敗したため、抽出のみ実行します。")
            extract_only = True
    
    # バッチ処理
    batch_count = 0
    total_batches = (len(md_files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(md_files), BATCH_SIZE):
        batch_count += 1
        batch_files = md_files[i:i+BATCH_SIZE]
        print(f"\nバッチ {batch_count}/{total_batches} を処理中... (ファイル数: {len(batch_files)})")
        
        for md_file in batch_files:
            patient_id = extract_patient_id(md_file)
            print(f"\n患者ID: {patient_id} のファイルを処理中...")
            
            # ファイルを読み込む
            context = read_markdown(md_file)
            if not context:
                print(f"患者ID: {patient_id} でコンテキストが空です。")
                continue
            
            print(f"コンテキストの長さ: {len(context)} 文字")
            
            # 抽出フェーズ
            if not generate_only:
                extracted_info = extract_important_info(context)
                output_file_path = os.path.join(EXTRACTED_DIR, f"{patient_id}_extracted.md")
                
                try:
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        f.write(extracted_info)
                    print(f"抽出情報を保存しました: {output_file_path}")
                    print(f"元の長さ: {len(context)}文字 → 抽出後: {len(extracted_info)}文字")
                except Exception as e:
                    print(f"ファイルへの書き込みに失敗しました: {e}")
                
                # 抽出のみの場合は次のファイルへ
                if extract_only:
                    continue
                
                # 抽出したコンテキストを使う
                context = extracted_info
            
            # サマリー生成フェーズ
            if not extract_only:
                # 患者ごとの出力ディレクトリを作成
                patient_output_dir = os.path.join(OUTPUT_DIR, patient_id)
                ensure_directory_exists(patient_output_dir)
                
                # 出力ファイル名を決定
                output_mark = "_optimized" if generate_only else ""
                output_file_path = os.path.join(patient_output_dir, f"{patient_id}{output_mark}_summary.txt")
                
                # もし以前のファイルが存在する場合はスキップ（再利用）
                if os.path.exists(output_file_path):
                    print(f"ファイルが既に存在します: {output_file_path}")
                    with open(output_file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if content and len(content) > 10:
                        print(f"既存のファイルを使用します")
                        continue
                
                # サマリーを生成
                answer = generate_summary(context)
                
                # 回答をファイルに保存
                if answer:
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        f.write(answer)
                    print(f"回答を保存しました: {output_file_path}")
                else:
                    print(f"回答が空のため、保存しませんでした")
            
            # GPUメモリを解放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()


def main() -> None:
    """メイン関数"""
    start_time = datetime.now()
    
    # コマンドライン引数の処理
    extract_only = False
    generate_only = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--extract":
            extract_only = True
            print("抽出モードで実行します。重要情報のみを抽出したMarkdownファイルを生成します。")
        elif sys.argv[1] == "--generate":
            generate_only = True
            print("生成モードで実行します。抽出済みファイルからサマリーを生成します。")
    
    print(f"処理を開始します: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if generate_only:
        print(f"入力ディレクトリ: {EXTRACTED_DIR}")
    else:
        print(f"入力ディレクトリ: {INPUT_DIR}")
    
    if extract_only:
        print(f"抽出情報保存先: {EXTRACTED_DIR}")
    else:
        print(f"出力ディレクトリ: {OUTPUT_DIR}")
    
    # 入力ディレクトリの存在確認
    if generate_only:
        if not os.path.exists(EXTRACTED_DIR):
            print(f"抽出済みファイルディレクトリが存在しません: {EXTRACTED_DIR}")
            print("まず '--extract' オプションでファイルを抽出してください。")
            sys.exit(1)
    elif not os.path.exists(INPUT_DIR):
        print(f"入力ディレクトリが存在しません: {INPUT_DIR}")
        sys.exit(1)
    
    try:
        process_files(extract_only=extract_only, generate_only=generate_only)
        end_time = datetime.now()
        processing_time = end_time - start_time
        print(f"処理が完了しました: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"処理時間: {processing_time}")
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main() 