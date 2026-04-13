import glob
import os
import re
import zipfile
from datetime import datetime
from collections import defaultdict


def extract_zip_files(xml_dir):
    """
    xml_dir内のzipファイルをその場に解凍します。
    """
    zip_files = glob.glob(os.path.join(xml_dir, "*.zip"))
    
    for zip_file in zip_files:
        print(f"  zipファイルを解凍中: {os.path.basename(zip_file)}")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # zipファイルと同じディレクトリに解凍
                zip_ref.extractall(xml_dir)
            print(f"    解凍完了: {os.path.basename(zip_file)}")
        except Exception as e:
            print(f"    zipファイル解凍エラー: {zip_file} - {e}")


def extract_date_from_filename(filename):
    """
    ファイル名から日付を抽出します。
    例: 20230209_xxx.txt -> "20230209"
        0311_20230209_xxx.txt -> "20230209"
    """
    # YYYYMMDDパターンを探す
    date_pattern = r'(\d{8})'
    matches = re.findall(date_pattern, filename)
    
    for match in matches:
        try:
            # 有効な日付かチェック
            datetime.strptime(match, '%Y%m%d')
            return match
        except ValueError:
            continue
    
    return None


def concat_txt_from_xml_subdirs(input_dir, output_dir):
    """
    input_dir/<患者ID>/xml/ フォルダ内のtxtファイルの内容を日付ごとにグループ化して
    output_dir/<患者ID>/<患者ID>.md を作成します。
    """
    if not os.path.isdir(input_dir):
        print(f"エラー: '{input_dir}' ディレクトリが見つかりません。")
        return

    print(f"入力ディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}")

    patient_ids = [
        d
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]

    if not patient_ids:
        print(f"'{input_dir}' 内に処理対象の患者フォルダが見つかりません。")
        return

    print(f"{len(patient_ids)} 人の患者データを処理します。")

    for patient_id in patient_ids:
        print(f"--- 患者ID: {patient_id} の処理を開始 ---")
        xml_dir = os.path.join(input_dir, patient_id, "xml")

        if not os.path.isdir(xml_dir):
            print(
                f"警告: 'xml' ディレクトリが見つかりません: {xml_dir}。スキップします。"
            )
            continue

        # zipファイルがあれば解凍
        extract_zip_files(xml_dir)

        # txtファイルを再帰的に検索（解凍されたファイルも含む）
        txt_files = sorted(glob.glob(os.path.join(xml_dir, "**", "*.txt"), recursive=True))

        if not txt_files:
            print(
                f"警告: {xml_dir} 内に .txt ファイルが見つかりません。スキップします。"
            )
            continue

        print(f"  {len(txt_files)}個の .txt ファイルを処理します。")
        
        # 日付ごとにファイルをグループ化
        date_groups = defaultdict(list)
        files_without_date = []
        
        for txt_file in txt_files:
            filename = os.path.basename(txt_file)
            date_str = extract_date_from_filename(filename)
            
            if date_str:
                date_groups[date_str].append(txt_file)
            else:
                files_without_date.append(txt_file)
        
        if files_without_date:
            print(f"  警告: 日付を抽出できないファイル {len(files_without_date)} 個をスキップします")
            for file in files_without_date:
                print(f"    - {os.path.basename(file)}")
        
        if not date_groups:
            print(
                f"  有効な日付付きファイルがなかったため、患者ID: {patient_id} をスキップします。"
            )
            continue

        # 出力ディレクトリを確保
        os.makedirs(output_dir, exist_ok=True)

        # 出力ファイルは直接output_dir内に{patient_id}.mdとして作成
        output_file_path = os.path.join(output_dir, f"{patient_id}.md")

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(f"# 患者ID: {patient_id}\n\n")
                
                # 日付順にソート
                sorted_dates = sorted(date_groups.keys())
                
                for date_str in sorted_dates:
                    # 日付をYYYYMMDD形式で表示
                    f.write(f"- {date_str}\n")
                    
                    files_for_date = sorted(date_groups[date_str])
                    
                    for i, txt_file in enumerate(files_for_date, 1):
                        f.write(f"  - カルテ#{i}\n")
                        
                        try:
                            with open(
                                txt_file, "r", encoding="cp932", errors="ignore"
                            ) as content_file:
                                content = content_file.read()
                                # HTMLタグを除去
                                clean_content = re.sub(r"<[^>]+>", "", content)
                                # 改行で分割して内容を取得（空行や余計な改行を除去）
                                lines = [line.strip() for line in clean_content.split('\n') if line.strip()]
                                # 内容を4スペースでインデント
                                indented_content = '\n'.join(f"    {line}" for line in lines)
                                f.write(f"{indented_content}\n")
                        except Exception as e:
                            print(f"    ファイル読み込みエラー: {txt_file} - {e}")
                            f.write(f"    *ファイル読み込みエラー: {os.path.basename(txt_file)}*\n")
                
            print(f"  結合したファイルを保存しました: {output_file_path}")
            print(f"  処理した日付: {len(sorted_dates)} 日分")
        except Exception as e:
            print(f"  ファイル書き込みエラー: {output_file_path} - {e}")

    print("--- すべての処理が完了しました ---")


if __name__ == "__main__":
    # 実際の環境に合わせてパスを修正してください
    # 例: Windows -> "D:/work/ASC/新記念病院_250623"
    # 例: macOS/Linux -> "/path/to/your/data"
    input_directory = "D:/work/LAiF/0_data/gs_shinkinen"
    # input_directory = "D:/work/ASC/新記念病院_250623"

    # request_custom.py が読み込むディレクトリ構造に合わせる
    output_directory = "D:/work/LAiF/1_preprocess/gs_shinkinen/input"
    # output_directory = "D:/work/LAiF/2_gspreprocess/shinkinen/input"

    # 出力ディレクトリがなければ作成
    os.makedirs(output_directory, exist_ok=True)

    concat_txt_from_xml_subdirs(input_directory, output_directory)
