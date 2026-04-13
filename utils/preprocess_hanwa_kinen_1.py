import os
import shutil
from pathlib import Path
import re
import argparse
import traceback
from bs4 import BeautifulSoup

def parse_care_record(content, filename):
    # HTMLタグを除去
    try:
        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text('\n')
    except:
        text_content = content

    # ファイル名から日付を抽出
    date_match = re.search(r'(\d{8})', filename)
    if date_match:
        date = date_match.group(1)
        records = {date: {'1': []}}  # カルテ番号は1で固定
        
        # 改行で分割して内容を取得
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        records[date]['1'].extend(lines)
        
        return records
    
    return {}

def format_care_record(records):
    # 構造化されたデータを整形
    formatted_content = []
    
    for date in sorted(records.keys()):
        if not records[date]:  # 空の日付はスキップ
            continue
            
        formatted_content.append(f"- {date}")
        
        for care_num, care_content in sorted(records[date].items()):
            if not care_content:  # 空のカルテはスキップ
                continue
                
            formatted_content.append(f"  - カルテ#{care_num}")
            for line in care_content:
                formatted_content.append(f"    {line}")
                
    return '\n'.join(formatted_content)

def read_file_with_encoding(file_path):
    encodings = ['utf-8', 'shift-jis', 'cp932']
    
    # まずバイナリモードでファイルを読み込む
    with open(file_path, 'rb') as f:
        content_bytes = f.read()
    
    # 各エンコーディングを試す
    for encoding in encodings:
        try:
            return content_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
            
    # すべてのエンコーディングが失敗した場合
    print(f"Warning: Could not decode file {file_path} with any of the encodings: {encodings}")
    print("Attempting to decode with errors='ignore'...")
    return content_bytes.decode('shift-jis', errors='ignore')

def process_patient_data(input_dir, output_dir):
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 処理済みの患者IDを追跡
    processed_patients = set()
    
    try:
        # 入力ディレクトリ内のすべての患者IDディレクトリを処理
        for patient_dir in os.listdir(input_dir):
            patient_path = os.path.join(input_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
                
            print(f"\nProcessing patient directory: {patient_dir}")
            
            # 患者ごとの全レコードを保持する辞書
            patient_records = {}
            
            try:
                # 患者IDディレクトリ内のすべての日付ディレクトリを処理
                for date_dir in os.listdir(patient_path):
                    date_path = os.path.join(patient_path, date_dir)
                    if not os.path.isdir(date_path):
                        continue
                        
                    print(f"  Processing date directory: {date_dir}")
                    
                    # 日付ごとのカルテ番号を追跡
                    care_number = 1
                    
                    try:
                        # 日付ディレクトリ内のすべての時間ディレクトリを処理
                        for time_dir in os.listdir(date_path):
                            time_path = os.path.join(date_path, time_dir)
                            if not os.path.isdir(time_path):
                                continue
                                
                            print(f"    Processing time directory: {time_dir}")
                            
                            # 同じ時間ディレクトリ内の全テキストファイルの内容を保持
                            time_dir_content = []
                            
                            try:
                                # 時間ディレクトリ内のすべてのテキストファイルを処理
                                for file in os.listdir(time_path):
                                    if not file.endswith('.txt'):
                                        continue
                                        
                                    input_file = os.path.join(time_path, file)
                                    print(f"      Processing file: {file}")
                                    
                                    try:
                                        # 複数のエンコーディングを試してファイルを読み込む
                                        content = read_file_with_encoding(input_file)
                                            
                                        # HTMLタグを除去
                                        try:
                                            soup = BeautifulSoup(content, 'html.parser')
                                            text_content = soup.get_text('\n')
                                        except:
                                            text_content = content
                                            
                                        # 改行で分割して内容を取得
                                        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                                        time_dir_content.extend(lines)
                                        
                                    except Exception as e:
                                        print(f"        Error processing file {file}:")
                                        print(f"        {str(e)}")
                                        print("        Traceback:")
                                        print(traceback.format_exc())
                                        continue
                                
                                # 時間ディレクトリ内の全ファイルの内容をまとめて保存
                                if time_dir_content:
                                    if date_dir not in patient_records:
                                        patient_records[date_dir] = {}
                                    patient_records[date_dir][str(care_number)] = time_dir_content
                                    care_number += 1
                                        
                            except Exception as e:
                                print(f"      Error processing time directory {time_dir}:")
                                print(f"      {str(e)}")
                                print("      Traceback:")
                                print(traceback.format_exc())
                                continue
                                
                    except Exception as e:
                        print(f"    Error processing date directory {date_dir}:")
                        print(f"    {str(e)}")
                        print("    Traceback:")
                        print(traceback.format_exc())
                        continue
                
                # 患者の全レコードを出力
                if patient_records:
                    output_file = os.path.join(output_dir, f"{patient_dir}.md")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"# 患者ID: {patient_dir}\n\n")
                        f.write(format_care_record(patient_records))
                    processed_patients.add(patient_dir)
                        
            except Exception as e:
                print(f"  Error processing patient directory {patient_dir}:")
                print(f"  {str(e)}")
                print("  Traceback:")
                print(traceback.format_exc())
                continue
                
        # 処理結果を表示
        print("\nProcessing summary:")
        print(f"Successfully processed {len(processed_patients)} patients:")
        for patient_id in sorted(processed_patients):
            print(f"- {patient_id}")
            
    except Exception as e:
        print(f"Error processing input directory:")
        print(str(e))
        print("Traceback:")
        print(traceback.format_exc())
        return

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='Process care records from Hanwa Kinen Hospital')
    parser.add_argument('--input', '-i', required=True,
                      help='Input directory containing patient data')
    parser.add_argument('--output', '-o', required=True,
                      help='Output directory for processed files')
    
    args = parser.parse_args()
    
    # データ処理を実行
    process_patient_data(args.input, args.output)
    print("Processing completed!")

if __name__ == "__main__":
    main() 