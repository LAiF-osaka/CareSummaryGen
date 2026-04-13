import os
import zipfile
import xml.etree.ElementTree as ET
import glob
import shutil
import codecs
import re
import chardet
import argparse

def process_txt_file(txt_file, patient_output_dir, safe_filename):
    """
    txtファイルを処理してHTMLタグをパースし、サマリ部分を抽出する
    """
    try:
        txt_file_name = os.path.splitext(os.path.basename(txt_file))[0]
        print(f"    処理中のtxtファイル: {txt_file_name}")
        
        # txtファイルのエンコーディングを自動検出
        with open(txt_file, 'rb') as f:
            raw_data = f.read(10000)  # 最初の10000バイトを読み込み
            detected = chardet.detect(raw_data)
            encoding = detected['encoding']
            confidence = detected['confidence']
            
            print(f"      検出されたエンコーディング: {encoding} (信頼度: {confidence:.2f})")
            
            if encoding is None or confidence < 0.5:
                # 検出されなかったか信頼度が低い場合は、一般的な日本語エンコーディングを試す
                encoding = 'utf-8'  # txtファイルの場合はUTF-8を優先
        
        # エンコーディングを指定してファイル全体を読み込む
        with codecs.open(txt_file, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        # HTMLタグを除去してテキストを抽出
        clean_content = re.sub(r'<[^>]+>', '', content)
        
        # 抽出対象のセクション
        target_sections = [
            "＜指導した内容＞",
            "＜医療機器装着・挿入・処置部位＞",
            "＜入院中の看護の経過（生活状況）＞",
            "＜患者への病状説明及び本人・家族の受け止め方＞",
            "＜継続される問題（今後のリスク）＞", 
            "＜その他＞"
        ]
        
        # セクションごとにテキストを抽出
        extracted_sections = {}
        
        for i, section in enumerate(target_sections):
            # 現在のセクションの開始位置を探す
            start_pos = clean_content.find(section)
            if start_pos == -1:
                continue
                
            # 次のセクションの開始位置を探す（終了位置として使用）
            end_pos = len(clean_content)  # デフォルトは文書の終端
            for j in range(i + 1, len(target_sections)):
                next_section_pos = clean_content.find(target_sections[j])
                if next_section_pos != -1:
                    end_pos = next_section_pos
                    break
            
            # セクションのタイトル行の後から次のセクションまでのテキストを抽出
            section_start = start_pos + len(section)
            section_text = clean_content[section_start:end_pos].strip()
            
            if section_text:
                extracted_sections[section] = section_text
                print(f"      「{section}」のテキストを抽出しました")
        
        if extracted_sections:
            # 抽出したセクションを結合
            ordered_text_parts = []
            for section in target_sections:
                if section in extracted_sections:
                    # セクション名から＜＞を除去
                    clean_section_name = section.replace('＜', '').replace('＞', '')
                    ordered_text_parts.append(f"--- {clean_section_name} ---")
                    # 改行で分割して空行を除去
                    lines = [line.strip() for line in extracted_sections[section].split('\n') if line.strip()]
                    ordered_text_parts.append('\n'.join(lines))
            
            summary_text = "\n".join(ordered_text_parts)
            
            # 出力ファイル名を作成
            output_file_name = f"{safe_filename}.txt"
            if len(output_file_name) > 100:  # ファイル名が長すぎる場合は短縮
                output_file_name = f"{safe_filename[:90]}....txt"
            
            output_file_path = os.path.join(patient_output_dir, output_file_name)
            
            # テキストを書き出し - 常にUTF-8で出力
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            
            print(f"    抽出完了: {output_file_name} (エンコーディング: {encoding})")
            print(f"    抽出されたセクション: {', '.join(extracted_sections.keys())}")
        else:
            print(f"    警告: {txt_file_name} に対象のセクションが見つかりませんでした")
            # セクションが見つからない場合は、HTMLタグを除去したテキスト全体を出力
            output_file_name = f"{safe_filename}.txt"
            if len(output_file_name) > 100:
                output_file_name = f"{safe_filename[:90]}....txt"
            
            output_file_path = os.path.join(patient_output_dir, output_file_name)
            
            # 改行で分割して空行を除去
            lines = [line.strip() for line in clean_content.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            
            print(f"    HTMLタグを除去した全テキストを出力: {output_file_name}")
        
    except Exception as e:
        print(f"    エラー: txtファイル {txt_file_name} の処理中に問題が発生しました: {str(e)}")
        return [txt_file]
    
    return []


def process_xml_file(xml_file, patient_output_dir, safe_filename, target_labels):
    """
    XMLファイルを処理して表示文字列を抽出する
    """
    error_files = []
    
    try:
        xml_file_name = os.path.splitext(os.path.basename(xml_file))[0]
        print(f"    処理中のXMLファイル: {xml_file_name}")
        
        # XMLファイルのエンコーディングを自動検出
        try:
            # バイナリモードでファイルの先頭部分を読み込み、エンコーディングを推測
            with open(xml_file, 'rb') as f:
                raw_data = f.read(10000)  # 最初の10000バイトを読み込み
                detected = chardet.detect(raw_data)
                encoding = detected['encoding']
                confidence = detected['confidence']
                
                print(f"      検出されたエンコーディング: {encoding} (信頼度: {confidence:.2f})")
                
                if encoding is None or confidence < 0.5:
                    # 検出されなかったか信頼度が低い場合は、一般的な日本語エンコーディングを試す
                    encoding = 'cp932'  # デフォルトはcp932（Windows用Shift-JIS拡張）
            
            # エンコーディングを指定してファイル全体を読み込む
            with codecs.open(xml_file, 'r', encoding=encoding, errors='replace') as f:
                xml_content = f.read()
            
            # XMLの解析
            root = ET.fromstring(xml_content)
            
            # 抽出対象の表示文字列とその次の部品の表示文字列を探す
            summary_text_parts = {}
            
            # 部品タグを探す（部品、コントロール、などの可能性がある）
            # 一般的にはこのようなXML構造を想定: <部品><表示文字列>ラベル</表示文字列></部品><部品><表示文字列>内容</表示文字列></部品>
            part_tags = ['部品', 'コントロール']
            
            for tag in part_tags:
                # 見つかったパーツのリスト
                parts = root.findall(f'.//{tag}')
                
                # パーツが見つからなかった場合は次のタグを試す
                if not parts:
                    continue
                
                # 対象ラベルを持つパーツとその次のパーツを探す
                for i in range(len(parts) - 1):  # 最後のパーツは次がないのでスキップ
                    current_part = parts[i]
                    next_part = parts[i + 1]
                    
                    # 現在のパーツの表示文字列を取得
                    display_text_elem = current_part.find('.//表示文字列')
                    if display_text_elem is not None and display_text_elem.text in target_labels:
                        label = display_text_elem.text
                        
                        # 次のパーツの表示文字列を取得
                        next_display_text_elem = next_part.find('.//表示文字列')
                        if next_display_text_elem is not None and next_display_text_elem.text:
                            content = next_display_text_elem.text
                            summary_text_parts[label] = content
                            print(f"      「{label}」の次の部品の表示文字列を抽出しました (タグ: {tag})")
            
            if summary_text_parts:
                # 抽出した表示文字列を順番に結合
                ordered_text_parts = []
                for label in target_labels:
                    if label in summary_text_parts:
                        ordered_text_parts.append(f"--- {label} ---")
                        ordered_text_parts.append(summary_text_parts[label])
                
                summary_text = "\n".join(ordered_text_parts)
                
                # 出力ファイル名を作成（XMLファイル名を基にする）
                # 長すぎるファイル名を避けるために短縮
                output_file_name = f"{safe_filename}.txt"
                if len(output_file_name) > 100:  # ファイル名が長すぎる場合は短縮
                    output_file_name = f"{safe_filename[:90]}....txt"
                
                output_file_path = os.path.join(patient_output_dir, output_file_name)
                
                # テキストを書き出し - 常にUTF-8で出力
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
                
                print(f"    抽出完了: {output_file_name} (エンコーディング: {encoding})")
                print(f"    抽出された表示文字列: {', '.join(summary_text_parts.keys())}")
            else:
                print(f"    警告: {xml_file_name} に対象の表示文字列が見つかりませんでした")
                # 代替処理は元のコードと同じロジックを使用
                
        except Exception as e:
            error_files.append(xml_file)
            print(f"    エラー: XMLファイル {xml_file_name} の処理中に問題が発生しました: {str(e)}")
    
    except Exception as e:
        error_files.append(xml_file)
        print(f"    エラー: XMLファイル処理中に予期しない問題が発生しました: {str(e)}")
    
    return error_files


def extract_zip_files(input_dir):
    """
    入力ディレクトリ内のsummaryディレクトリ以下のzipファイルを解凍する
    """
    print(f"\n入力ディレクトリ内のsummaryディレクトリ以下のzipファイルを解凍します: {input_dir}")
    
    # 患者IDごとのディレクトリを取得
    patient_dirs = [d for d in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(d)]
    
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        print(f"\n処理中: 患者ID {patient_id}")
        
        # summaryディレクトリ内のzipファイルを検索
        summary_dir = os.path.join(patient_dir, 'summary')
        if not os.path.exists(summary_dir):
            print(f"  summaryディレクトリが存在しません: {summary_dir}")
            continue
            
        zip_files = glob.glob(os.path.join(summary_dir, '*.zip'))
        
        if not zip_files:
            print(f"  zipファイルが見つかりません。既に解凍済みとみなして処理を継続します。")
            continue
        
        print(f"  見つかったzipファイル数: {len(zip_files)}")
        
        for zip_file in zip_files:
            try:
                zip_name = os.path.basename(zip_file)
                extract_dir = os.path.join(summary_dir, os.path.splitext(zip_name)[0])
                
                print(f"  解凍中: {zip_name}")
                print(f"  解凍先: {extract_dir}")
                
                # 解凍先ディレクトリが既に存在する場合は削除
                if os.path.exists(extract_dir):
                    print(f"    既存の解凍ディレクトリを削除します: {extract_dir}")
                    shutil.rmtree(extract_dir)
                
                # zipファイルを解凍
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                print(f"    解凍完了: {zip_name}")
                
            except Exception as e:
                print(f"    エラー: {zip_name} の解凍中に問題が発生しました: {str(e)}")
    
    print("\nzipファイルの解凍が完了しました")

def extract_summary_from_zip(input_dir, output_dir):
    """
    1. 患者IDごとにzipファイルを解凍
    2. XMLファイルから特定の「表示文字列」タグ内のテキストを抽出
       - 「表示文字列」が「入院中の経過及び看護上の問題経過」の部品タグ内の「表示文字列」とその次の部品タグ内の「表示文字列」
       - 「表示文字列」が「備考」の部品タグ内の「表示文字列」とその次の部品タグ内の「表示文字列」
       - 「表示文字列」が「入院から退院までの経過」の部品タグ内の「表示文字列」とその次の部品タグ内の「表示文字列」
    3. 結果を指定された出力ディレクトリに保存
    """
    # まず入力ディレクトリ内のzipファイルを解凍
    extract_zip_files(input_dir)
    
    # 抽出対象の表示文字列の内容
    target_labels = ["入院中の経過及び看護上の問題経過", "備考", "入院から退院までの経過"]
    
    print(f"データディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"抽出対象の表示文字列: {', '.join(target_labels)}")
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # エラーファイルリスト
    error_files = []
    
    # 患者IDごとのディレクトリを取得
    patient_dirs = [d for d in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(d)]
    
    print(f"見つかった患者ディレクトリ: {len(patient_dirs)}")
    
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        print(f"\n処理中: 患者ID {patient_id}")
        
        # この患者用の出力ディレクトリを作成
        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Zipファイルのパターンを検索
        zip_files = glob.glob(os.path.join(patient_dir, 'summary', '*.zip'))
        
        print(f"  見つかったZIPファイル: {len(zip_files)}")
        
        if not zip_files:
            # zipファイルがない場合は、txtファイルを優先し、なければXMLファイルを処理
            print(f"  zipファイルが見つかりません。既に解凍されたファイルを直接処理します。")
            summary_dir = os.path.join(patient_dir, 'summary')
            
            # まずtxtファイルを検索
            txt_files = glob.glob(os.path.join(summary_dir, '**', '*.txt'), recursive=True)
            
            if txt_files:
                print(f"  見つかったtxtファイル数: {len(txt_files)}")
                for txt_file in txt_files:
                    try:
                        txt_file_name = os.path.splitext(os.path.basename(txt_file))[0]
                        safe_filename = re.sub(r'[\\/*?:"<>|]', "_", txt_file_name)
                        
                        # txtファイルの内容をそのまま読み込んで出力
                        process_txt_file(txt_file, patient_output_dir, safe_filename)
                        
                    except Exception as e:
                        print(f"    エラー: txtファイル {os.path.basename(txt_file)} の処理中に問題が発生しました: {str(e)}")
            else:
                # txtファイルがない場合はXMLファイルを処理
                xml_files = glob.glob(os.path.join(summary_dir, '**', '*.xml'), recursive=True)
                
                if xml_files:
                    print(f"  txtファイルが見つからないため、XMLファイルを処理します。")
                    print(f"  見つかったXMLファイル数: {len(xml_files)}")
                    for xml_file in xml_files:
                        try:
                            xml_file_name = os.path.splitext(os.path.basename(xml_file))[0]
                            safe_filename = re.sub(r'[\\/*?:"<>|]', "_", xml_file_name)
                            
                            # XMLファイル処理のロジックを呼び出し
                            process_xml_file(xml_file, patient_output_dir, safe_filename, target_labels)
                            
                        except Exception as e:
                            print(f"    エラー: XMLファイル {os.path.basename(xml_file)} の処理中に問題が発生しました: {str(e)}")
                else:
                    print(f"  txtファイルもXMLファイルも見つかりませんでした。患者ID: {patient_id} をスキップします。")
            continue
        
        for zip_file_path in zip_files:
            try:
                print(f"  処理中のZIPファイル: {os.path.basename(zip_file_path)}")
                
                # Zipファイル名（拡張子なし）を取得して、安全なファイル名にする
                zip_file_name = os.path.splitext(os.path.basename(zip_file_path))[0]
                safe_filename = re.sub(r'[\\/*?:"<>|]', "_", zip_file_name)  # 安全でない文字を置換
                
                # 一時解凍ディレクトリ
                temp_extract_dir = os.path.join(patient_dir, 'summary', f'temp_extract_{safe_filename}')
                os.makedirs(temp_extract_dir, exist_ok=True)
                
                # Zipファイルを解凍
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                
                # XMLファイルを検索
                xml_files = glob.glob(os.path.join(temp_extract_dir, '**', '*.xml'), recursive=True)
                
                print(f"    解凍後のXMLファイル数: {len(xml_files)}")
                
                if not xml_files and os.path.exists(os.path.join(patient_dir, 'summary', f'{zip_file_name}.xml')):
                    # もし解凍されたディレクトリにXMLがなく、同じ場所に対応するXMLがある場合
                    xml_files = [os.path.join(patient_dir, 'summary', f'{zip_file_name}.xml')]
                    print(f"    ZIPファイル内にXMLが見つからなかったため、対応するXMLファイルを使用: {xml_files[0]}")
                
                for xml_file in xml_files:
                    try:
                        xml_file_name = os.path.splitext(os.path.basename(xml_file))[0]
                        print(f"    処理中のXMLファイル: {xml_file_name}")
                        
                        # XMLファイルのエンコーディングを自動検出
                        try:
                            # バイナリモードでファイルの先頭部分を読み込み、エンコーディングを推測
                            with open(xml_file, 'rb') as f:
                                raw_data = f.read(10000)  # 最初の10000バイトを読み込み
                                detected = chardet.detect(raw_data)
                                encoding = detected['encoding']
                                confidence = detected['confidence']
                                
                                print(f"      検出されたエンコーディング: {encoding} (信頼度: {confidence:.2f})")
                                
                                if encoding is None or confidence < 0.5:
                                    # 検出されなかったか信頼度が低い場合は、一般的な日本語エンコーディングを試す
                                    encoding = 'cp932'  # デフォルトはcp932（Windows用Shift-JIS拡張）
                            
                            # エンコーディングを指定してファイル全体を読み込む
                            with codecs.open(xml_file, 'r', encoding=encoding, errors='replace') as f:
                                xml_content = f.read()
                            
                            # XMLの解析
                            root = ET.fromstring(xml_content)
                            
                            # 抽出対象の表示文字列とその次の部品の表示文字列を探す
                            summary_text_parts = {}
                            
                            # 部品タグを探す（部品、コントロール、などの可能性がある）
                            # 一般的にはこのようなXML構造を想定: <部品><表示文字列>ラベル</表示文字列></部品><部品><表示文字列>内容</表示文字列></部品>
                            part_tags = ['部品', 'コントロール']
                            
                            for tag in part_tags:
                                # 見つかったパーツのリスト
                                parts = root.findall(f'.//{tag}')
                                
                                # パーツが見つからなかった場合は次のタグを試す
                                if not parts:
                                    continue
                                
                                # 対象ラベルを持つパーツとその次のパーツを探す
                                for i in range(len(parts) - 1):  # 最後のパーツは次がないのでスキップ
                                    current_part = parts[i]
                                    next_part = parts[i + 1]
                                    
                                    # 現在のパーツの表示文字列を取得
                                    display_text_elem = current_part.find('.//表示文字列')
                                    if display_text_elem is not None and display_text_elem.text in target_labels:
                                        label = display_text_elem.text
                                        
                                        # 次のパーツの表示文字列を取得
                                        next_display_text_elem = next_part.find('.//表示文字列')
                                        if next_display_text_elem is not None and next_display_text_elem.text:
                                            content = next_display_text_elem.text
                                            summary_text_parts[label] = content
                                            print(f"      「{label}」の次の部品の表示文字列を抽出しました (タグ: {tag})")
                            
                            if summary_text_parts:
                                # 抽出した表示文字列を順番に結合
                                ordered_text_parts = []
                                for label in target_labels:
                                    if label in summary_text_parts:
                                        ordered_text_parts.append(f"--- {label} ---")
                                        ordered_text_parts.append(summary_text_parts[label])
                                
                                summary_text = "\n".join(ordered_text_parts)
                                
                                # 出力ファイル名を作成（XMLファイル名を基にする）
                                # 長すぎるファイル名を避けるために短縮
                                output_file_name = f"{safe_filename}.txt"
                                if len(output_file_name) > 100:  # ファイル名が長すぎる場合は短縮
                                    output_file_name = f"{safe_filename[:90]}....txt"
                                
                                output_file_path = os.path.join(patient_output_dir, output_file_name)
                                
                                # テキストを書き出し - 常にUTF-8で出力
                                with open(output_file_path, 'w', encoding='utf-8') as f:
                                    f.write(summary_text)
                                
                                print(f"    抽出完了: {output_file_name} (エンコーディング: {encoding})")
                                print(f"    抽出された表示文字列: {', '.join(summary_text_parts.keys())}")
                            else:
                                print(f"    警告: {xml_file_name} に対象の表示文字列が見つからなかったか、部品タグの構造が想定と異なります")
                                
                                # 代替処理: XMLファイルをテキストとして行ごとに読み込み、必要な情報を抽出
                                print(f"    代替処理: テキスト解析を試みます...")
                                try:
                                    with codecs.open(xml_file, 'r', encoding=encoding, errors='replace') as f:
                                        content = f.read()
                                    
                                    # 部品タグを特定するための正規表現パターンを作成
                                    # 複数の可能性のあるタグ名で試す
                                    patterns = []
                                    for tag in part_tags:
                                        open_tag = f'<{tag}>'
                                        close_tag = f'</{tag}>'
                                        # パターン: <部品>...<表示文字列>ラベル</表示文字列>...</部品><部品>...<表示文字列>内容</表示文字列>...</部品>
                                        patterns.append((tag, open_tag, close_tag))
                                    
                                    summary_text_parts = {}
                                    
                                    for tag_name, open_tag, close_tag in patterns:
                                        # すべての部品タグを抽出
                                        part_pattern = f'{re.escape(open_tag)}(.*?){re.escape(close_tag)}'
                                        parts = re.findall(part_pattern, content, re.DOTALL)
                                        
                                        if not parts:
                                            continue
                                        
                                        # 各部品の表示文字列を抽出
                                        for i in range(len(parts) - 1):  # 最後の部品は次がないのでスキップ
                                            current_part = parts[i]
                                            next_part = parts[i + 1]
                                            
                                            # 現在の部品から表示文字列を抽出
                                            display_text_match = re.search(r'<表示文字列>(.*?)</表示文字列>', current_part, re.DOTALL)
                                            if display_text_match and display_text_match.group(1) in target_labels:
                                                label = display_text_match.group(1)
                                                
                                                # 次の部品から表示文字列を抽出
                                                next_display_text_match = re.search(r'<表示文字列>(.*?)</表示文字列>', next_part, re.DOTALL)
                                                if next_display_text_match and next_display_text_match.group(1):
                                                    content = next_display_text_match.group(1)
                                                    summary_text_parts[label] = content
                                                    print(f"      「{label}」の次の部品の表示文字列を正規表現で抽出しました (タグ: {tag_name})")
                                    
                                    if summary_text_parts:
                                        # 抽出した表示文字列を順番に結合
                                        ordered_text_parts = []
                                        for label in target_labels:
                                            if label in summary_text_parts:
                                                ordered_text_parts.append(f"--- {label} ---")
                                                ordered_text_parts.append(summary_text_parts[label])
                                        
                                        summary_text = "\n".join(ordered_text_parts)
                                        
                                        output_file_name = f"{safe_filename}.txt"
                                        if len(output_file_name) > 100:
                                            output_file_name = f"{safe_filename[:90]}....txt"
                                        
                                        output_file_path = os.path.join(patient_output_dir, output_file_name)
                                        
                                        with open(output_file_path, 'w', encoding='utf-8') as f:
                                            f.write(summary_text)
                                        
                                        print(f"    抽出完了: {output_file_name} (テキスト解析による抽出)")
                                        print(f"    抽出された表示文字列: {', '.join(summary_text_parts.keys())}")
                                    else:
                                        # 最後の手段: 単純に全ての表示文字列を抽出して順序で判断
                                        text_contents = []
                                        text_matches = re.findall(r'<表示文字列>(.*?)</表示文字列>', content, re.DOTALL)
                                        for text in text_matches:
                                            if text.strip():
                                                text_contents.append(text.strip())
                                        
                                        # 対象の表示文字列とその次の表示文字列を探す
                                        summary_text_parts = {}
                                        for i, text in enumerate(text_contents):
                                            if text in target_labels and i + 1 < len(text_contents):
                                                label = text
                                                content = text_contents[i + 1]
                                                summary_text_parts[label] = content
                                                print(f"      「{label}」の次の表示文字列を抽出しました (単純な順序検索)")
                                        
                                        if summary_text_parts:
                                            # 抽出した表示文字列を順番に結合
                                            ordered_text_parts = []
                                            for label in target_labels:
                                                if label in summary_text_parts:
                                                    ordered_text_parts.append(f"--- {label} ---")
                                                    ordered_text_parts.append(summary_text_parts[label])
                                            
                                            summary_text = "\n".join(ordered_text_parts)
                                            
                                            output_file_name = f"{safe_filename}.txt"
                                            if len(output_file_name) > 100:
                                                output_file_name = f"{safe_filename[:90]}....txt"
                                            
                                            output_file_path = os.path.join(patient_output_dir, output_file_name)
                                            
                                            with open(output_file_path, 'w', encoding='utf-8') as f:
                                                f.write(summary_text)
                                            
                                            print(f"    抽出完了: {output_file_name} (単純な順序検索による抽出)")
                                            print(f"    抽出された表示文字列: {', '.join(summary_text_parts.keys())}")
                                        else:
                                            error_files.append((patient_id, xml_file))
                                            print(f"    エラー: すべての抽出方法を試しましたが、対象の表示文字列が抽出できませんでした")
                                
                                except Exception as e:
                                    error_files.append((patient_id, xml_file))
                                    print(f"    エラー: テキスト解析中に問題が発生しました: {str(e)}")
                        
                        except Exception as e:
                            error_files.append((patient_id, xml_file))
                            print(f"    エラー: XMLファイル {xml_file_name} の処理中に問題が発生しました: {str(e)}")
                            print(f"    詳細: {e.__class__.__name__}")
                            # ファイルの情報を表示
                            try:
                                file_size = os.path.getsize(xml_file)
                                print(f"      ファイルサイズ: {file_size} バイト")
                                
                                # バイナリモードでファイルの一部を読み込んで16進数表示
                                with open(xml_file, 'rb') as f:
                                    header = f.read(100)  # 先頭100バイトを読み込み
                                    hex_header = ' '.join(f'{b:02x}' for b in header)
                                    print(f"      ファイルヘッダ(16進数): {hex_header[:100]}...")
                            except Exception as file_e:
                                print(f"      ファイル情報取得エラー: {str(file_e)}")
                    
                    except Exception as e:
                        error_files.append((patient_id, xml_file))
                        print(f"    エラー: XMLファイル {xml_file_name} の処理中に予期しない問題が発生しました: {str(e)}")
                
                # 一時ディレクトリを削除
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
                
            except Exception as e:
                print(f"    エラー: Zipファイル {os.path.basename(zip_file_path)} の処理中に問題が発生しました: {str(e)}")
    
    print("\nすべての処理が完了しました")
    
    if error_files:
        print(f"\n処理に失敗したファイル数: {len(error_files)}")
        print("処理に失敗したファイルのリスト:")
        for patient_id, file_path in error_files:
            print(f"  患者ID: {patient_id}, ファイル: {os.path.basename(file_path)}")

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='Extract summary from Hanwa Kinen Hospital XML files')
    parser.add_argument('--input', '-i', required=True,
                      help='Input directory containing patient data')
    parser.add_argument('--output', '-o', required=True,
                      help='Output directory for processed files')
    
    args = parser.parse_args()
    
    # データ処理を実行
    extract_summary_from_zip(args.input, args.output)
    print("Processing completed!")

if __name__ == "__main__":
    main() 