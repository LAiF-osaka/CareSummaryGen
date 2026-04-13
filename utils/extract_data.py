import codecs  # codecsをインポート
import glob
import os
import re
import xml.etree.ElementTree as ET  # XML解析のためにETをインポート

import chardet  # chardetをインポート
from bs4 import BeautifulSoup


def clean_html_tags(raw_html):
    """
    文字列からHTMLタグを除去する (BeautifulSoupを使用)。
    改行はある程度保持されるように調整。
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    # separatorを改行文字に変更し、strip=Trueで前後の余分な空白や改行を除去
    clean_text = soup.get_text(separator="\n", strip=True)
    return clean_text


def extract_summary_from_xml(base_input_dir, target_patient_id_only=None):
    """
    指定された入力ディレクトリ内の患者IDフォルダを走査し、その中のsummaryフォルダからXMLファイルを処理する。
    特定の患者IDのみを対象とすることも可能。
    抽出したテキストデータを辞書として返す。

    Args:
        base_input_dir (str): 全ての患者IDフォルダが含まれるルート入力ディレクトリ。
        target_patient_id_only (str, optional): 指定された場合、この患者IDのみを処理する。デフォルトはNone。

    Returns:
        tuple: (extracted_data, error_files)
               extracted_data (dict): {patient_id: {xml_filename_without_ext: "extracted text", ...}}
               error_files (list): (patient_id, xml_file_path) のタプルで構成されるエラーファイルのリスト。
    """
    target_labels = ["入院中の経過及び看護上の問題経過", "備考"]
    part_tags = ["部品", "コントロール"]

    print(f"XML抽出開始: 入力ルート = {base_input_dir}")
    if target_patient_id_only:
        print(f"対象患者ID: {target_patient_id_only}")

    all_patients_extracted_data = {}
    error_files = []
    patient_ids_to_process = []

    if target_patient_id_only:
        if os.path.isdir(os.path.join(base_input_dir, target_patient_id_only)):
            patient_ids_to_process.append(target_patient_id_only)
        else:
            print(
                f"  警告: 指定された対象患者IDフォルダ {os.path.join(base_input_dir, target_patient_id_only)} が見つかりません。"
            )
            return {}, error_files
    else:
        patient_ids_to_process = [
            d
            for d in os.listdir(base_input_dir)
            if os.path.isdir(os.path.join(base_input_dir, d))
        ]

    print(f"XML処理対象の患者IDフォルダ数: {len(patient_ids_to_process)}")

    for patient_id in patient_ids_to_process:
        print(f"\nXML抽出中: 患者ID {patient_id}")
        patient_input_folder_path = os.path.join(base_input_dir, patient_id)
        summary_folder_path = os.path.join(
            patient_input_folder_path, "summary"
        )

        current_patient_extracted_data = {}

        if not os.path.isdir(summary_folder_path):
            print(
                f"  警告: {patient_id} 内に 'summary' フォルダが見つかりません。スキップします。"
            )
            all_patients_extracted_data[patient_id] = (
                current_patient_extracted_data
            )
            continue

        xml_files = glob.glob(
            os.path.join(summary_folder_path, "**", "*.xml"), recursive=True
        )
        print(f"  見つかったXMLファイル (サブフォルダ含む): {len(xml_files)}")

        if not xml_files:
            print(
                f"  情報: {summary_folder_path} (サブフォルダ含む) 内に .xml ファイルが見つかりませんでした。"
            )
            all_patients_extracted_data[patient_id] = (
                current_patient_extracted_data
            )
            continue

        for xml_file_path in xml_files:
            extracted_content_for_dict = None
            xml_file_name_with_ext = os.path.basename(xml_file_path)
            xml_file_name_only = os.path.splitext(xml_file_name_with_ext)[0]
            # safe_xml_filename_only = re.sub(r'[\\/*?:"<>|]', "_", xml_file_name_only) # ファイル名にするわけではないので不要かも

            print(f"    処理中のXMLファイル: {xml_file_path}")
            encoding = "cp932"

            try:
                try:
                    with open(xml_file_path, "rb") as f_rb:
                        raw_data = f_rb.read(10000)
                        detected = chardet.detect(raw_data)
                        confidence = detected.get("confidence", 0)
                        detected_encoding = detected.get("encoding")
                        print(
                            f"      Chardet検出エンコーディング: {detected_encoding} (信頼度: {confidence:.2f})"
                        )
                        if detected_encoding and confidence > 0.7:
                            encoding = detected_encoding
                        else:
                            encodings_to_try_xml = [
                                "utf-8",
                                "utf-8-sig",
                                "cp932",
                                "shift_jis",
                                "euc_jp",
                                "iso2022_jp",
                            ]
                            found_enc = False
                            for enc_try in encodings_to_try_xml:
                                try:
                                    with codecs.open(
                                        xml_file_path,
                                        "r",
                                        encoding=enc_try,
                                        errors="strict",
                                    ) as test_f:
                                        test_f.read(100)
                                    encoding = enc_try
                                    found_enc = True
                                    print(
                                        f"      試行エンコーディング '{encoding}' で読み込み成功。"
                                    )
                                    break
                                except (UnicodeDecodeError, Exception):
                                    continue
                            if not found_enc:
                                print(
                                    f"      エンコーディングの特定に失敗。デフォルト '{encoding}' を使用。"
                                )
                except Exception as e_enc_detect:
                    print(
                        f"      エンコーディング検出中にエラー: {e_enc_detect}。デフォルト '{encoding}' を使用。"
                    )

                with codecs.open(
                    xml_file_path, "r", encoding=encoding, errors="replace"
                ) as f_xml:
                    xml_content = f_xml.read()

                root = ET.fromstring(xml_content)
                summary_text_parts = {}

                for tag_name_xml in part_tags:
                    parts = root.findall(f".//{tag_name_xml}")
                    if not parts:
                        continue
                    for i in range(len(parts) - 1):
                        current_part = parts[i]
                        next_part = parts[i + 1]
                        display_text_elem = current_part.find(".//表示文字列")
                        if (
                            display_text_elem is not None
                            and display_text_elem.text in target_labels
                        ):
                            label = display_text_elem.text
                            next_display_text_elem = next_part.find(
                                ".//表示文字列"
                            )
                            if (
                                next_display_text_elem is not None
                                and next_display_text_elem.text
                            ):
                                summary_text_parts[label] = (
                                    next_display_text_elem.text
                                )
                                # print(f"      XML解析: 「{label}」の次の部品の表示文字列を抽出 (タグ: {tag_name_xml})")

                if summary_text_parts:
                    ordered_text_parts = []
                    for label in target_labels:
                        if label in summary_text_parts:
                            ordered_text_parts.append(f"--- {label} ---")
                            ordered_text_parts.append(
                                summary_text_parts[label]
                            )
                    extracted_content_for_dict = "\n".join(ordered_text_parts)
                else:
                    # print(f"    XML構造解析で対象データ見つからず。正規表現によるテキスト解析を試行...")
                    summary_text_parts_regex = {}
                    for tag_name_regex in part_tags:
                        part_pattern_regex = (
                            f"<{tag_name_regex}[^>]*>(.*?)</{tag_name_regex}>"
                        )
                        display_text_pattern_regex = (
                            r"<表示文字列>(.*?)</表示文字列>"
                        )

                        all_parts_content = re.findall(
                            part_pattern_regex, xml_content, re.DOTALL
                        )
                        if not all_parts_content:
                            continue

                        for i in range(len(all_parts_content) - 1):
                            current_part_content = all_parts_content[i]
                            next_part_content = all_parts_content[i + 1]

                            current_display_match = re.search(
                                display_text_pattern_regex,
                                current_part_content,
                                re.DOTALL,
                            )
                            if (
                                current_display_match
                                and current_display_match.group(1).strip()
                                in target_labels
                            ):
                                label_regex = current_display_match.group(
                                    1
                                ).strip()
                                next_display_match = re.search(
                                    display_text_pattern_regex,
                                    next_part_content,
                                    re.DOTALL,
                                )
                                if (
                                    next_display_match
                                    and next_display_match.group(1)
                                ):
                                    summary_text_parts_regex[label_regex] = (
                                        next_display_match.group(1).strip()
                                    )
                                    # print(f"      正規表現: 「{label_regex}」の次の部品の表示文字列を抽出 (タグ: {tag_name_regex})")

                    if summary_text_parts_regex:
                        ordered_text_parts_regex = []
                        for label_regex_out in target_labels:
                            if label_regex_out in summary_text_parts_regex:
                                ordered_text_parts_regex.append(
                                    f"--- {label_regex_out} ---"
                                )
                                ordered_text_parts_regex.append(
                                    summary_text_parts_regex[label_regex_out]
                                )
                        extracted_content_for_dict = "\n".join(
                            ordered_text_parts_regex
                        )

                if extracted_content_for_dict:
                    current_patient_extracted_data[xml_file_name_only] = (
                        extracted_content_for_dict
                    )
                    print(
                        f"      XMLからデータを抽出: {xml_file_name_with_ext} (元エンコーディング: {encoding})"
                    )
                else:
                    print(
                        f"    警告: XMLファイル {xml_file_name_with_ext} から対象データを抽出できませんでした。"
                    )
                    error_files.append((patient_id, xml_file_path))

            except ET.ParseError as e_parse:
                print(
                    f"    エラー: XMLファイル {xml_file_name_with_ext} の解析中に問題が発生 (エンコーディング: {encoding}): {e_parse}"
                )
                error_files.append((patient_id, xml_file_path))
            except Exception as e_xml_proc:
                print(
                    f"    エラー: XMLファイル {xml_file_name_with_ext} の処理中に予期せぬ問題が発生: {e_xml_proc}"
                )
                error_files.append((patient_id, xml_file_path))

        all_patients_extracted_data[patient_id] = (
            current_patient_extracted_data
        )

    print(f"XML抽出完了。エラーファイル数: {len(error_files)}")
    return all_patients_extracted_data, error_files


def process_summary_files(input_root_folder, output_root_folder):
    """
    指定したフォルダ内の患者IDフォルダを走査し、その中のsummaryフォルダにある
    .txtファイルを読み取りHTMLタグを除去、または.xmlファイルから特定情報を抽出してテキスト化し、
    指定したアウトプットフォルダに患者IDフォルダを作成して保存する。
    summaryフォルダ内のサブフォルダも再帰的に探索する。
    """
    print(
        f"処理開始: 入力ルート = {input_root_folder}, 出力ルート = {output_root_folder}"
    )
    os.makedirs(output_root_folder, exist_ok=True)
    patient_id_folders = [
        d
        for d in os.listdir(input_root_folder)
        if os.path.isdir(os.path.join(input_root_folder, d))
    ]

    if not patient_id_folders:
        print(
            f"警告: {input_root_folder} 内に患者IDフォルダが見つかりませんでした。"
        )
        return

    print(f"見つかった患者IDフォルダ数: {len(patient_id_folders)}")

    for patient_id in patient_id_folders:
        print(f"\n処理中の患者ID: {patient_id}")
        patient_folder_path = os.path.join(input_root_folder, patient_id)
        summary_folder_path = os.path.join(patient_folder_path, "summary")
        patient_output_folder = os.path.join(output_root_folder, patient_id)
        os.makedirs(patient_output_folder, exist_ok=True)

        if not os.path.isdir(summary_folder_path):
            print(
                f"  警告: {patient_id} 内に 'summary' フォルダが見つかりません。スキップします。"
            )
            continue

        txt_files = glob.glob(
            os.path.join(summary_folder_path, "**", "*.txt"), recursive=True
        )
        processed_content_for_patient = (
            {}
        )  # {original_filename_without_ext: content}

        if txt_files:
            print(
                f"  見つかった.txtファイル数 (サブフォルダ含む): {len(txt_files)}"
            )
            for txt_file_path in txt_files:
                original_filename_with_ext = os.path.basename(txt_file_path)
                original_filename_only = os.path.splitext(
                    original_filename_with_ext
                )[0]
                print(f"    処理中のTXTファイル: {txt_file_path}")
                raw_content = None
                used_encoding = None
                encodings_to_try = [
                    "utf-8",
                    "cp932",
                    "shift_jis",
                    "euc_jp",
                    "iso2022_jp",
                ]
                for enc in encodings_to_try:
                    try:
                        with open(txt_file_path, "r", encoding=enc) as f:
                            raw_content = f.read()
                        used_encoding = enc
                        print(
                            f"      TXTファイル '{original_filename_with_ext}' をエンコーディング '{enc}' で読み込み成功。"
                        )
                        break
                    except UnicodeDecodeError:
                        # print(f"      TXTファイル '{original_filename_with_ext}' のエンコーディング '{enc}' での読み込み失敗。")
                        pass
                    except Exception as e_read:
                        print(
                            f"      TXTファイル '{original_filename_with_ext}' の読み込み中に予期せぬエラー ({enc}): {e_read}"
                        )
                        break

                if raw_content is not None:
                    try:
                        cleaned_content = clean_html_tags(raw_content)
                        processed_content_for_patient[
                            original_filename_only
                        ] = cleaned_content
                        print(
                            f"      TXTファイルからデータを準備: {original_filename_with_ext} (元エンコーディング: {used_encoding or '不明'})"
                        )
                    except Exception as e_clean:
                        print(
                            f"      TXTファイル {txt_file_path} のHTMLクリーン処理中にエラー: {e_clean}"
                        )
                else:
                    print(
                        f"      エラー: TXTファイル {txt_file_path} を既知のエンコーディングで読み込めませんでした。"
                    )
        else:
            print(
                f"  情報: {summary_folder_path} (サブフォルダ含む) 内に .txt ファイルが見つかりませんでした。XMLファイルの処理を試みます。"
            )
            xml_extracted_data, xml_errors = extract_summary_from_xml(
                input_root_folder, target_patient_id_only=patient_id
            )

            if (
                patient_id in xml_extracted_data
                and xml_extracted_data[patient_id]
            ):
                for xml_filename_key, content_from_xml in xml_extracted_data[
                    patient_id
                ].items():
                    processed_content_for_patient[xml_filename_key] = (
                        content_from_xml  # xml_filename_key は拡張子なし
                    )
                    print(
                        f"      XMLファイルからデータを準備: {xml_filename_key}.xml"
                    )
            if xml_errors:
                print(
                    f"  患者ID {patient_id} のXML処理中にエラーが発生したファイルがありました。詳細は上記ログを確認してください。"
                )

        # 共通のファイル保存処理
        if processed_content_for_patient:
            for (
                filename_key,
                content_to_save,
            ) in processed_content_for_patient.items():
                output_filename_txt = f"{filename_key}.txt"  # 保存は常に.txt
                output_file_path = os.path.join(
                    patient_output_folder, output_filename_txt
                )
                try:
                    with open(
                        output_file_path, "w", encoding="utf-8"
                    ) as outfile:
                        outfile.write(content_to_save)
                    print(
                        f"    抽出/処理されたデータを {output_file_path} に保存しました。"
                    )
                except Exception as e_save:
                    print(
                        f"    エラー: ファイル {output_file_path} の書き込み中にエラー: {e_save}"
                    )
        else:
            print(
                f"  患者ID {patient_id} に関して処理・保存するデータがありませんでした。"
            )

    print("\nすべての処理が完了しました。")


if __name__ == "__main__":
    input_dir_main = os.path.join("D:", "work", "ASC", "新記念_20250610")
    output_dir_main = os.path.join(
        "D:", "work", "LAiF", "2_gspreprocess", "summary_processed_v2"
    )
    process_summary_files(input_dir_main, output_dir_main)
