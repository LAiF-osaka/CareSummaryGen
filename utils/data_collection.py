import argparse
import os
import re
import zipfile

from bs4 import BeautifulSoup


def extract_zip_files(root_dir):
    """
    root_dir以下の全てのディレクトリを走査し、zipファイルを見つけたら、
    zipファイルがあるディレクトリの「1つ上の階層」に対して、
    「日付フォルダ/機能フォルダ」に整理して展開します。

    例：
      zipファイルパス: <some_path>/deep_folder/521_20230211_141228_2_0_0_....zip
      → 親ディレクトリ: <some_path>/deep_folder の1つ上の階層 (つまり <some_path>)
      → 展開先: <some_path>/20230211/521/
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".zip"):
                zip_path = os.path.join(dirpath, filename)

                # ファイル名を"_"で分割して、先頭2つを取り出す
                parts = filename.split("_")
                if len(parts) < 2:
                    print(
                        f"スキップ: {filename} (ファイル名の形式が想定と異なります)"
                    )
                    continue

                # parts[0] が機能番号、parts[1] が日付であると仮定
                function = parts[0]
                date = parts[1]

                # zipファイルがあるディレクトリの1つ上の階層を取得
                parent_dir = os.path.dirname(dirpath)

                # 展開先ディレクトリを「日付フォルダ/機能フォルダ」として指定
                extract_dir = os.path.join(parent_dir, date, function)
                os.makedirs(extract_dir, exist_ok=True)

                print(f"Extracting {zip_path} to {extract_dir}")
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                except Exception as e:
                    print(f"Error extracting {zip_path}: {e}")


import os


def aggregate_txt_to_markdown(root_dir):
    """
    ルートディレクトリ直下にある各IDフォルダ内を走査し、
    「日付フォルダ/機能フォルダ」内のすべての.txtファイルの内容を読み込み、
    Markdown形式で集約した内容を、各IDフォルダ直下に「ID名.md」として出力します。

    出力するMarkdownの構造例:
      - 日付
        - 機能
          - 結合したテキスト
    """
    # ルート直下の各フォルダをIDフォルダとみなす
    for id_name in os.listdir(root_dir):
        id_path = os.path.join(root_dir, id_name)
        if not os.path.isdir(id_path):
            continue  # フォルダでなければスキップ

        md_lines = []  # Markdownの行を保持するリスト

        # IDフォルダ内の「日付フォルダ」を走査（昇順に処理）
        for date_name in sorted(os.listdir(id_path)):
            date_path = os.path.join(id_path, date_name)
            if not os.path.isdir(date_path):
                continue

            # 日付フォルダのタイトル行を追加
            md_lines.append(f"- {date_name}")

            # 日付フォルダ内の各「機能フォルダ」を走査
            content_index = 1
            for func_name in sorted(os.listdir(date_path)):
                func_path = os.path.join(date_path, func_name)
                if not os.path.isdir(func_path):
                    continue

                # 機能フォルダ内のすべての.txtファイルの内容を結合
                combined_text = ""
                for file in sorted(os.listdir(func_path)):
                    if file.lower().endswith(".txt"):
                        file_path = os.path.join(func_path, file)
                        try:
                            with open(file_path, "r", encoding="cp932") as f:
                                # 各テキストファイルの内容の前後の余分な空白を除去して結合
                                content = f.read().strip()
                                if content:
                                    # 改行を入れて区切る
                                    combined_text += content + "\n"
                        except Exception as e:
                            print(
                                f"ファイル {file_path} の読み込みに失敗しました: {e}"
                            )
                combined_text = re.sub(r"<[^>]*>", "", combined_text)

                # もし結合したテキストがある場合は、機能フォルダの項目として追加
                if combined_text.strip():
                    # インデントを2段階にして機能名を出力
                    md_lines.append(f"  - カルテ#{content_index}")
                    content_index += 1
                    # さらにインデントを1段階追加してテキスト内容を出力
                    # ※内容が複数行の場合、各行の先頭に追加のインデントを入れる
                    for line in [
                        l.strip() for l in combined_text.splitlines()
                    ]:
                        if line:
                            md_lines.append(f"    {line}")
                else:
                    # .txtファイルが存在しないか、内容が空の場合はスキップ
                    continue

        # Markdownの内容がある場合はIDフォルダ直下にID名.mdとして出力
        if md_lines:
            md_filename = f"{id_name}.md"
            md_filepath = os.path.join(id_path, md_filename)
            try:
                with open(md_filepath, "w", encoding="utf-8") as md_file:
                    md_text = "\n".join(md_lines)
                    soup = BeautifulSoup(md_text, "html.parser")
                    context = soup.get_text()
                    md_file.write(context)
                print(f"Markdownファイルを作成しました: {md_filepath}")
            except Exception as e:
                print(
                    f"Markdownファイルの作成に失敗しました {md_filepath}: {e}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データ収集")
    parser.add_argument("--unzip", action="store_true")
    args = parser.parse_args()
    # 検索開始するディレクトリを指定してください
    # 例: "C:\\Users\\User\\Documents\\ZipFilesRoot" など
    root_directory = r"C:\Users\Administrator\CareSummaryGen\data\hanwa_2"
    if args.unzip:
        extract_zip_files(root_directory)
    # ルートディレクトリのパスを指定してください
    aggregate_txt_to_markdown(root_directory)
