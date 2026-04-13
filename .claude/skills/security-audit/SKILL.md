---
name: security-audit
description: セキュリティ監査ワークフロー。OWASP Top 10・認証認可・秘密情報漏洩を検査する
disable-model-invocation: true
---

# Security Audit Workflow

コードベース全体の包括的なセキュリティ監査を実施してください。

## 進め方

### 監査対象
- `app.py` — Flask アプリケーションのエンドポイント・認証・バリデーション
- プロジェクト全体 — インジェクション・秘匿情報漏洩・依存関係
- `.env.example` — 必要な環境変数の確認
- `pyproject.toml` — 依存ライブラリの脆弱性

### チェックポイント
1. OWASP Top 10 全項目
2. API キー・接続文字列のハードコードがないか
3. アップロードファイルのバリデーション
4. セッション管理の安全性
5. CORS 設定の妥当性
6. LangChain のプロンプトインジェクション対策

## 完了報告

セキュリティレポートを以下の形式でユーザーに報告:
- Critical / High / Medium / Low / Informational の件数サマリ
- Critical・High の詳細と修正提案
- 全体的なセキュリティ評価

引数 `$ARGUMENTS` が指定されている場合は、そのファイル/ディレクトリに絞って監査する。
