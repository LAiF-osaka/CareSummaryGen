---
name: review
description: コードレビューワークフロー。現在の変更に対して包括的なレビューを実施する
disable-model-invocation: true
---

# Code Review Workflow

現在の変更に対して包括的なコードレビューを実施してください。

## 進め方

コード品質レビューとセキュリティレビューを実施する。

### コード品質レビュー
- ステージングされた変更またはブランチ差分を確認
- コード品質・規約準拠・テストカバレッジ・パフォーマンスをレビュー
- **Docstring チェック**: Python は Google Style に準拠しているか確認
- Critical / Warning / Suggestion の3段階で評価を返す

### セキュリティレビュー
- 同じコード変更を OWASP Top 10 観点でセキュリティ監査
- 秘匿情報漏洩・インジェクション・認証の不備を確認
- セキュリティレポートを返す

## 結果の統合

両方の結果を統合して以下を報告:

```markdown
## レビュー結果サマリ

### 総合評価
[Approve / Request Changes]

### 必須対応（Critical）
[コード品質 + セキュリティの Critical 所見を統合]

### 推奨対応（Warning/High）
[Warning および High 所見を統合]

### 任意改善（Suggestion/Low）
[Suggestion および Low 所見を統合]
```

引数 `$ARGUMENTS` が指定されている場合は、そのファイル/ブランチを対象にする。
指定がない場合は現在の `git diff HEAD` を対象にする。
