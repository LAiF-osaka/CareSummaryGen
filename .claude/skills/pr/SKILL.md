---
name: pr
description: GitHub Pull Request 管理（create, list, show）
argument-hint: [create|list|show] [options]
disable-model-invocation: true
---

# PR 管理スキル

GitHub の Pull Request を管理する。`$ARGUMENTS` の最初のトークンでサブコマンドを判別する。

## サブコマンド一覧

| サブコマンド | 用途 | 例 |
|---|---|---|
| `create [options]` | PR 新規作成 | `/pr create --base develop` |
| `list [status]` | PR 一覧表示 | `/pr list` |
| `show <id>` | PR 詳細表示 | `/pr show 10` |

サブコマンドが未指定の場合は `list` として扱う。

---

## サブコマンド: create

### 手順

1. 現在のブランチと差分を確認する:
   ```bash
   CURRENT_BRANCH=$(git branch --show-current)
   git log develop..$CURRENT_BRANCH --oneline 2>/dev/null
   ```
2. PR を作成する:
   ```bash
   gh pr create \
     --base "<target|develop>" \
     --title "<title>" \
     --body "<description>"
   ```
3. 作成結果（PR URL）を報告する

### オプション
- `--base <branch>`: ターゲットブランチ（デフォルト: develop）
- `--title "<title>"`: PR タイトル（省略時はブランチ名から生成）
- `--draft`: ドラフト PR として作成

---

## サブコマンド: list

### 手順

```bash
gh pr list --state open
```

---

## サブコマンド: show

### 手順

```bash
gh pr view <id>
```

以下を表示:
- タイトル、状態、作成者、レビュアー
- ソース/ターゲットブランチ
- マージ状態

---

## 注意事項

- **PR URL を必ず提示**: 作成・言及時に URL を添える
- **リポジトリ名**: `git remote get-url origin` からリポジトリ名を自動取得する
