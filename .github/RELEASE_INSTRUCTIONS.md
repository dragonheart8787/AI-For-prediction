# 建立 GitHub Release

1. 確認 [`VERSION`](../VERSION) 與 [`CHANGELOG.md`](../CHANGELOG.md) 首條一致。
2. 已打 tag 並推送（維護者本機）：
   ```bash
   git tag -a "v$(cat VERSION)" -m "Predict-AI v$(cat VERSION)"
   git push origin "v$(cat VERSION)"
   ```
3. 到 GitHub：**Releases → Draft a new release** → 選取上述 tag → 標題例如 `v0.9.3` → 描述可貼上 CHANGELOG 對應段落 → **Publish release**。

或使用 [GitHub CLI](https://cli.github.com/)：

```bash
gh release create "v$(cat VERSION)" --title "v$(cat VERSION)" --notes-file CHANGELOG.md
```

（若 notes 太長，可手動節錄 CHANGELOG 中該版區塊。）
