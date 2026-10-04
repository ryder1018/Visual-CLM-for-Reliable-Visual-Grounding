# Proposal 圖解

圖片統一放在此資料夾；模型圖只畫**推論**，訓練與比較安排看實作順序圖。完整契約與 Steps 以 [proposal.md](../../proposal.md) 為準。

| 圖 | 看什麼？ | 圖片 |
|---|---|---|
| A 推論 | 圖片描述如何接到官方 CLM？ | [route_a.png](route_a.png) |
| B 推論 | adapter 如何接到官方 CLM？ | [route_b.png](route_b.png) |
| 實作順序 | 先做什麼、何時分支、如何比較？ | [roadmap.png](roadmap.png) |
| 資料夾分工 | 共用程式與各路線放在哪裡？ | [project_structure.png](project_structure.png) |

A／B 圖中：綠色＝同圖可重用；紫色＝官方 CLM 路徑中每個 query 需執行的模組；藍色＝組合／評分／拒絕；橘色＝輸出。綠色表示**固定 checkpoint 下**可快取，首次計算仍須計時。這個配色含義只適用 A／B 推論圖。

## A：文字橋接推論

![路線 A：文字橋接推論流程](route_a.png)

人工描述只供起步 debug；正式 benchmark 由 captioner 產生，成本計入首次 query。A-ft／A-random 沿用相同資料流，只改 heads 的初始化／訓練設定。目前下載的圖片下方 query 方框被裁切，後續重新匯出時需包含完整畫布。

## B：視覺 adapter 推論

![路線 B：視覺 adapter 推論流程](route_b.png)

圖為 B-pre 的推論；訓練使用 A 的 head 前 embeddings 作 teacher，另跑 B-random 對照。**每個新 query 仍需 Qwen3**。

## 實作順序與分支

![實作順序與路線分支](roadmap.png)

先量 A 的結果與瓶頸，再決定是否做 B；V0 可在 A 開發期間平行準備；C 不阻塞 A／B。

## 資料夾分工

![專案資料夾分工](project_structure.png)

實線表示包含關係，虛線表示條件式建立，不是資料流。每次實驗的設定、checkpoint、預測與指標保存在各路線的獨立 run 資料夾。
