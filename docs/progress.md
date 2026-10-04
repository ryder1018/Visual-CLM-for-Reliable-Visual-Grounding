# 實作進度

更新：2026-10-04（Asia/Taipei）。**C0、C1 已完成，C2 未開始。**

| Step | 狀態 | 證據／下一步 |
|---|---|---|
| C0 | 完成（來源／checkpoint 靜態查核） | `docs/decisions.md`、`docs/c0_manifest.json`、`artifacts/c0/verification.json`；模型推論與官方 ranking 未重現 |
| C1a：讀取及 schema 檢查 | 完成 | `common/data.py`；`tests/test_data.py`，6 個測試通過 |
| C1b：20 正＋20 負與圖片 | 完成 | 最終資料 `data/debug/final/samples.jsonl`；20 張圖片、來源 manifest 與 validation |
| C1c：人工審核 | 完成 | 原始 36 筆通過＋補審 4 筆通過；最終 40 approved、0 pending、0 rejected |
| C1 整體 | 完成 | 最終座標、groundable／gt_box／target_index、完整候選、來源與人工審核皆通過；proposal 已勾選 C1 |
| C2–C5／A1–A3 | 未開始 | 依使用者要求，完成 C1 後也不自動開始 C2；未建立 `common/regions.py` |

後續讀取資料請使用 **`data/debug/final/samples.jsonl`**。共有 40 筆（20 正、20 負）、20 張 COCO train2014 圖片，gRefCOCO 描述全部來自 train，候選數 2–13。JSONL SHA256：`7047f7745d419a1fabdc39a4019580d21f00a7e2389b7351b28b87227b6c4de2`。

最終資料的每筆 JSONL 與來源已通過樣本完全一致，候選依完整 COCO annotation ID 排序，未改寫 query 或修改框。直接重新讀取原始 gRefCOCO 與完整 COCO annotations，核對原始描述／target IDs、同圖完整候選、xywh→xyxy、來源／圖片 SHA256、圖片尺寸及兩份人工審核的來源；結果存於 `data/debug/final/validation.json`。最終審核頁有 20 組圖片、40 個控制項，JavaScript 通過 Node syntax check。

啟用 conda 環境後，在專案根目錄執行：

```bash
python -m unittest discover -s tests -v
python -m common.scripts.review_c1 --samples data/debug/final/samples.jsonl --review data/debug/final/review.json
python -c "from common.data import load_samples; s=load_samples('data/debug/final/samples.jsonl'); print(len(s), sum(x['groundable'] for x in s))"
```

已通過 13 個測試（loader 6、資料轉換／文字選擇 3、審核紀錄 4）。最終審核檢查輸出 `pending=0, approved=40, rejected=0`、exit code=0；loader 載入 40 筆、20 筆正例。conda 指令見根目錄 `README.md`，最終資料說明見 `data/debug/final/README.md`。

## C0：官方契約與證據

官方 repo 固定 commit `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`；CLM head 模型固定 revision `e939398d4556fcd9400c76fa8c5a513202f42b0a`。實際下載 `CLM_v0.1-8B.pt`（75,557,149 bytes），SHA256 `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`，與官方 LFS metadata 一致。

checkpoint／官方程式／encoder config 交叉核對：`Dh=4096`、`d=512`、head width=1536／depth=3／GELU／LayerNorm=true／residual=false。官方契約為 Qwen3-8B last-token pooling，encoder 及投影後各自 L2 normalize。checkpoint `logit_scale=4.613248825073242`，exp 約 100.811，官方 clamp 後 scale=100。完整 token／ranking／train recipe 差異與來源見 `docs/decisions.md`。

已核對 22 份本地來源／metadata／inspection script 的 SHA256、checkpoint 的 cfg／shapes／scalar、來源 repo revision 與 clean worktree；metadata parser 拒絕允許清單外 globals。13 個既有測試通過。重跑指令：

```bash
python artifacts/c0/inspect_checkpoint.py
python artifacts/c0/verify_sources.py
```

**未重現：** Qwen encoder 推論、vLLM runtime pooling／tokenization、PyTorch 官方 loader／forward、官方 ranking 數值對齊、GPU probe 或訓練。encoder 權重只記錄官方 hash metadata，未本地下載核對；本專案 encoder revision 不代表已知的原始 CLM 訓練版本。這些界線不影響 C0 的來源查證完成，但不能當成 C4／C5／A2 完成。

## 人工審核與保留紀錄

- 原始審核：Ryder Wu，2026-10-04 16:31:15（Taipei），36 通過、4 不通過。原資料／審核保留在 `data/debug/samples.jsonl`／`data/debug/review.json`；匯入備份 `artifacts/c1_reviews/20261004T163614/`。
- 未通過項目：image 204436 的蘋果正負例、image 278467 的 pizza 正負例。框與描述不符或文字有歧義，退件理由保留。36 筆通過子集仍保存在 `data/debug/approved.jsonl`。
- 替換審核：Ryder Wu，2026-10-04 16:46:02（Taipei），4 筆全通過。根目錄 `review (1).json` 已原樣移至 `data/debug/replacements/review.json`，匯入前產物與匯入原檔備份在 `artifacts/c1_reviews/20261004T164822/`。
- 最終合併：原通過 36 筆，再接替換通過 4 筆。`data/debug/final/review.json` 明確標示由兩份真實審核彙整，每筆保留原審核者、時間、備註與來源；頂層時間是最後一份來源審核時間，並非新一次人工審核。`manifest.json` 記錄來源 JSONL／審核 SHA256 與納入 IDs。

原始資料仍有 4 rejected，所以以預設路徑檢查原始審核會回 exit code=1；這是保留歷史退件的預期結果。C1 完成依據是 `data/debug/final/`。第一版文字較不清楚的資料仍保留於 `artifacts/c1_initial`。
