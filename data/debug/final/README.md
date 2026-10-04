# C1：最終人工審核資料

此目錄包含 20 張 COCO train2014 圖片、20 正例＋20 負例，40 筆全部經 Ryder Wu 人工審核通過。後續工作請使用本目錄的 `samples.jsonl`。來源為 gRefCOCO train，保留原始 raw query 與完整同圖 COCO 候選；只供 debug，正式 A6 設定仍為 RefCOCOg UMD。

資料由原審核通過的 36 筆，再接補審通過的 4 筆組成；每筆 JSONL 完整保留原值，圖片複製到 `images/`，路徑仍相對 JSONL 所在目錄。原始資料與退件原因在上一層，補審資料在 `../replacements/`，均保留。

`review.json` 是兩份真實人工審核的彙整，並非新一次人工審核。每筆保留原審核者、時間、備註與來源路徑；頂層 `reviewed_at` 表示最後一份來源審核的時間。`review_sources` 與 `manifest.json` 的 `composition` 記錄原資料／審核檔的 SHA256 和納入 IDs；原始檔案不可覆寫，以保留驗證依據。

- `samples.jsonl`：40 筆最終樣本；SHA256 `7047f7745d419a1fabdc39a4019580d21f00a7e2389b7351b28b87227b6c4de2`。
- `images/`：20 張圖片，hash 與尺寸記於 `manifest.json`。
- `review.json`：40 approved、0 pending、0 rejected；兩份來源人工審核的彙整。
- `manifest.json`：來源 revision／hash、圖片、樣本 IDs 與合併來源。
- `validation.json`：直接核對原始 COCO／gRefCOCO 與兩份審核的檢查結果。
- `review.html`：顯示最終樣本與已審核結果，可用於查看。

在專案根目錄啟用 conda 環境後驗證：

```bash
python -m unittest discover -s tests -v
python -m common.scripts.review_c1 --samples data/debug/final/samples.jsonl --review data/debug/final/review.json
python -c "from common.data import load_samples; s=load_samples('data/debug/final/samples.jsonl'); print(len(s), sum(x['groundable'] for x in s))"
```

預期審核檢查輸出 `pending=0, approved=40, rejected=0`、exit code=0；loader 載入 40 筆、20 筆正例。

合併可依 `manifest.json` 的 `composition` 重建：依清單順序读取各來源 JSONL，只保留該來源審核 approved 且列在 `included_sample_ids` 的原樣資料，再複製對應圖片。生成 JSONL 時每筆使用 `json.dumps(sample, ensure_ascii=False)` 加上換行；不變更座標、路徑、query 或候選順序。兩次審核的來源紀錄必須保留。

C2 依使用者要求維持未開始。
