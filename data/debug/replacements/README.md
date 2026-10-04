# C1：四筆替換樣本補審

原始審核有 36 筆通過、4 筆不通過。這份獨立資料只替換蘋果與 pizza 的四筆退件，不覆蓋原始 JSONL 或審核紀錄。

用瀏覽器開啟同目錄 `review.html`，審核下列兩組：

| 圖片 | 正例 | 負例 |
|---|---|---|
| 6004 | left cruise ship | teddy bear |
| 165 | man on the left | teddy bear |

正例確認描述有唯一目標且紅框正確；負例確認整張圖沒有符合描述的物件。紅框是正例的目標框，藍框是其他 COCO 候選。描述與框取自官方原始標註，自動驗證不等於人工通過。

填審核者並匯出 JSON，保存到 `data/debug/replacements/review.json`。也可以把下載檔放在專案根目錄並告知，讓代理協助歸位。**請保留原始 `data/debug/review.json`。**

啟用專案 conda 環境後檢查：

```bash
python -m common.scripts.review_c1 --samples data/debug/replacements/samples.jsonl --review data/debug/replacements/review.json
```

四筆已由 Ryder Wu 於 2026-10-04 16:46:02（Taipei）全部審核通過，檢查輸出 4 approved、0 pending、0 rejected，exit code=0。根目錄的 `review (1).json` 已原樣歸位，原 pending 產物與匯入檔備份位於 `artifacts/c1_reviews/20261004T164822/`。已與原通過 36 筆合併至 `data/debug/final/`，完成最終 20 正＋20 負驗證，**C1 已完成，C2 未開始**。

只取 gRefCOCO train，與原始 debug 圖片 IDs 不重複；完整同圖 COCO 候選依 annotation ID 排序，座標 xywh 轉 pixel xyxy。query 保留原始 raw。兩筆較清楚的既有 sentence 選擇與完整排除清單存於 `selection.json`；來源與圖片 hash 存於 `manifest.json`。

在專案根目錄重建到新位置：

```bash
python -m common.scripts.prepare_c1 --count 2 --seed 42 --selection data/debug/replacements/selection.json --out data/debug-replacements-rebuild
```

重新產生的審核一律 pending，不自動複製任何人工通過結果。
