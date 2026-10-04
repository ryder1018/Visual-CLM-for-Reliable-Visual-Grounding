# C1：COCO＋gRefCOCO debug subset

**C1 已完成。後續請使用 [final/samples.jsonl](final/samples.jsonl)，其 20 正例＋20 負例均已人工審核通過。C2 未開始。**

本目錄根層保留第一次審核的原始資料：36 筆通過、4 筆不通過，通過子集保存於 `approved.jsonl`。`replacements/` 的 4 筆替換已全部通過；最終資料在 `final/`，由這兩份審核的通過樣本原樣合併。最終驗證與來源見 [final/README.md](final/README.md)。本資料只供 debug，不是正式 benchmark，來源為 gRefCOCO **train split** 與 COCO `train2014`。

來源：[gRefCOCO 官方專案](https://github.com/henghuiding/gRefCOCO)、[標註下載](https://huggingface.co/datasets/FudanCVL/gRefCOCO/tree/main)、[COCO 官方下載](https://cocodataset.org/#download)。gRefCOCO revision 固定為 `81eede59b3ac070049f597d023c0ff08d1fb80e9`。使用條件依原始資料發布頁；gRefCOCO 為非商業研究用途。

## 原始人工審核紀錄（已完成）

用瀏覽器開啟同目錄的 `review.html`。每張圖片有原圖、編號候選框，以及正負兩個 query。

1. 正例：query 有唯一目標，紅框正確；留意位置、屬性與關係描述。
2. 負例：檢查整張原圖，確認沒有任何符合 query 的物件。
3. 候選框：位置與範圍應合理；圖上數字是從 0 起算的 index。
4. 對每筆選「通過」或「不通過／有歧義」，不通過時填原因。
5. 填審核者，點「匯出審核 JSON」，將下載檔案保存到 `data/debug/review.json`。關閉頁面前先匯出；下次可匯入繼續。

若 VS Code HTML 預覽無法執行匯出，請使用外部瀏覽器。也可在專案根目錄啟動本機服務：

```bash
python -m http.server 8000 --bind 127.0.0.1
# 瀏覽器開啟 http://127.0.0.1:8000/data/debug/review.html
```

檢查最終審核紀錄：

```bash
python -m common.scripts.review_c1 --samples data/debug/final/samples.jsonl --review data/debug/final/review.json
```

最終 40 筆全部 approved，exit code 為 0。省略上述路徑參數會檢查本目錄原審核，36 approved、4 rejected、exit code=1；原始退件刻意保留。人工紀錄綁定其來源 `samples.jsonl` 的 SHA256，不能移用到另一份資料。原始與補審 manifest 的 `human_review` 是生成時快照，目前狀態以各自 `review.json` 的檢查結果為準。

## 格式與候選

`samples.jsonl` 欄位遵循 proposal。`image_path` 相對 **JSONL 所在目錄**，例如 `images/COCO_train2014_...jpg`。loader 回傳路徑字串，不改成機器專屬的絕對路徑。

- `boxes`／`gt_box`：pixel xyxy；原始 COCO xywh 轉成 `[x, y, x+w, y+h]`，不做裁切或 normalized conversion。
- 候選：完整 `instances_train2014.json` 中同圖的所有 annotation boxes，依 annotation ID 排序；保留干擾物件，不只取 gRefCOCO 正解。
- 正例：只有一個有效 target annotation，`target_index` 是其候選索引。
- 負例：`groundable=false`、`gt_box=null`、`target_index=null`；原始 `ann_id=-1` 不作候選索引。
- `source` 保留 gRefCOCO ref／sentence IDs、revision 與候選來源；query 保留原始英文 `raw`，不改寫或翻譯。

原始取樣固定 seed=42；選 train 中有正負 query 的圖片，候選數 2–15，排除 crowd 及非法／越界框。`selection.json` 記錄初次文字檢查後的排除圖片與原始 sentence ID 選擇，未改寫文字。替換取樣使用 `replacements/selection.json`；最終採用兩份人工審核 approved 的樣本。

這些方便人工 debugging 的篩選會造成取樣偏差，不可用此 subset 宣稱泛化成績。正式 A6 仍使用 **RefCOCOg UMD**；尚未跨資料集檢查 image IDs。

## 重跑原始資料

啟用根目錄 README 所述的 conda 環境後：

```bash
python -m common.scripts.prepare_c1 --count 20 --seed 42 --out data/debug-rebuild
```

腳本自動取得缺少的原始標註並只下載所選圖片。已有完整來源檔案時重用；部分檔案以 `.part` 續傳。輸出目錄已有 JSONL／manifest／review 時拒絕覆寫，避免丟失人工審核紀錄。

新目錄與原資料的 `samples.jsonl`、圖片 IDs 應一致；新審核狀態仍為 pending。`manifest.json` 保存來源 URL／hash、subset IDs（在 JSONL／images 清單）、圖片 hash、尺寸、取樣條件與 dataset hash。

更新審核頁而不改資料：

```bash
python -m common.scripts.review_c1 --render
```

## 程式檢查

```bash
python -m unittest discover -s tests -v
python -c "from common.data import load_samples; s=load_samples('data/debug/final/samples.jsonl'); print(len(s), sum(x['groundable'] for x in s))"
```

預期載入 40 筆、20 筆正例。loader 檢查圖片可解碼、座標、IDs、標籤與索引；COCO GT 正例要求候選框與 GT 完全一致，未來 detector proposals 則須 IoU≥0.5。無候選與 proposal_miss 保留 groundable=true，須提供對應 reason；這些案例由測試覆蓋，不混入本次 GT debug 樣本。
