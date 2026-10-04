# Visual_CLM

Ground-or-Abstain：以官方 CLM 權重延伸視覺定位。研究契約與逐步實作見 [proposal.md](proposal.md)，目前進度見 [docs/progress.md](docs/progress.md)。

C0 官方來源與 checkpoint 靜態查核已完成，版本／hash／Dh／d／pooling／scale 見 [docs/decisions.md](docs/decisions.md) 與 [來源 manifest](docs/c0_manifest.json)；encoder 推論與官方 ranking 尚未重現。C1 資料準備與人工審核也已完成：20 正例＋20 負例，使用 COCO `train2014` 圖片＋gRefCOCO queries。後續請讀取 [data/debug/final/samples.jsonl](data/debug/final/samples.jsonl)，驗證指令與審核來源見 [最終資料說明](data/debug/final/README.md)。C2 依使用者要求維持未開始。

## 新電腦接續 C2

請依 [新電腦接續 C2 指引](docs/setup_c2.md) 建立 conda 環境、下載並核對 C1 最終的 20 張圖片，以及還原審核頁。指引沿用已完成的 C0／C1 紀錄；C2 實作仍待開始。

## conda 環境

```bash
conda env create --prefix "$PWD/.conda/visual-clm" --file environment.yml
conda activate "$PWD/.conda/visual-clm"
python -m unittest discover -s tests -v
```

這台機器的 conda 安裝在專案內 `.tools/miniconda3`。首次開啟終端時可使用：

```bash
source .tools/miniconda3/etc/profile.d/conda.sh
conda activate "$PWD/.conda/visual-clm"
```

只需 Python、Pillow 與串流 JSON parser ijson／YAJL，無 GPU／模型依賴。YAJL 加速大型標註的讀取。這台 Linux 機器的實際套件版本另存於 `environment-linux-64.explicit.txt`；跨平台使用 `environment.yml`。

大型原始資料、圖片、模型權重、第三方程式碼與環境不納入 Git；程式、文件、JSONL、人工審核紀錄、subset IDs、來源 manifest 與 C0 小型查核材料納入版本管理。新 checkout 可直接執行單元測試；載入實際 debug 資料或重跑 C0 查核前，請依各 README／manifest 取得圖片、固定版官方程式碼與權重。
