# 新電腦接續 C2：環境與 C1 圖片還原

這份指引準備接續 C2 所需的環境與資料。C0 的閱讀及查證紀錄、C1 的 JSONL 與人工審核結果都已在 repository 中；換機時沿用既有紀錄。C2 的實作仍待開始。

## 1. 準備工具與專案

新電腦先安裝 Git、conda 與 curl，並開啟能使用這些工具的終端。Linux／WSL 可用 shell；Windows 原生可用 Anaconda Prompt 或已初始化 conda 的 PowerShell。確認工具可執行：

```bash
git --version
conda --version
curl --version
```

首次取得專案：

```bash
git clone https://github.com/ryder1018/Visual-CLM-for-Reliable-Visual-Grounding.git
cd Visual-CLM-for-Reliable-Visual-Grounding
```

若已經 clone，進入該專案根目錄後更新：

```bash
git pull --ff-only
```

**以下指令都在專案根目錄執行**，也就是有 `environment.yml`、`common/` 與 `data/` 的目錄。

## 2. 建立並啟用 conda 環境

這份指引使用 `environment.yml` 中的名稱 `visual-clm`，建立命名環境：

```bash
conda env create -f environment.yml
conda activate visual-clm
python --version
```

預期 Python 為 3.12.x。環境包含 Pillow 與 ijson／YAJL；接續 C2 的準備只需這份環境，GPU／PyTorch／vLLM 的安裝與實測留到後續步驟。

環境已建立後，每次開啟新終端只需：

```bash
conda activate visual-clm
```

根目錄 README 的另一組指令使用 `.conda/visual-clm` 作為 prefix 環境，與此處的命名環境是兩種建立方式。若已依該方式建立環境，啟用既有 prefix 即可，選用一種方式就好。跨平台以 `environment.yml` 建立；`environment-linux-64.explicit.txt` 是原 Linux 電腦的套件快照。

## 3. 還原最終 C1 的 20 張圖片

Git 已提供 `data/debug/final/samples.jsonl`、`manifest.json` 和 `review.json`。下列指令依 manifest 的官方 URL 下載同一批圖片到 `data/debug/final/images/`，並核對每張圖片的 SHA256。**請整行複製執行：**

```bash
python -c "import json; from pathlib import Path; from common.scripts.prepare_c1 import download, sha256; r=Path('data/debug/final'); m=json.loads((r/'manifest.json').read_text(encoding='utf-8')); [download(a['url'],r/a['path']) for a in m['images']]; assert all(sha256(r/a['path'])==a['sha256'] for a in m['images']); print('20 images verified')"
```

成功時輸出：

```text
20 images verified
```

完整檔案已存在時會略過下載，但仍核對 hash；下載中斷的 `.part` 檔案可由再次執行指令續傳。若 hash 核對失敗，先查明失敗的圖片，移走該檔案後再下載，不要修改 manifest 的預期 hash。

這一步沿用已審核的最終樣本與順序，不重新選樣或覆寫人工審核紀錄。

## 4. 驗證並還原審核頁

```bash
python -m unittest discover -s tests -v
python -m common.scripts.review_c1 --samples data/debug/final/samples.jsonl --review data/debug/final/review.json --render
python -c "from common.data import load_samples; s=load_samples('data/debug/final/samples.jsonl'); print(len(s), sum(x['groundable'] for x in s))"
```

預期結果：

- 單元測試全部通過，目前共 13 個。
- 審核檢查輸出 `{"pending": 0, "approved": 40, "rejected": 0}`，exit code=0。
- loader 輸出 `40 20`：40 筆樣本，其中 20 筆正例，另外 20 筆為負例。
- `data/debug/final/review.html` 已產生，可用瀏覽器查看原有審核結果。

`--render` 也會經 loader 檢查圖片可解碼、座標與 target_index 一致性。上述步驟通過後，接續 C2 使用 **`data/debug/final/samples.jsonl`** 與其 `images/` 即可。

## 本次換機還原範圍

| 內容 | 取得方式 |
|---|---|
| 程式、文件、C1 最終 JSONL／manifest／審核結果 | Git clone／pull |
| conda 環境 | 本指引第 2 步 |
| C1 最終 20 張圖片 | 本指引第 3 步 |
| C1 最終審核 HTML | 本指引第 4 步 |

`data/raw/`、舊版審核圖片、`external/CLM/` 與 head checkpoint 留到需要重新產生樣本或實際載入官方權重時取得。C0 Markdown 與 metadata 直接沿用；其取得方式另見 [C0 查證材料說明](../artifacts/c0/README.md)。最終 C1 資料契約見 [data/debug/final/README.md](../data/debug/final/README.md)。
