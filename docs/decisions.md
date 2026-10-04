# 設計選擇

## 2026-10-04：C1 debug 資料

- 依使用者選擇使用 COCO `train2014`＋gRefCOCO，正負 query 取自同圖的官方標註；只用 train、只保留單目標／無目標。A6 的 RefCOCOg UMD 正式設定不變。
- 候選取完整 COCO instance annotations，按 annotation ID 排序，避免把 target-only 標註當成完整候選。座標由 xywh 轉 pixel xyxy，不靜默修正非法框。
- 固定 seed=42，候選數 2–15，排除含 crowd 的圖片，方便人工審核；此 debug subset 有篩選偏差。
- 文字檢查發現少數原始描述不適合 debugging；用 `data/debug/selection.json` 記錄一張排除圖片與三筆既有 sentence ID 選擇。仍保留原始 raw 文字，圖片語意審核保持 pending。
- 路徑相對 JSONL 所在目錄，可連同圖片搬移到另一台機器；不寫入絕對圖片路徑。
- 人工審核獨立存 `review.json`，綁定 samples SHA256，初始全 pending。官方 no_target 標籤與自動 schema 檢查不等於人工確認。
- 使用者要求環境優先 conda；專案內使用 Miniconda 24.11.1 與獨立環境 `.conda/visual-clm`，交接以 `environment.yml` 為依據。
- 此機器可用記憶體不足以一次解析完整 COCO JSON；使用 ijson 串流讀取並丟棄不用的 segmentation 欄位，加入小型 YAJL library 加速解析。初次產生使用 Python parser，後續環境自動選 YAJL backend。
- `images.cocodataset.org` HTTPS 憑證不符；同官方 S3 bucket 的 path-style HTTPS URL 可正常驗證，下載與 manifest 使用該位址。

## 2026-10-04：C0 官方來源與使用契約

C0 的來源閱讀、版本固定與 checkpoint 靜態查核已完成。先前僅有使用者完成回報，這次補齊可核對證據。**未執行 Qwen encoder、PyTorch head forward、官方 ranking 對齊或 GPU 資源測試**；它們分別留待 C4／C5／A2，不能把本次查核當成模型推論已重現。

### 固定版本與權重識別

| 項目 | 本專案固定值／證據 |
|---|---|
| 官方程式碼 | `Contrastive-LM/CLM`，commit `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`；本地 `external/CLM` |
| CLM head 模型 | `Contrastive-LM/CLM-v0.1-8B`，revision `e939398d4556fcd9400c76fa8c5a513202f42b0a` |
| head 檔案 | `CLM_v0.1-8B.pt`，75,557,149 bytes；本地 `artifacts/c0/model/CLM_v0.1-8B.pt` |
| head SHA256 | `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`；**實際下載計算，與該 revision 的官方 LFS SHA256／大小一致** |
| encoder | `Qwen/Qwen3-8B`；本專案查證固定 revision `b968826d9c46dd6066d109eabc6255188de91218` |
| encoder 查核界線 | 只取得 config／tokenizer config 及五份權重 shards 的官方 LFS metadata，未下載 encoder 權重、未本地驗證其 hash。此 encoder revision 是本專案固定值，**不是已證實的 CLM 原始訓練 encoder revision**；已檢查官方來源未交代後者 |

固定版本來源：[官方程式碼](https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7)、[CLM 模型卡](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/blob/e939398d4556fcd9400c76fa8c5a513202f42b0a/README.md)、[CLM config](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/blob/e939398d4556fcd9400c76fa8c5a513202f42b0a/config.json)、[Qwen config](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json)。完整來源檔案 SHA256、API metadata、encoder shards 的發布 hash 與未重現清單存於 [c0_manifest.json](c0_manifest.json)，本機原始查證資料位於 `artifacts/c0/`。

### Encoder 與 heads 分工、實際架構

- Encoder 是 frozen Qwen3-8B，將 state／action **文字**各自轉成 4096 維 embedding；不是讀圖的 vision encoder。
- `CLM_v0.1-8B.pt` 只包含 state head、action head、`logit_scale` 和設定，不含 Qwen3-8B backbone。兩個 heads 同架構、各自獨立權重，不能互換。
- checkpoint `cfg`：`model=Qwen/Qwen3-8B`、`hidden_size=4096`、`projection_dim=512`、`width=1536`、`depth=3`、`activation=gelu`、`layernorm=true`、`residual=false`。`Dh=4096`、`d=512` 已由 cfg、兩份 head tensor shapes 與 Qwen config 交叉核對。
- 每個 head 是 `Linear(4096,1536) → GELU → Linear(1536,1536) → LayerNorm(1536) → GELU → Linear(1536,512)`，投影後另外 L2 normalize。LayerNorm 只在中間 hidden block，不在第一層或輸出層。
- 兩個 head 的 `inp.weight=[1536,4096]`、`hidden.0.weight=[1536,1536]`、`out.weight=[512,1536]`；包含 biases 和 LayerNorm，共 18,887,680 個 head 參數，另有一個 scale scalar。
- 官方 loader 依 checkpoint cfg 建 head，載入兩份 state dict、切換 `eval()`，投影用 `torch.no_grad()`。不要只照 `make_head()` 的函式預設 depth=2／layernorm=false 建模；這份發布權重實際是 depth=3／layernorm=true。

來源：[heads.py 的架構與 loader](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/heads.py#L41)，實際 checkpoint 靜態結果為 `artifacts/c0/checkpoint_inspection.json`。本次使用只接受三個 pickle globals 的 metadata reader，將 tensor reconstruction 替換成純 metadata，不匯入 torch、不配置權重 tensors、不執行模型。

### Pooling、token recipe 與 normalization

官方模型卡與 config 定義 **last-token pooling**。此契約必須與 Qwen3-8B 搭配；不能因 CLIP features 維度相同就接到這份 heads。實際最後有效 token 的選取與 padding 處理仍須用選定 vLLM runtime 驗證；官方 launcher 只指定 `--runner pooling`，沒有明設 pooler type，且專案依賴只有 `vllm>=0.6`，不能由此證明任意版本都採同一 pooling 行為。

| 路徑 | 已讀到的官方 token／長度規則 |
|---|---|
| Serving／Engine.rank | 原始文字送 `/v1/embeddings`，`encoding_format=base64`，預設 `truncate_prompt_tokens=2048`；launcher 預設 `--max-model-len 2048`。這個 client 不自行套 chat template，server 如何 tokenize 仍須實測 |
| `train/embed_utils.py`，`clm` 從文字重建 embeddings | `add_special_tokens=False`；字串 state 留尾端，action 留開頭；token cap=`max_len-1`。聊天 state 使用 `apply_chat_template(..., add_generation_prompt=False)` 後留尾端；空文字改為一個空格。可見程式未額外 append EOS |
| `choice` fine-tuning | state／candidate 都依 schema 轉文字，兩者使用 `text_ids(..., keep="tail")`；與上列 action 留開頭不同 |
| CLI 預設 budget | `clm` 為 8192、`choice` 為 2048；與 serving 2048 的 budget 不能混為同一 recipe |

官方 embedder 把 encoder 回傳值轉 float32，再做 `x / (L2_norm(x) + 1e-12)`；兩份 head 的輸出再各自用 `torch.nn.functional.normalize(..., dim=-1)`。encoder 正規化與投影後正規化是兩個步驟，不能省掉前者或只對其中一份 head 正規化。這是已核對的程式契約，並非已測得本機 embedding／projection 的數值。

來源：[embedder.py](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/embedder.py)、[train/embed_utils.py](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/train/embed_utils.py)、[train/finetune.py](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/train/finetune.py)、[encoder launcher](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/serve_qwen3_8b.sh)。原始發布 head 完整訓練 tokenizer／截斷細節未全寫入 checkpoint cfg；可見 fine-tuning recipe 不能直接宣稱等同原始預訓練 recipe。A2 必須以固定 runtime／token IDs 做官方對齊檢查。

### Ranking 與實際 logit_scale

`Engine.rank` 先將 context／question 轉文字並各自 `.strip()` 去除前後空白，再將 `context + "\n\n" + question` 作為 state（兩者都有時）。每個非空 candidate 文字原樣作為 action，不加 query 前綴、不去除 candidate 的前後空白；空字串 candidate 則改用其數字 option key，例如 `"0"`。候選順序由 caller 提供；經 softmax 後依 probability 降冪排名。發布 head 本身不產生 bounding boxes 或 NONE，也不含本專案的拒絕門檻；這些需要後續實作。

令 `S`、`A_i` 為兩份 head 投影且 L2 normalized 的向量，官方算：

```text
cos_i = dot(S, A_i)
scale = min(exp(logit_scale), 100.0)
logit_i = scale * cos_i / T_api
probabilities = softmax(logits)
```

本 checkpoint 的 float32 `logit_scale=4.613248825073242`，`exp(logit_scale)≈100.81113593422504`；**官方 clamp 後實際 scale=100.0**。API temperature 預設 `T_api=1`；本專案沿用此值，不再另除訓練 temperature。若啟用 API 的額外溫度必須單獨記錄，不能把 `logit_scale` 當成已 exponentiate 的倍率，也不能重新用 `log(1/0.07)` 覆蓋發布權重值。拒絕 threshold 的 max cosine 與 scaled logits 也必須區分。

來源：[heads.py 的 scale 與 projection](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/heads.py#L81)、[engine.py 的 ranking](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/engine.py#L103)、[schema.py 的文字與 softmax](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/schema.py)。

### 官方訓練 recipe 與重現界線

官方 README 描述三階段：預訓練約 60M Nemotron DQA 正對、mid-training 約 30M synthetic hard negatives、post-training 約 1M agent trajectories；post-training 混合 40% DQA replay／60% agentic。官方目標為雙向 InfoNCE；這些是**官方發布敘述，未在此專案重跑或逐筆核對訓練資料**。本 proposal 的同圖 candidate CE 是下游 adaptation，不能稱為重現完整官方預訓練。

來源：[固定版本 README 的 recipe／algorithm](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/README.md#data-recipe)、[官方 fine-tuning objective](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/train/finetune.py#L174)。`docs/FINETUNING.md` 是自行改進 fine-tuning 的實驗流程，已閱讀但本次沒有啟動其訓練迴圈。

重跑本次 checkpoint 查核（專案根目錄、既有 conda 環境）：

```bash
python artifacts/c0/inspect_checkpoint.py
python artifacts/c0/verify_sources.py
```

前者檢查實際 cfg／shapes／scalar 並比對官方 LFS hash；後者核對固定 source commit、來源內容 hash 與紀錄的一致性。取得方式、固定 URLs 與檔案 hash 詳見 `docs/c0_manifest.json`／`artifacts/c0/README.md`。C0 完成不代表 C4 資源、C5 官方 loader 或 A2 ranking 對齊完成；C2 依使用者要求仍未開始。

## 2026-10-04：匯入人工審核與 C1 替換

- 使用者匯出的審核原樣移到 `data/debug/review.json`；驗證 samples SHA256、樣本覆蓋、審核者與時區時間。36 筆通過、4 筆不通過，保留原始資料及退件原因。
- 已通過的 18 正＋18 負原樣保存在 `data/debug/approved.jsonl`，不轉移審核結果到改過的樣本。匯入前產物與匯入檔備份位於 `artifacts/c1_reviews/20261004T163614/`。
- 蘋果與 pizza 的兩組退件以兩張新圖片替換；`data/debug/replacements/` 獨立保存 2 正＋2 負、圖片、manifest 與全 pending 的人工審核，避免覆蓋原本審核。query 不改寫，sentence 選擇與圖片排除清單記在該目錄的 `selection.json`。
- 可用原準備腳本重建替換資料：`python -m common.scripts.prepare_c1 --count 2 --seed 42 --selection data/debug/replacements/selection.json --out data/debug-replacements-rebuild`。新輸出仍須人工審核。
- 審核頁標題與匯出保存路徑由實際資料推導，補審頁只顯示 2 正＋2 負，保存到其獨立的 `review.json`。
- 在第一次匯入時，四筆補審與最終合併尚未完成，當時 C1 保持進行中。使用者明確要求先不開始 C2；即使 C1 後續完成也不自動開始 C2。

## 2026-10-04：C1 完成

- 補審四筆全部 approved，驗證 SHA256、覆蓋與審核者／時間後，把根目錄 `review (1).json` 原樣移至 `data/debug/replacements/review.json`；匯入前產物及匯入檔備份於 `artifacts/c1_reviews/20261004T164822/`。
- 最終資料獨立存於 `data/debug/final/`，不覆寫原始退件紀錄。36 筆原通過樣本再接 4 筆補審通過樣本，JSONL 每筆不改動；圖片複製至最終目錄，保留相對路徑。
- 最終 `review.json` 標示 `derived_from_human_reviews=true`，保留每筆原審核者、時間、備註與來源；頂層時間表示最後一份來源審核時間。這是驗證過的兩份真實審核彙整，不是新增人工審核。
- 最終 manifest 的 `composition` 記兩份來源資料／審核的 SHA256、納入 IDs 與順序。直接重新核對原始 gRefCOCO／完整 COCO 標註、來源與圖片 hash、座標／targets、完整候選及人工審核來源，結果存 `data/debug/final/validation.json`。
- 最終 20 正＋20 負、20 張圖片、40 approved；13 個測試與審核檢查通過。proposal C1 勾選完成；C2 保持未開始。
