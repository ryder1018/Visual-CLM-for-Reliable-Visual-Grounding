# C0 官方來源查證材料

固定 GitHub `Contrastive-LM/CLM` commit `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` 在 `external/CLM`；HF head 模型 revision `e939398d4556fcd9400c76fa8c5a513202f42b0a`；encoder 設定 revision `b968826d9c46dd6066d109eabc6255188de91218` 是本專案 audit pin，不是已知的官方原訓練 revision。

`model/CLM_v0.1-8B.pt` 是兩份投影 heads，75,557,149 bytes，不含 Qwen backbone。SHA256 `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5` 已實際計算，符合官方 pinned LFS metadata。

- `api/`：官方 GitHub／HF 的原始 metadata snapshots。
- `model/`：固定版 head 模型卡、config 與 checkpoint。
- `encoder/`：固定版 Qwen config／tokenizer config，未下載 encoder 權重。
- `checkpoint_inspection.json`：允許清單 metadata parser 得到的 cfg、tensor shapes、scale。
- `manifest.json`：與 `docs/c0_manifest.json` 相同，包含固定 URLs、SHA256 與未重現項目。
- `verification.json`：本地來源／checkpoint 的實際檢查結果。

啟用根目錄 README 所述 conda 環境後：

```bash
python artifacts/c0/inspect_checkpoint.py
python artifacts/c0/verify_sources.py
```

只依賴 Python 標準庫與 git；不匯入 torch、不執行 encoder 或 heads。`inspect_checkpoint.py` 只允許必要的 OrderedDict、FloatStorage dtype marker 與代換的 tensor metadata reconstruction，拒絕其他 pickle globals；只適用本次 pinned checkpoint 格式。

若交接時缺少 `external/CLM`，依 manifest clone 官方 repo 並 checkout 上述 commit。其他来源檔案使用 manifest 的固定 `sources[].url` 下載到對應 `sources[].path`，檔案下載後須重新計算 SHA256。API snapshots 是查證當時回傳值，可一併交接；若重查 API 必須使用固定 revision，不能用 main 重新取得不同版本後沿用旧 hash。

本目錄的小型查核腳本、metadata、config、結果紀錄與說明納入 Git；head checkpoint、pickle disassembly 與 `external/` 不納入 Git。新 checkout 執行查核前，須依 manifest 取得固定版本的官方程式碼與 head checkpoint，並核對下載 hash。永久交接紀錄另存於 `docs/decisions.md` 與 `docs/c0_manifest.json`。本次只完成 C0 靜態來源查核；Qwen inference、vLLM 實際 pooling、PyTorch loader／forward、官方 ranking 數值對齊與 GPU probe 均未重現。
