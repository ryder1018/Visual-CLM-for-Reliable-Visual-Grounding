# Ground-or-Abstain: Visual CLM for Reliable Visual Grounding

## 1. Project Overview

這份文件為最初構想，詳細的計畫請參閱proposal.md

本研究希望將 **Contrastive Language Model (CLM)** 的 state-action contrastive decision framework 延伸到 Visual Grounding。

核心問題為：

> 給定一張圖片與自然語言描述，模型不只需要找出描述所對應的區域；當圖片中不存在符合描述的物體時，也應該能夠選擇 **Abstain / None**，而不是強迫產生一個錯誤位置。

例如：

```text
Image:
桌上有紅色杯子、白色杯子、筆電

Query:
"The blue cup next to the laptop."
```

傳統 grounding system 可能仍選擇其中一個杯子。

本研究希望模型輸出：

```text
Region A        0.05
Region B        0.03
Region C        0.01
NONE            0.91
```

因此將問題定義為：

\[
\text{Ground}(I,Q)
\rightarrow
\{R_1,R_2,\dots,R_N,\text{NONE}\}
\]

其中：

- \(I\)：image
- \(Q\)：referring expression / instruction
- \(R_i\)：candidate visual region
- `NONE`：目前影像不存在可靠對應物件

---

# 2. Motivation

目前許多 Visual Grounding / MLLM 方法使用 autoregressive generation 產生 bounding box、座標或相關描述。

這類方法可能有兩個值得研究的問題：

1. **Grounding hallucination**：即使 query 對應物體不存在，模型仍可能產生位置。
2. **Decision efficiency**：若最終問題只是從有限候選區域中選擇目標，autoregressive generation 未必是最直接的 formulation。

CVPR 2026 的 **GroundingME** 將 visual grounding 的困難分為 Discriminative、Spatial、Limited 與 Rejection 四個面向，其中 Rejection 專門測試「不存在可 grounding target 時，模型是否能拒絕」。該 benchmark 顯示目前 MLLM 在這類案例仍存在明顯問題。

另外，CVPR 2026 的 **VGent** 也指出 autoregressive MLLM grounding 存在 inference latency 與 hallucination 等問題，並改採 proposal selection 的 modular architecture。這表示「將 grounding 視為 candidate selection 而非 coordinate generation」本身是一個合理方向。

本研究因此考慮：

> 是否可以將 Visual Grounding formulation 成類似 CLM 的 **contrastive state-action decision problem**？

---

# 3. Connection to CLM

CLM 的核心不是 Qwen3-8B 本身，而是：

\[
State \leftrightarrow Action
\]

將 state 和 candidate actions 分別投影至同一 embedding space，再使用 contrastive objective 學習：

\[
sim(S,A_{positive})
>
sim(S,A_{negative})
\]

官方 CLM 將 `state` 與 `action` 分開編碼，核心 primitive 即是「給定 state，對 candidate action 打分」。

原始 CLM：

```text
Text State
    ↓
State Encoder
    ↓
State Projection
    ↓
S

Text Action
    ↓
Action Encoder
    ↓
Action Projection
    ↓
A
```

本研究希望將其延伸為：

```text
Image + Query
      ↓
Visual / Multimodal State Encoder
      ↓
State Projection
      ↓
S

Candidate Region
      ↓
Visual Action Encoder
      ↓
Action Projection
      ↓
Ai
```

最後計算：

\[
score_i=S^TA_i
\]

再從：

\[
\{R_1,R_2,\dots,R_N,\text{NONE}\}
\]

中做決策。

因此本研究目前暫稱：

> **Visual CLM / Multimodal Contrastive Decision Model**

這個名稱與最終 architecture 仍可能調整。

值得注意的是，CLM 官方 roadmap 本身已列出：

> Vision and multimodal support: images, video and other modalities for robotics and computer-use tasks.

因此這項研究可以視為 CLM framework 的 multimodal extension，而非單純將 CLM 套用到任意 CV task。

---

# 4. Research Questions

初期主要研究四個問題。

### RQ1 — Visual State Representation

如何將：

\[
(Image,\ Query)
\]

表示為適合 contrastive decision 的 state representation？

例如 query：

```text
"the red cup to the left of the laptop"
```

模型需要理解的不只是 `red cup`，還需要：

- object appearance
- global scene
- spatial relationship
- language description

因此不能只使用單純 text embedding。

---

### RQ2 — Candidate Region Representation

如何將 candidate region 表示成 Action？

簡單方法可能是：

```text
Region crop
→ Vision Encoder
→ Action Head
```

但這可能無法處理：

```text
"the cup to the left of the laptop"
```

因為 crop 本身不知道自己與 laptop 的相對位置。

因此目前假設 region representation 可能需要包含：

\[
A_i=f(
\text{ROI feature},
\text{position},
\text{global context}
)
\]

實際最佳設計仍需透過實驗決定。

---

### RQ3 — Ground-or-Abstain

這可能是本研究最主要的問題。

一般 contrastive learning 假設：

\[
State_i \leftrightarrow PositiveAction_i
\]

但 ungroundable query 沒有 positive region。

例如：

```text
Image:
red cup, laptop

Query:
blue cup
```

不存在：

\[
R_{positive}
\]

因此需要設計適合 **no-positive sample** 的訓練方式。

目前可能方向包括：

1. Explicit `NONE` prototype
2. Similarity / energy threshold
3. 額外的 groundability head
4. Unknown-aware contrastive objective

初期不預設其中任何一個一定最佳，預計透過 ablation 比較。

---

### RQ4 — Efficiency

研究 discriminative candidate scoring 是否能在 grounding task 中降低：

- decoding latency
- unnecessary autoregressive generation

同時維持或改善：

- grounding accuracy
- rejection accuracy

CLM 原本的一大效率來源是 state/action disaggregation 與 action embedding caching；但 visual regions 通常會隨 image 改變，因此這項優勢在本 task 上可能較弱。

因此本研究不預設 Visual CLM 一定比所有 grounding architecture 快，而是實際測量：

\[
Accuracy \leftrightarrow Latency
\]

trade-off。

---

# 5. Tentative Architecture

> **注意：以下僅為目前 initial design。主要目的為建立第一個可驗證 prototype，實際 architecture 很可能隨實驗結果調整。**

```text
                         Image
                           │
                    Vision Encoder
                           │
                ┌──────────┴──────────┐
                │                     │
          Global Features        Region Features
                │                     │
Query ──► Text Encoder                 │
                │                     │
                ▼                     ▼
       Multimodal Fusion       Region Representation
                │                     │
          State Head              Action Head
                │                     │
                S              A1 A2 ... AN
                │                     │
                └──────────┬──────────┘
                           │
                   cosine similarity
                           │
                ┌──────────┴───────────┐
                │                      │
            Region Scores           NONE
                │
                ▼
            Prediction
```

初期可以優先採用 frozen backbone，只訓練：

```text
State fusion module
State projection head
Action projection head
Abstention-related parameters
```

這會比較接近原始 CLM：

> frozen backbone + lightweight trainable projection heads

的設計哲學。原始 CLM 目前使用 Qwen3-8B encoder 搭配可訓練 head。

---

# 6. Candidate Region Generation

Visual CLM 本身暫不負責產生 bounding boxes。

初期 pipeline：

```text
Image
 ↓
Region Proposals
 ↓
R1, R2, ... RN
 ↓
Visual CLM
 ↓
Target Region / NONE
```

Region proposals 可考慮：

- Grounding DINO
- DETR-family detector
- SAM / SAM2 搭配 proposal source
- dataset ground-truth regions

為了避免 proposal quality 干擾最初研究，建議先進行：

### Experiment A

使用 Ground Truth Candidate Boxes

目的：

> 單純驗證 contrastive decision model 是否具有 region discrimination / abstention 能力。

### Experiment B

使用 predicted proposals

目的：

> 評估完整 real-world pipeline。

需要注意：

\[
Recall_{system}
\leq
Recall_{region-proposal}
\]

如果 target 根本沒有被 proposal system 找到，後續 Visual CLM 無法選到正確答案。

---

# 7. Training Objective

最基本的 positive grounding sample：

```text
State:
Image + Query

Positive:
Correct Region

Negatives:
Other Regions
```

可以沿用 CLM-style contrastive objective：

\[
\mathcal L_{contrastive}
=
-\log
\frac{
\exp(sim(S,A^+)/\tau)
}{
\sum_j\exp(sim(S,A_j)/\tau)
}
\]

並加入 hard negatives。

例如：

```text
Query:
"the red cup beside the laptop"

Positive:
correct red cup

Hard Negatives:
another red cup
red bottle
cup far from laptop
```

Hard negatives 對於 fine-grained grounding 可能特別重要，但具體 sampling strategy 仍待驗證。

---

# 8. Abstention Design

目前預計至少比較以下三種：

## A. Explicit NONE

加入：

\[
A_{\text{NONE}}
\]

作為 candidate。

```text
R1
R2
R3
NONE
```

Ungroundable sample：

\[
Positive=A_{\text{NONE}}
\]

優點是設計簡單且符合 CLM candidate formulation。

潛在問題是所有 ungroundable cases 都被拉向同一 prototype，可能產生過度 abstain 等問題。

---

## B. Similarity Threshold

不加入 NONE embedding。

若：

\[
\max_i sim(S,A_i)<\delta
\]

則：

```text
ABSTAIN
```

需要研究 threshold calibration 與跨 dataset generalization。

---

## C. Groundability Prediction

額外加入：

\[
P(\text{groundable}|I,Q)
\]

Groundable：

```text
→ candidate ranking
```

Ungroundable：

```text
→ abstain
```

此方法結構較明確，但會增加額外 classifier，也可能偏離最純粹的 CLM formulation。

---

# 9. Potential Datasets / Benchmarks

初期可優先調查：

- RefCOCO
- RefCOCO+
- RefCOCOg
- GroundingME
- GroundingME / RefCOCOg Rejection split

GroundingME 特別適合本研究，因為它明確包含：

```text
Discriminative
Spatial
Limited
Rejection
```

四種能力。

其中 Rejection 可以直接用來測試：

> Ground-or-Abstain

是否真的有效。

---

# 10. Baselines

至少考慮：

### Traditional / Contrastive

- CLIP / SigLIP similarity
- region-text embedding matching

### Grounding Models

- Grounding DINO
- existing referring expression grounding model

### MLLM

依實驗成本挑選：

- Qwen-VL family
- InternVL family
- 其他可以直接輸出 bounding box 的 MLLM

### Recent Modular Grounding

- VGent

VGent 特別值得作為 conceptual baseline，因為它同樣把 grounding 從純 autoregressive coordinate generation 轉成 proposal selection。

---

# 11. Evaluation

主要 metrics：

### Grounding

```text
Accuracy
IoU
Acc@0.5 IoU
```

### Abstention

```text
Rejection Accuracy
AUROC
AUPRC
False Accept Rate
False Reject Rate
```

### Calibration

```text
ECE
confidence distribution
```

### Efficiency

```text
Latency
Throughput
GPU memory
FLOPs / approximate compute
```

最後主要觀察：

\[
Grounding\ Accuracy
+
Rejection\ Accuracy
+
Latency
\]

而不是只追求單一 accuracy。

---

# 12. Important Ablations

如果時間允許，優先做：

### Region Representation

```text
Crop only
vs
Crop + Position
vs
Crop + Global Context
```

### Negative Strategy

```text
Random Negatives
vs
Hard Negatives
```

### Abstention

```text
NONE Prototype
vs
Threshold
vs
Groundability Head
```

### Backbone

```text
Frozen
vs
Partial Fine-tuning
```

### Candidate Count

```text
10
50
100+
```

觀察 candidate 數量增加對 accuracy / latency 的影響。

---

# 13. Initial Experimental Roadmap

## Stage 0 — Reproduce CLM

先直接跑官方 CLM：

```text
state → action ranking
```

確認：

- InfoNCE pipeline
- state/action heads
- logit scaling
- ranking API
- training code

目的是避免在尚未理解原始 CLM 前就直接修改成 multimodal。

---

## Stage 1 — Simplified Visual CLM

先限制問題：

```text
Ground Truth Regions
+
Groundable Queries Only
```

目標：

> 驗證 Image + Query ↔ Region 的 contrastive formulation 是否可行。

暫時不碰 abstention。

---

## Stage 2 — Hard Negative Grounding

加入：

```text
similar objects
spatial relations
attribute confusion
```

測試 Visual CLM 是否能處理 fine-grained candidate discrimination。

---

## Stage 3 — Ground-or-Abstain

加入 ungroundable queries。

比較：

```text
NONE
Threshold
Groundability Head
```

---

## Stage 4 — Full Pipeline

改成：

```text
Detector / Proposal Model
        ↓
Visual CLM
        ↓
Region / Abstain
```

並加入 latency / throughput benchmark。

---

# 14. Main Risks

### Risk 1 — Proposal Bottleneck

如果 region proposal recall 不佳，CLM 無法補救。

因此需要把：

```text
Oracle proposal performance
```

與：

```text
End-to-end performance
```

分開報告。

---

### Risk 2 — CLM Advantage May Be Limited

Visual grounding 本身已經存在許多 contrastive / proposal-selection 方法。

因此不能只把：

```text
CLM → Vision Encoder
```

當作 novelty。

真正需要驗證的 contribution 應該來自：

```text
state-action formulation
+
hard-negative decision learning
+
explicit abstention
+
efficiency / calibration
```

---

### Risk 3 — NONE May Collapse

單一 NONE representation 可能造成：

```text
uncertain → NONE
```

過度拒絕。

因此 abstention mechanism 本身可能需要重新設計。

---

### Risk 4 — Multimodal Fusion May Become Too Heavy

如果最後需要一個大型 VLM 每次進行完整 cross-attention，可能失去 CLM 原本追求快速 System-One decision 的部分意義。

因此 architecture 應盡可能控制：

```text
expensive multimodal reasoning
```

的使用。

---

# 15. Expected Contribution

如果研究結果支持最初假設，可能形成三個主要 contribution：

1. **Visual CLM**

   將 CLM 的 state-action contrastive decision formulation 延伸到 multimodal visual grounding。

2. **Ground-or-Abstain**

   將 ungroundable visual query 明確納入 candidate decision problem，而非強制模型產生 region。

3. **Efficient Discriminative Grounding**

   探討 candidate-ranking architecture 與 autoregressive grounding 在：

   \[
   accuracy / rejection / latency
   \]

   之間的 trade-off。

目前這些仍是假設性的 contribution，需要實驗結果支持後才能確定。

---

# 16. References

### CLM

**Contrastive Language Models: A System One Model for Fast and Generalizable Decision-Making**  
Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré, Azalia Mirhoseini, 2026.

目前官方 citation 將其列為 project / Notion publication，而不是傳統 arXiv paper。

- [CLM Official GitHub Repository](https://github.com/Contrastive-LM/CLM?utm_source=chatgpt.com)
- [CLM Official Article / Project Page](https://contrastive-lm.notion.site/?utm_source=chatgpt.com)
- [CLM-8B Hugging Face Model](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B?utm_source=chatgpt.com)

### Visual Grounding

**GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation**, CVPR 2026.

提出 Discriminative、Spatial、Limited、Rejection 四類 visual grounding challenge，其中 Rejection 與本研究的 Ground-or-Abstain 最直接相關。

- [GroundingME Project Page](https://groundingme.github.io/?utm_source=chatgpt.com)

**VGent: Visual Grounding via Modular Design for Disentangling Reasoning and Prediction**, CVPR 2026.

使用 detector proposals + selection-style decoder，指出 autoregressive MLLM grounding 的 latency / hallucination 問題，與本計畫的 candidate-selection motivation 有高度相關。