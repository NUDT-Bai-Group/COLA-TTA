# COLA: Context-aware Language-driven Test-time Adaptation<!-- omit in toc -->

Official implementation of **COLA: Context-aware Language-driven Test-time Adaptation**.  
[![IEEE TIP](https://img.shields.io/badge/Paper-IEEE%20TIP-009689.svg)](https://ieeexplore.ieee.org/document/11174099) [![arXiv](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2509.17598)

![Overview](media/overview.png)

> **TL;DR**: COLA is a CLIP-based **test-time adaptation (TTA) method without source data**. It adapts from **unlabeled target samples** via a **lightweight, efficient** module on a frozen CLIP backbone, achieving **strong performance**.

- **Label-space agnostic**: no need to align source/target label spaces.
- **Without source data**: adapt using unlabeled target data at test time.
- **Plug-and-play**: lightweight module on top of frozen CLIP; efficient and easy to integrate.


---

## Table of Contents <!-- omit in toc -->
- [Abstract](#abstract)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Experiments](#experiments)
  - [TTA](#tta)
  - [Datasets](#datasets)
- [Notes](#notes)
- [Updates \& Reproducibility](#updates--reproducibility)
- [Acknowledgements](#acknowledgements)
- [Reference](#reference)

---

## Abstract
Test-time adaptation (TTA) has gained increasing popularity for mitigating distribution shift while protecting data privacy. Most prior methods assume a paired source model and target domain share the same label space, limiting applicability. We explore a more general source model that adapts to multiple targets without shared labels by leveraging a pre-trained vision–language model (VLM), e.g., CLIP. While zero-shot VLMs are strong, they may not capture target-specific attributes. We propose **COLA**, a lightweight context-aware module (task-aware adapter, context-aware unit, residual connection) that plugs into a frozen VLM to extract task-specific, domain-specific, and prior knowledge efficiently. We also introduce **Class-Balanced Pseudo-labeling (CBPL)** to alleviate class imbalance. COLA is effective for TTA and class generalization.

---

## Quick Start

1) **Install the environment** — see [Installation](#installation). *(Both the conda env and the pip extras are required.)*  
2) **Set dataset root (REQUIRED)** — in `scripts/*.sh`, replace `PATH_TO_YOUR_DATASET_ROOT` with your absolute dataset path (e.g., `/abs/path/to/your/datasets`).  
3) **Run all TTA experiments**:
```bash
bash scripts/train.sh

```
Dataset layouts & list files: see **[DATASETS.md](./DATASETS.md)**.

---

## Installation
- Tested with **Python 3.8**, **PyTorch 1.11.0**, **CUDA 11.3**.  

```bash
# create & activate
conda env create -f environment.yaml
conda activate COLA

# install extras
pip install -r requirements.txt

```
> Ensure your PyTorch CUDA build matches your local drivers.

---

## Experiments

### TTA
- This repo supports **CLIP-based test-time adaptation (TTA) without source data** on five benchmarks: **VisDA-C**, **DomainNet-126**, **VLCS**, **PACS**, **OfficeHome**.
- Logs & results are saved under `./output/` with timestamped folders.
- Global data config: `configs/data/basic.yaml` (overridable via Hydra CLI).


### Datasets

See **[DATASETS.md](./DATASETS.md)** for detailed preparation (folder layouts, *_list.txt generation, Windows notes).

<details>
<summary><b>Per-dataset commands (click to expand)</b></summary>

**VisDA-C**
```bash
bash scripts/train_VISDA-C_target.sh            # ViT-B/16
```

**DomainNet-126**

```bash
bash scripts/train_domainnet-126_target.sh
```

**VLCS**
```bash
bash scripts/train_VLCS_target.sh               # ViT-B/16
bash scripts/train_VLCS_target_RN101.sh         # RN101
```

**PACS**
```bash
bash scripts/train_PACS_target.sh
```

**OfficeHome**
```bash
bash scripts/train_OfficeHome_target.sh
```
</details>
---

## Notes

- Set `DATA_ROOT` env var to point to datasets.
- `requirements.txt` pins pip deps; `environment.yaml` provides full conda env.


---

## Updates & Reproducibility
> Because the original experiments spanned a long period, involved multiple datasets, and the computing environment has changed, some exact original configs are no longer available. We made minor code improvements and bug fixes **without changing the core method**. Under fair and comparable settings, we reconstructed a subset of configs and re-ran the experiments; on several datasets, the current code yields results comparable to or slightly better than those reported in the paper, and the main conclusions remain unchanged.  
> **Reproduce** using the scripts in `scripts/` and default configs in `configs/`. Exact dependencies are in `requirements.txt` and `environment.yaml`.  


---

## Acknowledgements
Built upon:
- [CLIP](https://github.com/openai/CLIP)
- [AdaContrast](https://github.com/DianCh/AdaContrast)
- [CLIP-Adapter](https://github.com/CLIP-Adapter/CLIP-Adapter)

We thank the authors and the open-source community for their contributions.

---

## Reference
If you find this work helpful, please cite:
```bibtex
@ARTICLE{Zhang2025COLA,
  author={Zhang, Aiming and Yu, Tianyuan and Bai, Liang and Tang, Jun and Guo, Yanming and Ruan, Yirun and Zhou, Yun and Lu, Zhihe},
  journal={IEEE Transactions on Image Processing},
  title={COLA: Context-aware Language-driven Test-time Adaptation},
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2025.3607634}
}
```
