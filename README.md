[![arXiv](https://img.shields.io/badge/arXiv-2501.03461-b31b1b.svg)](https://arxiv.org/abs/2501.03461) [![Kaggle](https://img.shields.io/badge/Kaggle-LabRF-blue?logo=kaggle)](https://www.kaggle.com/datasets/abcxyzi/self-supervised-rf-adaptation-labrf-dataset) [![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# Self-Supervised Adaptation for Radar Signal Recognition

This repo contains the LabRF dataset accompanying the paper "Self-Supervised RF Adaptation for Robust Radar Signal Recognition", submitted to the IEEE Transactions on Aerospace and Electronic Systems (TAES). It also contains RadChar-Holdout and the RadChar-nShot support-set draws used in the paper.

The paper extends our IEEE International Workshop on Machine Learning for Signal Processing (MLSP) 2025 paper, "Few-Shot Radar Signal Recognition through Self-Supervised Learning and Radio Frequency Domain Adaptation". The datasets introduced in that paper are available at [RadCharSSL](https://github.com/abcxyzi/RadCharSSL).

You can access the MLSP preprint 📄 here: [https://arxiv.org/abs/2501.03461](https://arxiv.org/abs/2501.03461)

You can download LabRF ⬇️ here: [https://www.kaggle.com/datasets/abcxyzi/labrf-taes-2026](https://www.kaggle.com/datasets/abcxyzi/labrf-taes-2026)

## Quick Links

- [Dataset Overview](#dataset-overview)
- [LabRF (Real-World RF Data)](#labrf-real-world-rf-data)
- [RadChar-Holdout and Support-Set Draws](#radchar-holdout-and-support-set-draws)
- [Usage](#usage)
- [Other Datasets](#other-datasets)
- [Download](#download)
- [Citation](#citation)

## Dataset Overview

⚙️ LabRF contains over-the-air captures of everyday civilian RF activity, recorded with a commercial off-the-shelf software-defined radio (SDR). It is used for self-supervised pre-training only, so its frames carry no labels.

LabRF contains four subsets:

1. **LabRF.fm1** - FM broadcast band, captured outdoors at a narrow sampling rate.
2. **LabRF.fm2** - FM broadcast band, captured outdoors at a wide sampling rate.
3. **LabRF.sb1** - 2.4 GHz ISM band, captured indoors at a narrow sampling rate.
4. **LabRF.sb2** - 2.4 GHz ISM band, captured indoors at a wide sampling rate.

## LabRF (Real-World RF Data)

**LabRF** is our own dataset, captured with a HackRF One SDR. The data were collected across several days, during which small batches of I/Q samples were collected periodically within fixed time intervals at various times of the day.

⚙️ Each LabRF subset has the following characteristics:

| Subset | Band | Setting | Centre frequency | Sampling rate | I/Q samples per frame | Temporal resolution | Frames |
|---|---|---|---|---|---|---|---|
| LabRF.fm1 | FM broadcast | Outdoor | 106.9 MHz | 1 MHz | 1,024 | 1 μs | 500,000 |
| LabRF.fm2 | FM broadcast | Outdoor | 106.9 MHz | 20 MHz | 1,024 | 0.05 μs | 500,000 |
| LabRF.sb1 | 2.4 GHz ISM | Indoor | 2.402 GHz | 1 MHz | 512 | 1 μs | 500,000 |
| LabRF.sb2 | 2.4 GHz ISM | Indoor | 2.402 GHz | 3.2 MHz | 512 | 0.3125 μs | 500,000 |

Each frame is a complex baseband I/Q sequence sampled at a constant rate. Each subset is provided as an `.h5` file holding a single dataset `iq` of shape `(500000, N)` in `complex64`, where `N` is the number of I/Q samples per frame. Frames are stored as received, without normalisation. The capture settings are stored as file attributes (`sampling_rate`, `center_frequency`, `samples_per_frame`, `number_of_frames`).

## RadChar-Holdout and Support-Set Draws

The paper also uses the following RadChar data, released here alongside LabRF:

- **RadChar-Holdout** - 100,000 further RadChar frames, disjoint from RadChar-Eval and every support set, and unused in training or model selection. This is used for evaluation only.
- **RadChar-nShot-Draws** - ten independent support-set draws for each of n = 1, 5, 10, each holding 205n labelled frames. These are used for few-shot fine-tuning only.

Both follow the format of RadChar-nShot and RadChar-Eval in [RadCharSSL](https://github.com/abcxyzi/RadCharSSL): a dataset `iq` of shape `(N, 512)` in `complex128` and a dataset `labels`, where the class label is `signal_type` and the SNR is `signal_to_noise_ratio`. RadChar-Holdout holds these at the root of its file. The draws share one file, with each draw in its own group named `{n}shot/draw{k}`, e.g. `1shot/draw1`.

## Usage

Install packages:

```bash
pip install -r requirements.txt
```

Load the selected dataset:

```python
from load_dataset import LabRFDataModule

LabRF = LabRFDataModule(data_path="./labrf/")
ds = LabRF.loadPretrainingSet(subset="fm1")    # One of fm1, fm2, sb1, sb2
print(ds[0]["iq_cplx"].shape)                  # torch.Size([1024])

ds_finetune = LabRF.loadFinetuningSet(fewshot=1, draw=1)   # fewshot in 1, 5, 10 and draw in 1 to 10
ds_eval = LabRF.loadHoldoutSet()
print(ds_eval[0]["label_class"], ds_eval[0]["snr"])
```

## Other Datasets

The paper also uses the following datasets, which are described and linked in [RadCharSSL](https://github.com/abcxyzi/RadCharSSL):

- **RadChar-SSL**, **RadChar-nShot** and **RadChar-Eval** (radar data, ours) for pre-training, few-shot fine-tuning and evaluation.
- **RadioML** (telecommunications data), **DeepRadar** (radar data) and **RadarComm** (mixed telecommunications and radar data) for pre-training.

## Download

⬇️ LabRF, RadChar-Holdout and the support-set draws can be downloaded from [Kaggle](https://www.kaggle.com/datasets/abcxyzi/labrf-taes-2026).

| Dataset | File | Size |
|---|---|---|
| LabRF.fm1 | `LabRF-fm1.h5` | 326 MB |
| LabRF.fm2 | `LabRF-fm2.h5` | 320 MB |
| LabRF.sb1 | `LabRF-sb1.h5` | 176 MB |
| LabRF.sb2 | `LabRF-sb2.h5` | 146 MB |
| RadChar-Holdout | `RadChar-Holdout.h5` | 798 MB |
| RadChar-nShot-Draws | `RadChar-nShot-Draws.h5` | 262 MB |

## Citation

💡 Please cite the following papers if you find RadChar or LabRF helpful for your research. Cheers.

```latex
@inproceedings{huang2025fewshot,
  title     = {Few-Shot Radar Signal Recognition through Self-Supervised Learning and Radio Frequency Domain Adaptation},
  author    = {Zi Huang and Simon Denman and Akila Pemasiri and Clinton Fookes and Terrence Martin},
  booktitle = {Proc. IEEE 35th Int. Workshop Mach. Learn. Signal Process. (MLSP)},
  year      = {2025}
}

@unpublished{huang2026selfsupervised,
  title  = {Self-Supervised RF Adaptation for Robust Radar Signal Recognition},
  author = {Zi Huang and Simon Denman and Akila Pemasiri and Clinton Fookes and Terrence Martin},
  note   = {Submitted to IEEE Transactions on Aerospace and Electronic Systems},
  year   = {2026}
}
```
