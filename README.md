# 2.5 CNN: Leveraging 2D CNNs to Pretrain 3D Models in Low-Data Regimes for COVID-19 Diagnosis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Felectronics14132571-blue)](https://doi.org/10.3390/electronics14132571)

Official code for **2.5 CNN**, a two-stage pipeline for volumetric CT classification under 3D label scarcity. Stage 1 trains a 2D CNN on individual CT slices, which expands the effective training set and learns slice-level pathology features. Stage 2 freezes that backbone and fits a lightweight 3D head over stacked slice embeddings, recovering the volumetric context pure-2D methods discard.

**Result:** 94.73 % weighted / 95.35 % unweighted accuracy on MosMed (1130 chest CT scans, 5-class severity), surpassing both purely 2D and purely 3D pipelines trained on the same data. Per-task and per-severity breakdowns are in the paper.

## Citation

Garg, A.; Garg, A.; Duncan, D. *2.5 CNN: Leveraging 2D CNNs to Pretrain 3D Models in Low-Data Regimes for COVID-19 Diagnosis.* Electronics **2025**, 14 (13), 2571. https://doi.org/10.3390/electronics14132571

```bibtex
@Article{electronics14132571,
  AUTHOR  = {Garg, Arnav and Garg, Aksh and Duncan, Dominique},
  TITLE   = {2.5 CNN: Leveraging 2D CNNs to Pretrain 3D Models in Low-Data Regimes for COVID-19 Diagnosis},
  JOURNAL = {Electronics},
  VOLUME  = {14},
  YEAR    = {2025},
  NUMBER  = {13},
  ARTICLE-NUMBER = {2571},
  DOI     = {10.3390/electronics14132571}
}
```

## Install

```bash
git clone https://github.com/arnavgarg233/2.5CNN.git
cd 2.5CNN
conda create -n 25cnn python=3.10 -y && conda activate 25cnn
# install PyTorch for your CUDA / MPS / CPU build, then:
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch ≥ 2.0, MONAI ≥ 1.2. A GPU is recommended for the 3D stage.

## Run

```bash
python src/launch.py --config Configs/config.yaml        # 5-class severity (default)
python src/launch.py --config Configs/1v3.yaml           # CT-1 vs CT-3 binary
python src/evaluate.py --config Configs/config.yaml --checkpoint outputs/checkpoints/best.pt
```

Batch sweeps (the ablation tables in the paper) run through the GPU scheduler:

```bash
cd multiplexer && python config_generator.py && python scheduler.py
```

## Data

[MosMed](https://mosmed.ai/en/datasets/covid19_1110) volumes are converted into per-slice `.npy` arrays under `data/{train,val}/class_{0..4}` (CT-0 normal → CT-4 critical):

```bash
python Scripts/store_data_in_files.py --src /path/to/MosMed --dst data/
python Scripts/store_slices.py        --src data/ --dst data_slices/
```

Class imbalance is handled by `Data/weighted_sampler.py` and weighted cross-entropy; splits are subject-level to prevent patient leakage.

## Structure

```
Configs/      YAML training configs, one per task / split
Data/         dataloaders, augmentations, weighted sampler
Models/       models2d.py (stage 1) · models3d.py (baseline) · models_half.py (2.5D)
Scripts/      volume → slice conversion, visualization, analysis
Utils/        helpers, timing, system checks
multiplexer/  GPU job scheduler for sweeps
src/          launch.py · evaluate.py
```

## License

MIT — see [LICENSE](LICENSE). The article is open access under CC BY 4.0.
Copyright © 2025 Arnav Garg, Aksh Garg, Dominique Duncan.
