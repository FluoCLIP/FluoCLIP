# FluoCLIP 

This repository is the official PyTorch implementation of ["FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy"](https://arxiv.org/abs/2602.23791), accepted at CVPR 2026.

For more information, please check our [Project Page](https://fluoclip.github.io/)

## Environment Settings
All experiments were executed on a single 3090 GPU.

### Clone repository
```
git clone https://github.com/FluoCLIP/FluoCLIP.git
```

### Create conda environment
```
conda env create -f env.yml
pip install -e .
```

### Install CLIP
```
cd CLIP
pip install -e .
cd ..
```

### Run Code
```
python scripts/run.py --config config.yaml
```

## Acknowledgement

Many thanks to the following repositories

- [OrdinalCLIP](https://github.com/xk-huang/OrdinalCLIP)
- [CLIP](https://github.com/openai/CLIP)

## Citation
If you find our paper interesting, please cite
```
@article{park2026fluoclip,
  title={FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy},
  author={Park, Hyejin and Yoon, Jiwon and Park, Sumin and Kim, Suree and Jang, Sinae and Lee, Eunsoo and Kang, Dongmin and Min, Dongbo},
  journal={arXiv preprint arXiv:2602.23791},
  year={2026}
}
```
