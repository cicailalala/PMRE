# [JBHI2026] PMRE

This repository is the official implementation of **[A Comprehensive Domain Generalization Framework for Prostate MRI Segmentation with Patch Mixing and Risk Extrapolation], JBHI2026**. 

## Data

Pass the dataset root through `--data-root`. It must contain the RUNMC, BMC,
I2CVB, UCL, BIDMC, and HK folders.

## Training

The example holds out RUNMC and trains on the remaining five domains:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_main.sh /path/to/dataset /path/to/output
```

Change `--target_dataset` to select another held-out domain.
Use absolute input and output paths when calling the script from another directory.


```bibtex
@article{qi2026PMRE,
  title={A Comprehensive Domain Generalization Framework for Prostate MRI Segmentation with Patch Mixing and Risk Extrapolation},
  author={Qi, Wenbo and Chan, Shing Chow and Wu, Jiafei and Liu, Zhe and Yu, Baosheng},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026}
}
```