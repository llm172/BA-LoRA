# BA-LoRA

Official implementation of **BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models**, accepted by ICLR 2026.

BA-LoRA builds on LoRA/PiSSA-style parameter-efficient fine-tuning and adds three output-space regularizers for mitigating catastrophic inheritance during downstream adaptation:

- consistency regularization for reducing knowledge drift;
- diversity regularization for mitigating representation collapse;
- SVD-based regularization for suppressing noisy high-frequency logit patterns.

The implementation is designed to reproduce the NLG experiments in the paper and can also be used as a regularization layer on top of LoRA-style adapters.

## Environment Setup

```bash
conda create -n ba-lora python=3.10
conda activate ba-lora

conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# Optional acceleration. After installing it, run with:
# ATTN_IMPLEMENTATION=flash_attention_2 bash scripts/ba-lora.sh
pip install flash-attn --no-build-isolation
```

## Quick Start

Update `BASE_MODEL` and `DATA_PATH` in `scripts/ba-lora.sh` if needed, then run:

```bash
bash scripts/ba-lora.sh
```

The script performs three steps:

1. initialize PiSSA weights when they are not already available;
2. fine-tune with BA-LoRA regularization;
3. generate responses and report accuracy.

## Key Training Arguments

The main BA-LoRA options are exposed in `train.py`:

```bash
--use_ba_lora True
--base_model_for_pt meta-llama/Llama-2-7b-hf
--lambda1 0.025
--lambda2 0.005
--lambda3 0.005
--lambda1_schedule cosine
--lambda_focus_schedule two_phase
--lambda_warmup_ratio 0.2
--lambda_ramp_up_ratio 0.05
--svd_k 10
--top_k_entropy 20
--distill_temp 2.0
--svd_frob_norm True
```

These defaults follow the LLaMA-2-7B NLG setting described in the paper.

## t-SNE Visualization

After fine-tuning, you can visualize the last hidden-layer features with:

```bash
python tsne_visualization.py
```

Before running it, set the feature directory in `tsne_visualization.py`. The script expects:

- `features_step_final.npy`
- `labels_step_final.npy`

The output figure is saved as `tsne_step_final.pdf`.

## Citation

If this repository is helpful for your research, please cite:

```bibtex
@inproceedings{
chang2026balora,
title={{BA}-Lo{RA}: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models},
author={Yupeng Chang and Yi Chang and Yuan Wu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=q0X9SiXiRO}
}
```

## Acknowledgements

This codebase is implemented on top of the excellent PiSSA codebase. We sincerely thank the PiSSA authors for releasing their implementation and for providing a strong foundation for efficient low-rank adaptation research.
