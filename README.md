# Official Implementation for Our Paper

This repository contains the official implementation for our paper submitted for review.

-----

## Environment Setup

```bash
# Create a conda environment
conda create -n project-env python=3.10
conda activate project-env

# Install CUDA Toolkit, PyTorch and other dependencies
conda install nvidia/label/cuda-12.1.0::cuda-toolkit
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# Optional: Install Flash Attention for acceleration
pip install flash-attn --no-build-isolation
```

-----

## Quick Start

```bash
# Start training with default parameters
# Please ensure the script name in `scripts/` matches this command.
bash scripts/run_training.sh
```

### Visualizing Last Hidden Layer Features with t-SNE

After fine-tuning the model, you can visualize the last hidden layer features using t-SNE to analyze class separability in the feature space.

1.  **Prepare Data**:

      * Save the last hidden layer features and labels as `.npy` files:
          * `features_step_final.npy`: Features.
          * `labels_step_final.npy`: Labels.

2.  **Run Script**:

      * Update paths in `tsne_visualization.py`:
        ```python
        last_hidden_features_dir = '/path/to/last_hidden_features'
        output_dir = './output'
        step = 'final'
        ```
      * Run:
        ```bash
        python tsne_visualization.py
        ```

3.  **Output**:

      * The t-SNE plot (`tsne_step_final.pdf`) will be saved in the specified `output_dir`.
