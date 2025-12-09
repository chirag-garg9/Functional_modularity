# Modular Representation Learning Framework

A comprehensive experimental framework for studying modularity in representation learning using PyTorch. This framework enables systematic experimentation with different architectural components, loss functions, and regularization techniques to understand how they affect the modularity and disentanglement of learned representations.

## Features

- **Modular Architecture**: Configurable encoder-decoder with factor-specific subspaces
- **Multiple Loss Functions**: Reconstruction, contrastive (InfoNCE), mutual information
- **Regularization Techniques**: Orthogonality constraints, sparsity penalties, redundancy reduction
- **FiLM Modulation**: Feature-wise Linear Modulation for conditional generation
- **Comprehensive Metrics**: Modularity scores, DCI metrics, mutual information estimators
- **Synthetic Datasets**: dSprites, Shapes3D, and simple shapes for controlled experiments
- **Real Datasets**: CIFAR10, MNIST, Fashion-MNIST with synthetic factor generation
- **Experiment Management**: YAML-based configuration system for reproducible experiments
- **Analysis Tools**: Comprehensive plotting and analysis scripts

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd modular-representation-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
modular-representation-learning/
├── models/                 # Model architectures
│   ├── base_models.py     # Base encoder-decoder
│   ├── modular_models.py  # Modular architectures
│   └── film_models.py     # FiLM modulation
├── losses/                # Loss functions
│   ├── reconstruction_losses.py
│   ├── contrastive_losses.py
│   ├── mutual_info_losses.py
│   └── regularization_losses.py
├── metrics/               # Evaluation metrics
│   ├── modularity_metrics.py
│   ├── dci_metrics.py
│   ├── mutual_info_metrics.py
│   └── classification_metrics.py
├── datasets/              # Dataset implementations
│   ├── synthetic_datasets.py
│   └── real_datasets.py
├── configs/               # Experiment configurations
│   └── experiment_configs.py
├── train.py              # Main training script
├── analyze.py            # Analysis and visualization
├── run_experiments.py    # Experiment manager
└── requirements.txt      # Dependencies
```

## Quick Start

### 1. Save Experiment Configurations

First, save all experiment configurations:

```bash
python train.py --save-configs
```

This creates YAML configuration files for all experiments (E1-E7) in the `configs/` directory.

### 2. Run a Single Experiment

Train a specific experiment:

```bash
python train.py --experiment E1
```

Available experiments:
- **E1**: Baseline encoder-decoder with reconstruction loss only
- **E2**: Modular architecture with factor-specific subspaces
- **E3**: Modular architecture with contrastive learning
- **E4**: Modular architecture with orthogonality constraints
- **E5**: Modular architecture with sparsity regularization
- **E6**: Modular architecture with redundancy reduction (Barlow Twins)
- **E7**: Modular architecture with FiLM modulation

### 3. Run All Experiments

Use the experiment manager to run all experiments sequentially:

```bash
python run_experiments.py
```

### 4. Analyze Results

Generate comprehensive analysis:

```bash
python analyze.py --generate-all
```

## Experiment Configurations

Each experiment (E1-E7) is designed to study the impact of different components:

### E1: Baseline
- **Purpose**: Establish baseline performance
- **Components**: Basic encoder-decoder, reconstruction loss only
- **Expected**: Standard autoencoder behavior

### E2: Modular Architecture
- **Purpose**: Study impact of modular design
- **Components**: Factor-specific subspaces, reconstruction loss
- **Expected**: Better factor separation than baseline

### E3: Contrastive Learning
- **Purpose**: Study impact of contrastive learning
- **Components**: Modular architecture + InfoNCE loss
- **Expected**: Improved representation quality and factor separation

### E4: Orthogonality Constraints
- **Purpose**: Study impact of factor orthogonality
- **Components**: Modular architecture + contrastive + orthogonality loss
- **Expected**: Reduced factor interference

### E5: Sparsity Regularization
- **Purpose**: Study impact of sparsity
- **Components**: Previous + sparsity regularization
- **Expected**: More focused factor representations

### E6: Redundancy Reduction
- **Purpose**: Study impact of redundancy reduction
- **Components**: Previous + Barlow Twins-style redundancy loss
- **Expected**: Reduced redundancy between factors

### E7: FiLM Modulation
- **Purpose**: Study impact of conditional generation
- **Components**: Previous + FiLM modulation
- **Expected**: Better conditional generation and factor control

## Metrics

The framework computes comprehensive metrics:

### Modularity Metrics
- **Modularity Score**: Measures how well factors are separated
- **Factor Alignment**: Measures alignment between latent dimensions and factors
- **Factor Interference**: Measures cross-factor interference

### DCI Metrics
- **Disentanglement**: How well each factor is captured by a single latent dimension
- **Completeness**: How well each latent dimension captures a single factor
- **Informativeness**: How well factors can be predicted from latents

### Mutual Information Metrics
- **Total MI**: Total mutual information between latents and factors
- **Factor-wise MI**: Mutual information for each factor
- **Subspace MI**: Mutual information between different subspaces

### Classification Metrics
- **Downstream Accuracy**: Classification accuracy using learned representations
- **Factor-wise Accuracy**: Classification accuracy for individual factors
- **Transferability**: Ability to transfer representations to new tasks

## Customization

### Adding New Datasets

1. Create a new dataset class in `datasets/synthetic_datasets.py` or `datasets/real_datasets.py`
2. Implement the required methods (`__getitem__`, `get_factor_names`, `get_factor_dims`)
3. Add the dataset to the `get_dataset()` function

### Adding New Loss Functions

1. Create a new loss class in the appropriate file in `losses/`
2. Inherit from `nn.Module` and implement the `forward()` method
3. Add the loss to the `ExperimentRunner._setup_losses()` method

### Adding New Metrics

1. Create a new metric class in the appropriate file in `metrics/`
2. Implement the computation methods
3. Add the metric to the `ExperimentRunner._setup_metrics()` method

### Creating Custom Experiments

1. Modify the configuration in `configs/experiment_configs.py`
2. Add new experiment configurations to `EXPERIMENT_CONFIGS`
3. Run with `python train.py --experiment YOUR_EXPERIMENT`

## Analysis and Visualization

The analysis script provides comprehensive visualization:

### Training Curves
- Loss curves over training epochs
- Metric evolution during training
- Comparison plots between experiments

### Final Metrics Comparison
- Bar charts comparing final metrics across experiments
- Statistical significance testing
- Ablation study plots

### Factor Analysis
- Factor-wise classification accuracy
- Factor alignment and interference analysis
- Correlation heatmaps

### Ablation Studies
- Component-wise impact analysis
- Progressive improvement visualization
- Statistical significance testing

## Logging and Monitoring

### TensorBoard
Enable TensorBoard logging in the configuration:
```yaml
logging:
  use_tensorboard: true
  log_dir: "./logs"
```

View logs:
```bash
tensorboard --logdir ./logs
```

### Weights & Biases
Enable W&B logging:
```yaml
logging:
  use_wandb: true
```

### Checkpoints
Models are automatically saved during training:
- Regular checkpoints every N epochs
- Best model based on validation loss
- Training history and metrics

## Reproducibility

The framework ensures reproducibility through:
- Fixed random seeds
- Deterministic operations
- Comprehensive logging
- Version-controlled configurations

## Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Batch Size**: Adjust based on available memory
3. **Data Loading**: Use multiple workers for faster data loading
4. **Mixed Precision**: Consider using `torch.cuda.amp` for faster training

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model size
2. **Slow Training**: Check data loading, use GPU, enable mixed precision
3. **Poor Results**: Check hyperparameters, try different loss weights
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{modular_representation_learning_2024,
  title={Modular Representation Learning Framework},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/modular-representation-learning}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by disentanglement literature (Higgins et al., Locatello et al.)
- Built on PyTorch ecosystem
- Uses metrics from disentanglement-lib
