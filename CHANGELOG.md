# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2025-12-15

### Added
- Initial release of Chemeleon2
- Three-stage training pipeline: VAE → LDM → RL
- VAE module for encoding crystal structures into latent space
- LDM module with diffusion Transformer (DiT) architecture
- RL module with Group Relative Policy Optimization (GRPO)
- Custom reward system for material property optimization
- Support for multiple datasets: MP-20, Alex-MP-20, MP-120
- Comprehensive documentation with Jupyter Book
- Tutorial notebook for sampling and evaluation
- Benchmark datasets for de novo generation
  - 10,000 generated structures from MP-20 RL model
  - 10,000 generated structures from Alex-MP-20 RL model
- Testing suite with pytest (baseline, unit, integration tests)
- Pre-commit hooks for code quality (ruff, pyright)
- WandB integration for experiment tracking
- Configuration management with Hydra
- CrystalBatch data schema for crystal structure handling

### Documentation
- Training guide for VAE, LDM, RL, and predictor models
- Evaluation guide for sampling and metrics
- Custom reward implementation guide
- API reference documentation
- Contributing guidelines

[Unreleased]: https://github.com/hspark1212/chemeleon2/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/hspark1212/chemeleon2/releases/tag/v0.0.1
