"""Main executable scripts for DriftWeight.

This package contains the main executable scripts:
- generate_training_data: Train MLP models on MNIST to generate weight data
- train_flow_model: Train MeanFlow or Flow Matching on collected weights

Usage:
    python -m src.main.generate_training_data [options]
    python -m src.main.train_flow_model [options]
"""
