"""Flow Matching / MeanFlow module for neural network weight generation.

This module implements Flow Matching and MeanFlow for learning to generate
neural network weights.

Supported methods:
- FlowMatching: Conditional Flow Matching based on DeepWeightFlow
  (https://arxiv.org/pdf/2601.05052)
- WeightSpaceMeanFlow: MeanFlow with average velocity for few-step generation
  (https://arxiv.org/pdf/2505.13447)

Main components:
- FlowMatching: Standard flow matching training and generation
- WeightSpaceMeanFlow: MeanFlow training and generation
- WeightSpaceFlowModel: Neural network for instantaneous velocity prediction
- WeightSpaceMeanFlowModel: Neural network for average velocity prediction
- WeightSpaceObjectStateDict: Architecture-agnostic weight space representation
- FlowMatchingConfig: Pydantic configuration models

Example:
    >>> from src.flowmatching import FlowMatching, WeightSpaceFlowModel
    >>> model = WeightSpaceFlowModel(input_dim=26506, hidden_dim=512)
    >>> fm = FlowMatching(target_loader, model, source_std=0.01)
    >>> fm.train(n_iters=10000, optimizer=optimizer)

    >>> from src.flowmatching import WeightSpaceMeanFlow, WeightSpaceMeanFlowModel
    >>> model = WeightSpaceMeanFlowModel(input_dim=99, hidden_dim=512)
    >>> mf = WeightSpaceMeanFlow(target_loader, model, source_std=0.01)
    >>> mf.train(n_iters=30000, optimizer=optimizer)
"""

from src.flowmatching.config import (
    FlowMatchingConfig,
    FlowModelConfig,
    GenerationConfig,
    MLPLayerConfig,
)
from src.flowmatching.flow_matching import FlowMatching
from src.flowmatching.flow_model import Bunch, WeightSpaceFlowModel
from src.flowmatching.mean_flow import WeightSpaceMeanFlow, adaptive_l2_loss
from src.flowmatching.meanflow_model import WeightSpaceMeanFlowModel
from src.flowmatching.pca import WeightSpacePCA, apply_pca_to_weights
from src.flowmatching.weight_space import WeightSpaceObjectStateDict

__all__ = [
    # Flow Matching
    "FlowMatching",
    "WeightSpaceFlowModel",
    # MeanFlow
    "WeightSpaceMeanFlow",
    "WeightSpaceMeanFlowModel",
    "adaptive_l2_loss",
    # Shared
    "WeightSpaceObjectStateDict",
    "Bunch",
    # PCA
    "WeightSpacePCA",
    "apply_pca_to_weights",
    # Config classes
    "FlowMatchingConfig",
    "FlowModelConfig",
    "MLPLayerConfig",
    "GenerationConfig",
]
