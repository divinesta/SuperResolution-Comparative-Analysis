"""Fusion methods used in Phase 5."""

from app.fusion.back_projection import (
    BackProjectionFusionResult,
    back_projection_fusion,
    calculate_back_projection_weight,
    evaluate_back_projection_fusion,
)
from app.fusion.edge_aware import (
    DEFAULT_EDGE_NEDI_WEIGHTS,
    DEFAULT_EDGE_PERCENTILES,
    EdgeAwareFusionConfig,
    edge_aware_fusion,
    evaluate_edge_aware_grid,
    select_best_edge_aware_config,
    summarise_edge_aware_results,
)
from app.fusion.weighted import (
    DEFAULT_IMDN_WEIGHTS,
    evaluate_weight_grid,
    select_best_weight,
    summarise_weight_results,
    weighted_fusion,
)

__all__ = [
    "BackProjectionFusionResult",
    "DEFAULT_EDGE_NEDI_WEIGHTS",
    "DEFAULT_EDGE_PERCENTILES",
    "DEFAULT_IMDN_WEIGHTS",
    "EdgeAwareFusionConfig",
    "back_projection_fusion",
    "calculate_back_projection_weight",
    "edge_aware_fusion",
    "evaluate_back_projection_fusion",
    "evaluate_edge_aware_grid",
    "evaluate_weight_grid",
    "select_best_edge_aware_config",
    "select_best_weight",
    "summarise_edge_aware_results",
    "summarise_weight_results",
    "weighted_fusion",
]
