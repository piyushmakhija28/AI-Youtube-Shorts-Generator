"""Forwarding proxy for ViralScriptOptimizer satisfying GitHub Issue #987 spec.

Allows callers to import from shorts_generator.sovereign.retention_optimizer
identically to shorts_generator.retention.retention_optimizer.
"""

from ..retention.retention_optimizer import (
    AdaptiveScriptRewriter,
    ConvergenceExhaustionError,
    CuriosityAuditor,
    CuriosityLoop,
    DropOffScanner,
    HazardCategory,
    InvalidTranscriptError,
    OptimizationResult,
    OptimizationStatus,
    OscillatoryRewriteError,
    PacingHazardScorer,
    RetentionHazard,
    RetentionOptimizerError,
    RetentionScorecard,
    ViralScriptOptimizer,
)

__all__ = [
    "AdaptiveScriptRewriter",
    "ConvergenceExhaustionError",
    "CuriosityAuditor",
    "CuriosityLoop",
    "DropOffScanner",
    "HazardCategory",
    "InvalidTranscriptError",
    "OptimizationResult",
    "OptimizationStatus",
    "OscillatoryRewriteError",
    "PacingHazardScorer",
    "RetentionHazard",
    "RetentionOptimizerError",
    "RetentionScorecard",
    "ViralScriptOptimizer",
]
