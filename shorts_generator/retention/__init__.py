"""Retention optimization and viral script tuning package for YouTube Shorts."""

from .retention_optimizer import (
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
