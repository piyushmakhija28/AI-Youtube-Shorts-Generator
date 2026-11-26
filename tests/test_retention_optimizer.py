"""Triad test suite for 50M retention curve LLM script optimizer.

Validates positive golden path retention approval, iterative multi-round
hazard remediation, negative failure modes (empty transcripts, invalid types,
stagnation, max-rounds exhaustion, oscillatory cycle rejection), and edge cases
(ultra-short scripts, 60s boundaries, unicode/emojis, auto-duration estimation).
"""

from typing import Dict, List
from unittest.mock import MagicMock
import pytest

from shorts_generator.retention import (
    AdaptiveScriptRewriter,
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
from shorts_generator.sovereign.retention_optimizer import (
    ViralScriptOptimizer as SovereignViralScriptOptimizer,
)


@pytest.fixture
def golden_viral_transcript() -> str:
    """Fixture providing a high-retention viral YouTube Shorts script.

    Returns:
        Spoken script with immediate curiosity hook, zero hazards, and open loop tension.
    """
    return (
        "Wait until you see what happens next when liquid nitrogen hits boiling water. "
        "The reaction creates a massive pressure wave that shatters every expectation. "
        "Nobody expected this exact outcome, and the real reason will leave you stunned."
    )


@pytest.fixture
def suboptimal_hazard_transcript() -> str:
    """Fixture providing a script containing multiple drop-off hazards.

    Returns:
        Script with weak greeting hook, hesitation fillers, meta filler, and premature closure.
    """
    return (
        "Hey everyone, welcome back to another video. "
        "Before we get started, make sure to like and subscribe. "
        "Today i will show you how quantum levitation works. "
        "So basically, the superconductor locks into the magnetic field. "
        "Kind of like magic, but pure physics. "
        "In conclusion, that is all for today. Thanks for watching."
    )


class TestRetentionOptimizerPositive:
    """Positive golden path tests for retention curve optimization."""

    def test_golden_viral_script_approves_round_zero(self, golden_viral_transcript: str) -> None:
        """Verifies that an already optimized script passes immediately on round zero."""
        optimizer = ViralScriptOptimizer(target_score=9.0)
        result = optimizer.optimize_script(golden_viral_transcript, duration_sec=30.0)

        assert result.status == OptimizationStatus.APPROVED
        assert result.rounds_executed == 0
        assert result.initial_scorecard.composite_score >= 9.0
        assert result.initial_scorecard.hazard_count == 0
        assert result.optimized_transcript == golden_viral_transcript

    def test_suboptimal_script_reaches_approved_status(
        self, suboptimal_hazard_transcript: str
    ) -> None:
        """Verifies that a suboptimal script is iteratively remediated to approved status."""
        optimizer = ViralScriptOptimizer(target_score=9.0, max_iterations=3)
        result = optimizer.optimize_script(suboptimal_hazard_transcript, duration_sec=45.0)

        assert result.status in (OptimizationStatus.APPROVED, OptimizationStatus.REVISED_CONVERGED)
        assert result.rounds_executed >= 1
        assert result.final_scorecard.composite_score > result.initial_scorecard.composite_score
        assert result.final_scorecard.hazard_count < result.initial_scorecard.hazard_count
        assert len(result.applied_rewrites) > 0

    def test_retention_curve_auc_improvement(
        self, suboptimal_hazard_transcript: str
    ) -> None:
        """Verifies that mean retention area-under-curve improves post-optimization."""
        optimizer = ViralScriptOptimizer(target_score=9.0, max_iterations=3)
        result = optimizer.optimize_script(suboptimal_hazard_transcript, duration_sec=45.0)

        assert result.final_scorecard.mean_retention_auc > result.initial_scorecard.mean_retention_auc

    def test_drop_off_scanner_detects_all_categories(self) -> None:
        """Verifies that DropOffScanner detects all four drop-off hazard categories."""
        scanner = DropOffScanner()
        test_script = (
            "Hey guys, welcome back! "
            "Before we get started, check this out. "
            "So basically it works like this. "
            "In conclusion, that is it."
        )
        hazards = scanner.detect_hazards(test_script, duration_sec=30.0)
        detected_categories = {h.category for h in hazards}

        assert HazardCategory.WEAK_HOOK in detected_categories
        assert HazardCategory.META_FILLER in detected_categories
        assert HazardCategory.HESITATION_STAGNATION in detected_categories
        assert HazardCategory.PREMATURE_CLOSURE in detected_categories

    def test_curiosity_auditor_detects_open_loops(self) -> None:
        """Verifies that CuriosityAuditor detects open loops and flags terminal closures."""
        auditor = CuriosityAuditor()
        script = "Wait until you see what happens next with this secret experiment."
        loops = auditor.audit_loops(script, duration_sec=35.0)

        assert len(loops) >= 1
        assert any(l.is_terminal_closure for l in loops)

    def test_pacing_hazard_scorer_curve_length(self) -> None:
        """Verifies that PacingHazardScorer produces second-by-second points matching duration."""
        scorer = PacingHazardScorer()
        scorecard = scorer.compute_scorecard("Sample clean text", 25.0, [], [])

        assert len(scorecard.second_by_second_retention) == 26
        assert scorecard.second_by_second_retention[0] == 1.0
        assert scorecard.composite_score >= 8.0

    def test_adaptive_rewriter_substitutes_all_hazards(self) -> None:
        """Verifies that AdaptiveScriptRewriter removes offending phrases."""
        rewriter = AdaptiveScriptRewriter()
        scanner = DropOffScanner()
        script = "In conclusion, that is all for today."
        hazards = scanner.detect_hazards(script, 20.0)

        rewritten, mapping = rewriter.rewrite(script, hazards, [], 20.0)

        assert "In conclusion" not in rewritten
        assert len(mapping) > 0


class TestRetentionOptimizerNegative:
    """Negative and failure mode tests for script optimization."""

    def test_empty_transcript_raises_invalid_transcript_error(self) -> None:
        """Verifies that empty transcript string raises InvalidTranscriptError."""
        optimizer = ViralScriptOptimizer()
        with pytest.raises(InvalidTranscriptError, match="cannot be empty"):
            optimizer.optimize_script("")

    def test_whitespace_transcript_raises_invalid_transcript_error(self) -> None:
        """Verifies that pure whitespace transcript raises InvalidTranscriptError."""
        optimizer = ViralScriptOptimizer()
        with pytest.raises(InvalidTranscriptError, match="cannot be empty"):
            optimizer.optimize_script("   \n\t  ")

    def test_non_string_transcript_raises_invalid_transcript_error(self) -> None:
        """Verifies that non-string input raises InvalidTranscriptError."""
        optimizer = ViralScriptOptimizer()
        with pytest.raises(InvalidTranscriptError, match="must be a string"):
            optimizer.optimize_script(None)  # type: ignore[arg-type]

    def test_stagnation_detection_halts_loop(self) -> None:
        """Verifies that a score improvement below convergence threshold halts loop."""
        optimizer = ViralScriptOptimizer(target_score=9.99, convergence_delta=0.50, max_iterations=3)
        mock_rewriter = MagicMock()
        mock_rewriter.rewrite.side_effect = [
            ("Slight variation step 1", {"old": "new"}),
            ("Slight variation step 2", {"old": "new"}),
        ]
        optimizer.rewriter = mock_rewriter

        result = optimizer.optimize_script("Before we get started, watch this experiment.", 30.0)
        assert result.status in (OptimizationStatus.STAGNANT_CONVERGED, OptimizationStatus.MAX_ROUNDS_REACHED)

    def test_max_rounds_cap_enforced(self) -> None:
        """Verifies that optimizer strictly honors max_iterations limit."""
        optimizer = ViralScriptOptimizer(target_score=9.99, max_iterations=2, convergence_delta=0.001)
        result = optimizer.optimize_script("Before we start, so basically in conclusion.", 30.0)

        assert result.rounds_executed <= 2
        assert result.status in (OptimizationStatus.MAX_ROUNDS_REACHED, OptimizationStatus.APPROVED, OptimizationStatus.REVISED_CONVERGED)

    def test_anti_oscillation_guard_prevents_infinite_ping_pong(self) -> None:
        """Verifies that re-entering an already visited state triggers defect termination."""
        optimizer = ViralScriptOptimizer(target_score=9.9, max_iterations=3)
        original_script = "Before we get started, look here."
        alt_script = "Look here now."

        mock_rewriter = MagicMock()
        mock_rewriter.rewrite.side_effect = [
            (alt_script, {"Before we get started": ""}),
            (original_script, {"": "Before we get started"}),
        ]
        optimizer.rewriter = mock_rewriter

        result = optimizer.optimize_script(original_script, 30.0)
        assert result.status == OptimizationStatus.REJECTED_DEFECT


class TestRetentionOptimizerEdgeCases:
    """Edge case and boundary tests for script optimization."""

    def test_short_transcript_boundary(self) -> None:
        """Verifies that ultra-short 5-second scripts process cleanly without errors."""
        optimizer = ViralScriptOptimizer()
        result = optimizer.optimize_script("Wait until you see this.", duration_sec=5.0)

        assert result.status == OptimizationStatus.APPROVED
        assert len(result.final_scorecard.second_by_second_retention) == 6

    def test_long_transcript_boundary(self) -> None:
        """Verifies that 59.9-second scripts process full trajectory correctly."""
        optimizer = ViralScriptOptimizer()
        script = "Wait until you see what happens next in this incredible robotics challenge."
        result = optimizer.optimize_script(script, duration_sec=59.9)

        assert len(result.final_scorecard.second_by_second_retention) == 61

    def test_unicode_and_emojis(self) -> None:
        """Verifies that transcripts with emojis and unicode characters match without error."""
        optimizer = ViralScriptOptimizer()
        script = "🚀 Wait until you see this unbelievable magic trick! In conclusion, it works."
        result = optimizer.optimize_script(script, duration_sec=30.0)

        assert "In conclusion" not in result.optimized_transcript
        assert "🚀" in result.optimized_transcript

    def test_script_with_zero_hazards_but_weak_hook(self) -> None:
        """Verifies handling of script with no lexical drop-off words but mild hook."""
        optimizer = ViralScriptOptimizer()
        script = "This is a simple educational video about the laws of thermodynamics."
        result = optimizer.optimize_script(script, duration_sec=25.0)

        assert result.final_scorecard.composite_score >= 7.0

    def test_auto_duration_estimation_from_word_count(self) -> None:
        """Verifies that duration is automatically inferred when None is provided."""
        optimizer = ViralScriptOptimizer()
        transcript = " ".join(["word"] * 75)
        result = optimizer.optimize_script(transcript, duration_sec=None)

        assert len(result.final_scorecard.second_by_second_retention) >= 25

    def test_sovereign_forwarding_import(self) -> None:
        """Verifies that sovereign package forwards ViralScriptOptimizer correctly."""
        sovereign_optimizer = SovereignViralScriptOptimizer()
        res = sovereign_optimizer.optimize_script("Wait until you see this impossible stunt.", 20.0)

        assert res.status == OptimizationStatus.APPROVED
