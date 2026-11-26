"""Verification and scorecard audit script for 50M retention curve script optimizer.

Applies ViralScriptOptimizer to an authentic 45-second YouTube Shorts transcript
loaded with viral drop-off hazards, renders second-by-second ASCII retention
trajectories, and verifies multi-agent convergence and quality metrics.
"""

import sys
from typing import List

from shorts_generator.retention import (
    OptimizationResult,
    OptimizationStatus,
    RetentionScorecard,
    ViralScriptOptimizer,
)


def render_ascii_curve(baseline: List[float], optimized: List[float], width: int = 40) -> str:
    """Renders a formatted ASCII curve comparing baseline and optimized retention.

    Args:
        baseline: Second-by-second retention curve R_raw(t).
        optimized: Second-by-second retention curve R_opt(t).
        width: Character width for progress bar display.

    Returns:
        Multi-line string containing the rendered ASCII timeline.
    """
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"{'SEC':<4} | {'BASELINE R_raw(t)':<22} | {'OPTIMIZED R_opt(t)':<22} | {'LIFT'}")
    lines.append("-" * 72)

    sample_rate = max(1, len(baseline) // 15)
    for t in range(0, len(baseline), sample_rate):
        raw_val = baseline[t]
        opt_val = optimized[t] if t < len(optimized) else raw_val
        lift_pct = ((opt_val - raw_val) / max(0.01, raw_val)) * 100.0

        raw_bars = int(raw_val * 12)
        opt_bars = int(opt_val * 12)

        raw_display = f"{raw_val*100:>5.1f}% [{'#' * raw_bars}{'.' * (12 - raw_bars)}]"
        opt_display = f"{opt_val*100:>5.1f}% [{'#' * opt_bars}{'.' * (12 - opt_bars)}]"
        lift_str = f"+{lift_pct:>5.1f}%" if lift_pct >= 0 else f"{lift_pct:>5.1f}%"

        lines.append(f"T+{t:02d}s | {raw_display:<22} | {opt_display:<22} | {lift_str}")

    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    """Runs the retention verification scorecard audit.

    Returns:
        Exit code 0 on success, 1 on failure.
    """
    print("\n" + "=" * 75)
    print("  AUTONOMOUS MULTI-AGENT 50M RETENTION CURVE OPTIMIZER AUDIT")
    print("  GitHub Issue #987 | ViralScriptOptimizer Verification")
    print("=" * 75 + "\n")

    realistic_script = (
        "Hey everyone, welcome back to my channel! "
        "Before we get started, make sure to like and subscribe. "
        "Today in this video, I will show you how autonomous AI swarms actually work. "
        "So basically, the agent breaks the user prompt into sub-tasks and delegates them. "
        "Kind of like an assembly line, you know what I mean? "
        "Anyway, that is pretty much the core mechanism. "
        "In conclusion, that is all for today. Thanks for watching!"
    )
    duration_sec = 45.0

    print("--- RAW SCRIPT TRANSCRIPT ---")
    print(realistic_script)
    print(f"\nTarget Video Duration: {duration_sec} seconds")

    optimizer = ViralScriptOptimizer(target_score=9.0, max_iterations=3)
    result: OptimizationResult = optimizer.optimize_script(
        realistic_script, duration_sec=duration_sec
    )

    print("\n" + "=" * 75)
    print("  OPTIMIZATION DELIBERATION SUMMARY")
    print("=" * 75)
    print(f"Status:             {result.status.value.upper()}")
    print(f"Rounds Executed:    {result.rounds_executed}")
    print(f"Hazards Remediated: {len(result.remediated_hazards)}")
    print(f"Applied Rewrites:   {len(result.applied_rewrites)}")

    print("\n--- REWRITE TRANSFORMATIONS ---")
    for old, new in result.applied_rewrites.items():
        print(f"  [-] REMOVED:   '{old}'")
        print(f"  [+] INJECTED:  '{new}'")
        print()

    print("--- OPTIMIZED VIRAL TRANSCRIPT ---")
    print(result.optimized_transcript)

    print("\n" + "=" * 75)
    print("  RETENTION SCORECARD AUDIT")
    print("=" * 75)
    init_sc: RetentionScorecard = result.initial_scorecard
    final_sc: RetentionScorecard = result.final_scorecard

    print(f"Metric                    | Baseline        | Optimized       | Gain")
    print("-" * 65)
    score_gain = final_sc.composite_score - init_sc.composite_score
    print(f"Composite Score (0-10)    | {init_sc.composite_score:>6.2f} / 10.0 | {final_sc.composite_score:>6.2f} / 10.0 | {score_gain:+6.2f}")
    auc_gain = (final_sc.mean_retention_auc - init_sc.mean_retention_auc) * 100.0
    print(f"Mean Retention AUC        | {init_sc.mean_retention_auc*100:>6.1f}%       | {final_sc.mean_retention_auc*100:>6.1f}%       | {auc_gain:+6.1f}%")
    print(f"Active Hazard Count       | {init_sc.hazard_count:>6d}          | {final_sc.hazard_count:>6d}          | {final_sc.hazard_count - init_sc.hazard_count:+6d}")
    print(f"Cumulative Hazard Penalty | {init_sc.hazard_penalty_sum:>6.2f}          | {final_sc.hazard_penalty_sum:>6.2f}          | {final_sc.hazard_penalty_sum - init_sc.hazard_penalty_sum:+6.2f}")
    print(f"Hook Velocity Score       | {init_sc.hook_quality_score:>6.2f} / 1.0  | {final_sc.hook_quality_score:>6.2f} / 1.0  | {final_sc.hook_quality_score - init_sc.hook_quality_score:+6.2f}")
    print(f"Curiosity Loop Tension    | {init_sc.curiosity_loop_score:>6.2f} / 1.0  | {final_sc.curiosity_loop_score:>6.2f} / 1.0  | {final_sc.curiosity_loop_score - init_sc.curiosity_loop_score:+6.2f}")

    print("\n" + render_ascii_curve(
        init_sc.second_by_second_retention,
        final_sc.second_by_second_retention,
    ))

    assert final_sc.composite_score >= 8.5, f"Score {final_sc.composite_score} below acceptance threshold 8.5"
    assert final_sc.hazard_count == 0, f"Remaining hazards {final_sc.hazard_count} > 0"
    assert final_sc.mean_retention_auc > init_sc.mean_retention_auc, "AUC did not increase"

    print("\n[SUCCESS] All verification gates passed with zero residual hazards!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
