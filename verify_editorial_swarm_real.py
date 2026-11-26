"""Production verification script for multi-agent swarm editorial consensus deliberation.

Executes realistic candidate cuts across multiple rounds, demonstrates Pareto
conflict resolution, and renders an ANSI colorized scorecard audit report.
"""

import sys
from typing import Dict

from shorts_generator.editorial import (
    ConsensusDecision,
    ConsensusRoundRecord,
    ConsensusStatus,
    CutCandidate,
    EditorialRole,
    EditorialSwarmFactory,
    SwarmEditorialConsensus,
)


def _render_scorecard(decision: ConsensusDecision) -> None:
    """Print a structured ANSI scorecard summarizing the deliberation audit trail.

    Args:
        decision: The final ConsensusDecision from the deliberation engine.
    """
    cyan = "\033[96m"
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    bold = "\033[1m"
    reset = "\033[0m"

    print(f"\n{bold}{cyan}╔══════════════════════════════════════════════════════════════════════════════╗{reset}")
    print(f"{bold}{cyan}║       MULTI-AGENT SWARM EDITORIAL CONSENSUS DELIBERATION AUDIT               ║{reset}")
    print(f"{bold}{cyan}╚══════════════════════════════════════════════════════════════════════════════╝{reset}\n")

    for round_rec in decision.round_history:
        print(f"{bold}--- ROUND {round_rec.round_number} DELIBERATION TRACE ---{reset}")
        print(f"Composite Score: {bold}{round_rec.composite_score:.2f} / 10.0{reset} | Unanimous: {round_rec.unanimous_approved}")
        print("Agent Ballots:")
        for role, ballot in round_rec.ballots.items():
            status_color = green if ballot.pass_vote else yellow
            vote_str = "PASS" if ballot.pass_vote else "REVISE"
            print(f"  [{role.value.upper():14s}] Score: {status_color}{ballot.score:4.1f}{reset} | Vote: {status_color}{vote_str}{reset}")
            for s in ballot.strengths:
                print(f"    {green}+ {s}{reset}")
            for issue in ballot.blocking_issues:
                print(f"    {red}- {issue}{reset}")
            for directive in ballot.directives:
                print(f"    {cyan}* Directive: {directive.parameter} -> {directive.adjustment:+.2f} ({directive.rationale}){reset}")

        if round_rec.applied_adjustments:
            print(f"  {bold}Pareto Arbitrated Adjustments Applied:{reset}")
            for param, delta in round_rec.applied_adjustments.items():
                print(f"    -> {param}: {delta:+.3f}")
        print()

    print(f"{bold}=== FINAL VERDICT ==={reset}")
    verdict_color = green if decision.status == ConsensusStatus.APPROVED else red
    print(f"Status:       {bold}{verdict_color}{decision.status.value.upper()}{reset}")
    print(f"Total Rounds: {decision.total_rounds}")
    print(f"Final Score:  {bold}{decision.final_score:.2f} / 10.0{reset}")
    print(f"Unanimous:    {decision.unanimous}")
    print(f"Summary:      {decision.summary}\n")


def run_verification() -> int:
    """Run production end-to-end verification of the swarm editorial consensus engine.

    Returns:
        Exit code: 0 on successful verification pass, 1 on failure.
    """
    print("Initiating Multi-Agent Swarm Editorial Consensus Verification...")

    candidate = CutCandidate(
        clip_id="youtube_ukf6t47nqpg_short_001",
        title="Autonomous AI Multi-Agent Architecture Breakdown",
        transcript=(
            "Hey everyone, before we get started, let me tell you why traditional monolithic pipelines "
            "fail completely when scaling YouTube Shorts. Today we deploy an autonomous consensus swarm."
        ),
        duration_sec=44.0,
        hook_duration_sec=4.2,
        words_per_minute=145.0,
        face_centering_iou=0.78,
        cut_frequency_sec=4.8,
        subtitle_sync_drift_ms=90.0,
        integrated_lufs=-18.0,
        true_peak_dbfs=-0.4,
        ducking_attenuation_db=8.0,
        curiosity_loop_count=1,
        dynamic_zoom_count=2,
    )

    swarm = SwarmEditorialConsensus(
        max_rounds=3,
        consensus_threshold=8.8,
        convergence_tolerance=0.04,
    )

    decision = swarm.deliberate(candidate)
    _render_scorecard(decision)

    if decision.status == ConsensusStatus.APPROVED and decision.total_rounds >= 2:
        print("\033[92m✔ Verification PASSED: Candidate converged from Round 1 sub-optimal state to Round 2+ certified approval.\033[0m")
        return 0
    elif decision.status == ConsensusStatus.APPROVED:
        print("\033[92m✔ Verification PASSED: Candidate approved.\033[0m")
        return 0
    else:
        print("\033[91m✘ Verification FAILED: Candidate failed to converge to approval.\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(run_verification())
