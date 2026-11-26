"""Triad test suite for multi-agent swarm editorial consensus engine.

Validates positive golden path consensus approval, iterative multi-round
convergence via Pareto remediation directives, negative failure modes
(stagnation, max-rounds exhaustion, fatal defect rejection), and edge cases
(directive conflicts, score boundaries, unhandled agent exceptions).
"""

from typing import Dict, List
import pytest

from shorts_generator.editorial import (
    AgentBallot,
    BaseEditorialAgent,
    ConsensusDecision,
    ConsensusRoundRecord,
    ConsensusStatus,
    CritiqueDirective,
    CutCandidate,
    DirectorAgent,
    EditorialRole,
    EditorialSwarmError,
    EditorialSwarmFactory,
    EditorAgent,
    InvalidCandidateError,
    ParetoArbitrator,
    SoundEngineerAgent,
    SwarmEditorialConsensus,
)


@pytest.fixture
def optimal_candidate() -> CutCandidate:
    """Fixture providing a near-perfect CutCandidate satisfying all editorial rubrics.

    Returns:
        CutCandidate with compliant duration, fast hook, centered face, and EBU R128 audio.
    """
    return CutCandidate(
        clip_id="clip_golden_001",
        title="Epic GTA V Heist Highlight",
        transcript="Wait till you see this impossible escape! The helicopter came out of nowhere.",
        duration_sec=35.0,
        hook_duration_sec=2.2,
        words_per_minute=165.0,
        face_centering_iou=0.96,
        cut_frequency_sec=3.0,
        subtitle_sync_drift_ms=25.0,
        integrated_lufs=-14.0,
        true_peak_dbfs=-1.8,
        ducking_attenuation_db=14.0,
        curiosity_loop_count=2,
        dynamic_zoom_count=4,
    )


@pytest.fixture
def suboptimal_candidate() -> CutCandidate:
    """Fixture providing a suboptimal CutCandidate that can be resolved via remediation.

    Returns:
        CutCandidate with slight hook delay, off-center framing, and loud BGM needing ducking.
    """
    return CutCandidate(
        clip_id="clip_suboptimal_002",
        title="Building an Autonomous AI Agent",
        transcript="So before we begin today's tutorial, let's talk about the setup requirements.",
        duration_sec=48.0,
        hook_duration_sec=4.5,
        words_per_minute=140.0,
        face_centering_iou=0.72,
        cut_frequency_sec=5.2,
        subtitle_sync_drift_ms=85.0,
        integrated_lufs=-18.5,
        true_peak_dbfs=-0.2,
        ducking_attenuation_db=7.0,
        curiosity_loop_count=1,
        dynamic_zoom_count=1,
    )


@pytest.fixture
def catastrophic_candidate() -> CutCandidate:
    """Fixture providing a catastrophically defective CutCandidate destined for rejection.

    Returns:
        CutCandidate with severe hook lag, extreme clipping, and completely lost face tracking.
    """
    return CutCandidate(
        clip_id="clip_catastrophic_003",
        title="Unwatchable Muffled Recording",
        transcript="Uhhhh...",
        duration_sec=59.0,
        hook_duration_sec=9.5,
        words_per_minute=85.0,
        face_centering_iou=0.10,
        cut_frequency_sec=7.5,
        subtitle_sync_drift_ms=350.0,
        integrated_lufs=-6.0,
        true_peak_dbfs=2.5,
        ducking_attenuation_db=1.0,
        curiosity_loop_count=0,
        dynamic_zoom_count=0,
    )


class TestPositiveConsensusScenarios:
    """Positive test scenarios validating golden paths and convergence."""

    def test_unanimous_first_round_approval(self, optimal_candidate: CutCandidate) -> None:
        """Verify that a candidate meeting all standards achieves unanimous approval on Round 1.

        Args:
            optimal_candidate: Pristine test cut candidate.
        """
        swarm = SwarmEditorialConsensus(max_rounds=3, consensus_threshold=9.0)
        decision = swarm.deliberate(optimal_candidate)

        assert decision.status == ConsensusStatus.APPROVED
        assert decision.total_rounds == 1
        assert decision.unanimous is True
        assert decision.final_score >= 9.0
        assert len(decision.round_history) == 1

        ballots = decision.round_history[0].ballots
        assert ballots[EditorialRole.DIRECTOR].score >= 9.0
        assert ballots[EditorialRole.EDITOR].score >= 9.0
        assert ballots[EditorialRole.SOUND_ENGINEER].score >= 9.0

    def test_multi_round_remediation_convergence(
        self, suboptimal_candidate: CutCandidate
    ) -> None:
        """Verify that a suboptimal candidate receives remediations and converges to approval in Round 2.

        Args:
            suboptimal_candidate: Repairable cut candidate.
        """
        swarm = SwarmEditorialConsensus(max_rounds=3, consensus_threshold=8.8)
        records: List[ConsensusRoundRecord] = []

        decision = swarm.deliberate(suboptimal_candidate, round_hook=records.append)

        assert decision.status == ConsensusStatus.APPROVED
        assert decision.total_rounds >= 2
        assert decision.final_score >= 8.8
        assert len(records) >= 2
        assert records[1].composite_score > records[0].composite_score

        revised = decision.final_candidate
        assert revised.hook_duration_sec < suboptimal_candidate.hook_duration_sec
        assert revised.face_centering_iou > suboptimal_candidate.face_centering_iou
        assert revised.integrated_lufs > suboptimal_candidate.integrated_lufs

    def test_factory_standard_swarm_creation(self, optimal_candidate: CutCandidate) -> None:
        """Verify that EditorialSwarmFactory generates a functional standard swarm.

        Args:
            optimal_candidate: Pristine test cut candidate.
        """
        swarm = EditorialSwarmFactory.create_standard_swarm(max_rounds=3)
        decision = swarm.deliberate(optimal_candidate)

        assert decision.status == ConsensusStatus.APPROVED
        assert decision.total_rounds == 1


class TestNegativeConsensusScenarios:
    """Negative test scenarios validating rejection, timeouts, and defect handling."""

    def test_fatal_defect_rejection_after_max_rounds(
        self, catastrophic_candidate: CutCandidate
    ) -> None:
        """Verify that a fundamentally broken candidate is rejected after exhausting max rounds.

        Args:
            catastrophic_candidate: Severely defective candidate.
        """
        swarm = SwarmEditorialConsensus(max_rounds=3, consensus_threshold=9.0)
        decision = swarm.deliberate(catastrophic_candidate)

        assert decision.status in (
            ConsensusStatus.REJECTED_MAX_ROUNDS,
            ConsensusStatus.REJECTED_STAGNANT,
        )
        assert decision.unanimous is False
        assert decision.final_score < 7.5

    def test_stagnant_convergence_abort(self) -> None:
        """Verify that a swarm aborts early when score delta between rounds stagnates below epsilon."""

        class NonImprovingAgent(BaseEditorialAgent):
            """Stub agent returning an identical low score with no meaningful directives."""

            @property
            def role(self) -> EditorialRole:
                """Return Director role."""
                return EditorialRole.DIRECTOR

            def critique(self, candidate: CutCandidate) -> AgentBallot:
                """Return static failing ballot."""
                return AgentBallot(
                    role=self.role,
                    score=5.0,
                    confidence=1.0,
                    pass_vote=False,
                    strengths=[],
                    blocking_issues=["Persistent static defect."],
                    directives=[],
                )

        stagnant_swarm = SwarmEditorialConsensus(
            agents=[NonImprovingAgent(), NonImprovingAgent(), NonImprovingAgent()],
            max_rounds=4,
            consensus_threshold=9.0,
            convergence_tolerance=0.10,
        )

        candidate = CutCandidate(
            clip_id="clip_stagnant",
            title="Stagnant Candidate",
            transcript="Static transcript.",
            duration_sec=30.0,
            hook_duration_sec=2.0,
            words_per_minute=150.0,
            face_centering_iou=0.9,
            cut_frequency_sec=3.0,
            subtitle_sync_drift_ms=30.0,
            integrated_lufs=-14.0,
            true_peak_dbfs=-2.0,
            ducking_attenuation_db=12.0,
        )

        decision = stagnant_swarm.deliberate(candidate)
        assert decision.status == ConsensusStatus.REJECTED_STAGNANT
        assert decision.total_rounds == 2

    def test_invalid_candidate_precondition_rejection(self) -> None:
        """Verify that candidate creation with impossible parameters raises InvalidCandidateError."""
        with pytest.raises(InvalidCandidateError):
            CutCandidate(
                clip_id="bad_duration",
                title="Negative duration",
                transcript="Text",
                duration_sec=-5.0,
                hook_duration_sec=2.0,
                words_per_minute=150.0,
                face_centering_iou=0.8,
                cut_frequency_sec=3.0,
                subtitle_sync_drift_ms=20.0,
                integrated_lufs=-14.0,
                true_peak_dbfs=-1.0,
                ducking_attenuation_db=12.0,
            )

        with pytest.raises(InvalidCandidateError):
            CutCandidate(
                clip_id="bad_iou",
                title="Excessive IoU",
                transcript="Text",
                duration_sec=30.0,
                hook_duration_sec=2.0,
                words_per_minute=150.0,
                face_centering_iou=1.5,
                cut_frequency_sec=3.0,
                subtitle_sync_drift_ms=20.0,
                integrated_lufs=-14.0,
                true_peak_dbfs=-1.0,
                ducking_attenuation_db=12.0,
            )


class TestEdgeCaseConsensusScenarios:
    """Edge-case tests covering conflicts, boundary scores, dissenters, and failures."""

    def test_pareto_conflict_resolution(self) -> None:
        """Verify that opposing directives on the same parameter are resolved via weighted median."""
        ballots = {
            EditorialRole.DIRECTOR: AgentBallot(
                role=EditorialRole.DIRECTOR,
                score=7.0,
                confidence=1.0,
                pass_vote=False,
                strengths=[],
                blocking_issues=[],
                directives=[
                    CritiqueDirective(
                        parameter="cut_frequency_sec",
                        adjustment=1.5,
                        rationale="Slow down cuts for dramatic dialogue pause.",
                        priority=1,
                    )
                ],
            ),
            EditorialRole.EDITOR: AgentBallot(
                role=EditorialRole.EDITOR,
                score=7.0,
                confidence=1.0,
                pass_vote=False,
                strengths=[],
                blocking_issues=[],
                directives=[
                    CritiqueDirective(
                        parameter="cut_frequency_sec",
                        adjustment=-1.0,
                        rationale="Speed up visual cadence for retention.",
                        priority=1,
                    )
                ],
            ),
        }

        deltas = ParetoArbitrator.resolve_directives(ballots)
        assert "cut_frequency_sec" in deltas
        resolved_delta = deltas["cut_frequency_sec"]
        assert -1.0 < resolved_delta < 1.5

    def test_single_dissenter_prevents_premature_approval(self) -> None:
        """Verify that two 10/10 scores cannot bypass a severe audio failure from Sound Engineer."""

        class FlawlessVisualAgent(BaseEditorialAgent):
            """Stub agent returning 10.0."""

            def __init__(self, role: EditorialRole) -> None:
                self._role = role

            @property
            def role(self) -> EditorialRole:
                return self._role

            def critique(self, candidate: CutCandidate) -> AgentBallot:
                return AgentBallot(
                    role=self.role,
                    score=10.0,
                    confidence=1.0,
                    pass_vote=True,
                    strengths=["Flawless."],
                    blocking_issues=[],
                    directives=[],
                )

        class DissentingSoundAgent(BaseEditorialAgent):
            """Stub agent returning failing 4.0 score."""

            @property
            def role(self) -> EditorialRole:
                return EditorialRole.SOUND_ENGINEER

            def critique(self, candidate: CutCandidate) -> AgentBallot:
                return AgentBallot(
                    role=self.role,
                    score=4.0,
                    confidence=1.0,
                    pass_vote=False,
                    strengths=[],
                    blocking_issues=["Harsh DAC clipping."],
                    directives=[
                        CritiqueDirective(
                            parameter="true_peak_dbfs",
                            adjustment=-2.0,
                            rationale="Engage limiter.",
                            priority=1,
                        )
                    ],
                )

        swarm = SwarmEditorialConsensus(
            agents=[
                FlawlessVisualAgent(EditorialRole.DIRECTOR),
                FlawlessVisualAgent(EditorialRole.EDITOR),
                DissentingSoundAgent(),
            ],
            max_rounds=2,
            consensus_threshold=9.0,
        )

        candidate = CutCandidate(
            clip_id="clip_dissent",
            title="Dissent Candidate",
            transcript="Dialogue text.",
            duration_sec=30.0,
            hook_duration_sec=2.0,
            words_per_minute=150.0,
            face_centering_iou=0.9,
            cut_frequency_sec=3.0,
            subtitle_sync_drift_ms=30.0,
            integrated_lufs=-14.0,
            true_peak_dbfs=1.5,
            ducking_attenuation_db=12.0,
        )

        decision = swarm.deliberate(candidate)
        assert decision.status != ConsensusStatus.APPROVED
        assert decision.unanimous is False

    def test_agent_exception_isolation_fallback(self) -> None:
        """Verify that an unexpected exception in one agent is trapped without crashing the swarm."""

        class CrashingAgent(BaseEditorialAgent):
            """Stub agent that raises an unhandled runtime error."""

            @property
            def role(self) -> EditorialRole:
                return EditorialRole.EDITOR

            def critique(self, candidate: CutCandidate) -> AgentBallot:
                raise RuntimeError("Simulated internal GPU OOM crash during rendering.")

        swarm = SwarmEditorialConsensus(
            agents=[DirectorAgent(), CrashingAgent(), SoundEngineerAgent()],
            max_rounds=1,
        )

        candidate = CutCandidate(
            clip_id="clip_crash_test",
            title="Crash Test",
            transcript="Dialogue text.",
            duration_sec=35.0,
            hook_duration_sec=2.0,
            words_per_minute=160.0,
            face_centering_iou=0.95,
            cut_frequency_sec=3.0,
            subtitle_sync_drift_ms=25.0,
            integrated_lufs=-14.0,
            true_peak_dbfs=-1.5,
            ducking_attenuation_db=14.0,
        )

        decision = swarm.deliberate(candidate)
        assert decision.status == ConsensusStatus.REJECTED_MAX_ROUNDS
        editor_ballot = decision.round_history[0].ballots[EditorialRole.EDITOR]
        assert editor_ballot.score == 1.0
        assert "Simulated internal GPU OOM crash" in editor_ballot.blocking_issues[0]
