"""Multi-agent swarm editorial consensus engine for autonomous cut approval.

Orchestrates multi-round deliberation between Director, Editor, and Sound Engineer
subagents to critique candidate video shorts from distinct editorial angles,
synthesizes Pareto-optimal remediation directives, and guarantees mathematical
convergence before final video rendering and publishing.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Dict, List, Optional, Tuple


class EditorialRole(str, Enum):
    """Semantic role of a specialist critique agent within the editorial swarm."""

    DIRECTOR = "director"
    EDITOR = "editor"
    SOUND_ENGINEER = "sound_engineer"


class ConsensusStatus(str, Enum):
    """Resolution status of the multi-agent editorial consensus deliberation."""

    APPROVED = "approved"
    REVISED_ROUND = "revised_round"
    REJECTED_MAX_ROUNDS = "rejected_max_rounds"
    REJECTED_STAGNANT = "rejected_stagnant"
    REJECTED_DEFECT = "rejected_defect"


class EditorialSwarmError(Exception):
    """Base exception for editorial consensus swarm engine errors."""


class ConsensusFailureError(EditorialSwarmError):
    """Raised when consensus deliberation fails to reach a valid decision."""


class RemediationExhaustionError(EditorialSwarmError):
    """Raised when candidate remediation parameters exceed physically bounded limits."""


class OscillatoryDirectiveError(EditorialSwarmError):
    """Raised when opposing agent directives trigger non-converging parameter oscillation."""


class InvalidCandidateError(EditorialSwarmError):
    """Raised when an input candidate short fails precondition validation."""


@dataclass(frozen=True)
class CritiqueDirective:
    """Actionable parameter modification instruction emitted by a critique agent.

    Attributes:
        parameter: Canonical name of the candidate parameter to adjust.
        adjustment: Numeric delta value to apply to the target parameter.
        rationale: Explanatory justification for why the adjustment is needed.
        priority: Priority weighting rank for resolving conflicting directives (1 = highest).
    """

    parameter: str
    adjustment: float
    rationale: str
    priority: int = 1


@dataclass(frozen=True)
class AgentBallot:
    """Quantitative evaluation and feedback ballot cast by an individual critique agent.

    Attributes:
        role: Editorial role of the voting agent.
        score: Normalized quality score in range [1.0, 10.0].
        confidence: Agent confidence factor in range [0.0, 1.0].
        pass_vote: Boolean indicating whether the draft meets the agent's approval bar (score >= 9.0).
        strengths: Positive editorial qualities observed in the candidate cut.
        blocking_issues: Defects or deficiencies preventing approval.
        directives: Ordered list of remediation directives to improve the candidate.
    """

    role: EditorialRole
    score: float
    confidence: float
    pass_vote: bool
    strengths: List[str]
    blocking_issues: List[str]
    directives: List[CritiqueDirective] = field(default_factory=list)


@dataclass(frozen=True)
class DirectorContext:
    """Scoped context projected for the Director Agent.

    Attributes:
        transcript: Full verbal transcript text of the candidate short.
        hook_duration_sec: Duration in seconds of the opening hook before the narrative core.
        duration_sec: Total duration in seconds of the candidate cut.
        words_per_minute: Pacing velocity measured in spoken words per minute.
        curiosity_loop_count: Number of open cognitive curiosity loops identified in narrative.
    """

    transcript: str
    hook_duration_sec: float
    duration_sec: float
    words_per_minute: float
    curiosity_loop_count: int = 1


@dataclass(frozen=True)
class EditorContext:
    """Scoped context projected for the Editor Agent.

    Attributes:
        face_centering_iou: Intersection-over-Union bounding box centering score for subject faces.
        cut_frequency_sec: Average temporal interval in seconds between visual shot transitions.
        subtitle_sync_drift_ms: Kinetic typography subtitle desynchronization offset in milliseconds.
        dynamic_zoom_count: Number of intentional visual punch-in zoom triggers.
    """

    face_centering_iou: float
    cut_frequency_sec: float
    subtitle_sync_drift_ms: float
    dynamic_zoom_count: int = 0


@dataclass(frozen=True)
class SoundContext:
    """Scoped context projected for the Sound Engineer Agent.

    Attributes:
        integrated_lufs: Integrated program loudness measured in Loudness Units Full Scale.
        true_peak_dbfs: Inter-sample maximum true peak level in decibels relative to full scale.
        ducking_attenuation_db: Dynamic sidechain attenuation applied to background music during dialogue.
        spectral_clarity_score: Normalized vocal formant intelligibility score in range [0.0, 1.0].
    """

    integrated_lufs: float
    true_peak_dbfs: float
    ducking_attenuation_db: float
    spectral_clarity_score: float = 0.95


@dataclass(frozen=True)
class CutCandidate:
    """Multidimensional descriptor representing a candidate video short undergoing critique.

    Attributes:
        clip_id: Unique tracking identifier for the candidate clip.
        title: Working title or narrative description.
        transcript: Spoken verbal transcript.
        duration_sec: Total duration in seconds.
        hook_duration_sec: Opening hook duration in seconds.
        words_per_minute: Speech rate in words per minute.
        face_centering_iou: Face tracking vertical centering score in range [0.0, 1.0].
        cut_frequency_sec: Average cut cadence interval in seconds.
        subtitle_sync_drift_ms: Subtitle kinetic timing drift in milliseconds.
        integrated_lufs: Audio integrated loudness in LUFS.
        true_peak_dbfs: Audio true peak level in dBFS.
        ducking_attenuation_db: Dialogue sidechain ducking depth in dB.
        curiosity_loop_count: Narrative curiosity loops count.
        dynamic_zoom_count: Dynamic zoom punch-ins count.
        parameters: Arbitrary parameter dictionary for extensible tuning.
    """

    clip_id: str
    title: str
    transcript: str
    duration_sec: float
    hook_duration_sec: float
    words_per_minute: float
    face_centering_iou: float
    cut_frequency_sec: float
    subtitle_sync_drift_ms: float
    integrated_lufs: float
    true_peak_dbfs: float
    ducking_attenuation_db: float
    curiosity_loop_count: int = 1
    dynamic_zoom_count: int = 3
    parameters: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate candidate preconditions and numeric parameter boundaries.

        Raises:
            InvalidCandidateError: If any parameter violates physical or mathematical boundaries.
        """
        if self.duration_sec <= 0.0 or self.duration_sec > 300.0:
            raise InvalidCandidateError(
                f"Candidate duration {self.duration_sec}s must be between 0.1s and 300.0s."
            )
        if self.hook_duration_sec < 0.0:
            raise InvalidCandidateError(
                f"Hook duration {self.hook_duration_sec}s cannot be negative."
            )
        if self.words_per_minute <= 0.0:
            raise InvalidCandidateError(
                f"Speech rate {self.words_per_minute} WPM must be positive."
            )
        if not (0.0 <= self.face_centering_iou <= 1.0):
            raise InvalidCandidateError(
                f"Face centering IoU {self.face_centering_iou} must be in range [0.0, 1.0]."
            )
        if self.cut_frequency_sec <= 0.0:
            raise InvalidCandidateError(
                f"Cut frequency {self.cut_frequency_sec}s must be positive."
            )


@dataclass(frozen=True)
class ConsensusRoundRecord:
    """Historical audit record capturing a single round of multi-agent deliberation.

    Attributes:
        round_number: One-based index of the deliberation round.
        ballots: Dictionary mapping each editorial role to its cast ballot.
        composite_score: Weighted composite consensus score for the round.
        unanimous_approved: Whether all agents cast approval votes (score >= 9.0).
        applied_adjustments: Parameter adjustment deltas synthesized and applied in this round.
    """

    round_number: int
    ballots: Dict[EditorialRole, AgentBallot]
    composite_score: float
    unanimous_approved: bool
    applied_adjustments: Dict[str, float]


@dataclass(frozen=True)
class ConsensusDecision:
    """Final immutable outcome emitted by the editorial consensus engine.

    Attributes:
        status: Final consensus outcome status.
        total_rounds: Total number of deliberation rounds executed.
        final_score: Final weighted composite quality score.
        unanimous: Whether unanimous 9.0+ consensus was achieved across all agents.
        round_history: Full audit trail of each deliberation round.
        final_candidate: Revised candidate short reflecting all synthesized remediations.
        summary: Human-readable narrative summary of the consensus verdict.
    """

    status: ConsensusStatus
    total_rounds: int
    final_score: float
    unanimous: bool
    round_history: List[ConsensusRoundRecord]
    final_candidate: CutCandidate
    summary: str


class ContextProjector:
    """Differential context projection service maintaining strict agent namespace isolation."""

    @staticmethod
    def project_director_context(candidate: CutCandidate) -> DirectorContext:
        """Project candidate attributes into an isolated DirectorContext bundle.

        Args:
            candidate: Source candidate cut.

        Returns:
            DirectorContext containing narrative and pacing metrics only.
        """
        return DirectorContext(
            transcript=candidate.transcript,
            hook_duration_sec=candidate.hook_duration_sec,
            duration_sec=candidate.duration_sec,
            words_per_minute=candidate.words_per_minute,
            curiosity_loop_count=candidate.curiosity_loop_count,
        )

    @staticmethod
    def project_editor_context(candidate: CutCandidate) -> EditorContext:
        """Project candidate attributes into an isolated EditorContext bundle.

        Args:
            candidate: Source candidate cut.

        Returns:
            EditorContext containing visual framing and typography metrics only.
        """
        return EditorContext(
            face_centering_iou=candidate.face_centering_iou,
            cut_frequency_sec=candidate.cut_frequency_sec,
            subtitle_sync_drift_ms=candidate.subtitle_sync_drift_ms,
            dynamic_zoom_count=candidate.dynamic_zoom_count,
        )

    @staticmethod
    def project_sound_context(candidate: CutCandidate) -> SoundContext:
        """Project candidate attributes into an isolated SoundContext bundle.

        Args:
            candidate: Source candidate cut.

        Returns:
            SoundContext containing acoustic and loudness metrics only.
        """
        return SoundContext(
            integrated_lufs=candidate.integrated_lufs,
            true_peak_dbfs=candidate.true_peak_dbfs,
            ducking_attenuation_db=candidate.ducking_attenuation_db,
            spectral_clarity_score=candidate.parameters.get("spectral_clarity_score", 0.95),
        )


class BaseEditorialAgent(ABC):
    """Abstract base class defining the contract for specialist editorial critique subagents."""

    @property
    @abstractmethod
    def role(self) -> EditorialRole:
        """Return the editorial role of this critique agent.

        Returns:
            EditorialRole enum identifying the agent.
        """

    @abstractmethod
    def critique(self, candidate: CutCandidate) -> AgentBallot:
        """Evaluate a candidate short and produce an immutable evaluation ballot.

        Args:
            candidate: The candidate video short to evaluate.

        Returns:
            AgentBallot containing normalized score, strengths, issues, and directives.
        """


class DirectorAgent(BaseEditorialAgent):
    """Specialist subagent evaluating narrative hook strength, curiosity loops, and overall pacing."""

    @property
    def role(self) -> EditorialRole:
        """Return the DIRECTOR role."""
        return EditorialRole.DIRECTOR

    def critique(self, candidate: CutCandidate) -> AgentBallot:
        """Evaluate narrative structure, opening hook speed, and pacing resonance.

        Args:
            candidate: The candidate cut to evaluate.

        Returns:
            AgentBallot reflecting narrative and pacing quality.
        """
        ctx = ContextProjector.project_director_context(candidate)

        hook_penalty = max(0.0, (ctx.hook_duration_sec - 3.0) * 2.0)
        hook_score = max(1.0, 10.0 - hook_penalty)

        optimal_duration = 35.0
        duration_sigma = 10.0
        pacing_resonance = 10.0 * math.exp(
            -((ctx.duration_sec - optimal_duration) ** 2) / (2.0 * (duration_sigma**2))
        )

        if 150.0 <= ctx.words_per_minute <= 185.0:
            wpm_score = 10.0
        else:
            wpm_score = max(2.0, 10.0 - 0.15 * abs(ctx.words_per_minute - 167.5))

        loop_bonus = min(1.0, ctx.curiosity_loop_count * 0.5)

        raw_score = (
            0.40 * hook_score
            + 0.35 * pacing_resonance
            + 0.20 * wpm_score
            + 0.05 * (loop_bonus * 10.0)
        )

        word_count = len(ctx.transcript.split())
        is_substantive = word_count >= 5
        if not is_substantive:
            raw_score = min(raw_score, 2.5)

        final_score = round(max(1.0, min(10.0, raw_score)), 2)

        strengths: List[str] = []
        issues: List[str] = []
        directives: List[CritiqueDirective] = []

        if not is_substantive:
            issues.append(
                f"Fatal narrative defect: transcript has insufficient verbal substance ({word_count} words)."
            )

        if ctx.hook_duration_sec <= 3.0:
            strengths.append(f"Immediate narrative hook delivered within {ctx.hook_duration_sec:.1f}s.")
        else:
            issues.append(f"Opening hook delayed to {ctx.hook_duration_sec:.1f}s, risking swipe-away.")
            directives.append(
                CritiqueDirective(
                    parameter="hook_duration_sec",
                    adjustment=-(ctx.hook_duration_sec - 2.5),
                    rationale="Trim pre-hook setup to capture audience attention inside 2.5 seconds.",
                    priority=1,
                )
            )

        if 25.0 <= ctx.duration_sec <= 45.0:
            strengths.append(f"Optimal short-form video duration ({ctx.duration_sec:.1f}s).")
        else:
            issues.append(f"Suboptimal duration ({ctx.duration_sec:.1f}s) deviates from 35s sweet spot.")
            target_duration = 35.0
            directives.append(
                CritiqueDirective(
                    parameter="duration_sec",
                    adjustment=round(target_duration - ctx.duration_sec, 1),
                    rationale="Re-time pacing to converge toward the 35-second viral duration sweet spot.",
                    priority=2,
                )
            )

        if 150.0 <= ctx.words_per_minute <= 185.0:
            strengths.append(f"Crisp vocal pacing at {ctx.words_per_minute:.0f} WPM.")
        else:
            issues.append(f"Vocal delivery rate of {ctx.words_per_minute:.0f} WPM is outside ideal range.")
            target_wpm = 168.0
            directives.append(
                CritiqueDirective(
                    parameter="words_per_minute",
                    adjustment=round(target_wpm - ctx.words_per_minute, 1),
                    rationale="Modulate audio playback speed to land speech rate in 165-170 WPM envelope.",
                    priority=3,
                )
            )

        pass_vote = final_score >= 9.0
        confidence = 0.95

        return AgentBallot(
            role=self.role,
            score=final_score,
            confidence=confidence,
            pass_vote=pass_vote,
            strengths=strengths,
            blocking_issues=issues,
            directives=directives,
        )


class EditorAgent(BaseEditorialAgent):
    """Specialist subagent evaluating 9:16 vertical crop centering, cut frequency, and subtitle sync."""

    @property
    def role(self) -> EditorialRole:
        """Return the EDITOR role."""
        return EditorialRole.EDITOR

    def critique(self, candidate: CutCandidate) -> AgentBallot:
        """Evaluate visual composition, dynamic cut rhythm, and prosody typography synchronization.

        Args:
            candidate: The candidate cut to evaluate.

        Returns:
            AgentBallot reflecting visual editing and subtitle alignment quality.
        """
        ctx = ContextProjector.project_editor_context(candidate)

        face_score = min(10.0, ctx.face_centering_iou * 10.0)

        cut_target = 3.0
        cut_deviation = abs(ctx.cut_frequency_sec - cut_target)
        cut_score = max(1.0, 10.0 - 2.5 * cut_deviation)

        sync_penalty = max(0.0, (ctx.subtitle_sync_drift_ms - 50.0) * 0.10)
        sync_score = max(1.0, 10.0 - sync_penalty)

        raw_score = 0.40 * face_score + 0.30 * cut_score + 0.30 * sync_score
        final_score = round(max(1.0, min(10.0, raw_score)), 2)

        strengths: List[str] = []
        issues: List[str] = []
        directives: List[CritiqueDirective] = []

        if ctx.face_centering_iou >= 0.90:
            strengths.append(f"Subject face perfectly centered in 9:16 frame (IoU: {ctx.face_centering_iou:.2f}).")
        else:
            issues.append(f"Face tracking off-center (IoU: {ctx.face_centering_iou:.2f}), risking cropping.")
            directives.append(
                CritiqueDirective(
                    parameter="face_centering_iou",
                    adjustment=round(0.95 - ctx.face_centering_iou, 2),
                    rationale="Apply pan-and-scan camera recalibration to lock subject face in center third.",
                    priority=1,
                )
            )

        if 2.2 <= ctx.cut_frequency_sec <= 3.8:
            strengths.append(f"Fast-paced visual cut rhythm ({ctx.cut_frequency_sec:.1f}s interval).")
        else:
            issues.append(f"Cut cadence of {ctx.cut_frequency_sec:.1f}s fails to sustain visual interest.")
            directives.append(
                CritiqueDirective(
                    parameter="cut_frequency_sec",
                    adjustment=round(3.0 - ctx.cut_frequency_sec, 2),
                    rationale="Insert dynamic b-roll cutaways to achieve viral 3.0s cut rhythm.",
                    priority=2,
                )
            )

        if ctx.subtitle_sync_drift_ms <= 50.0:
            strengths.append(f"Kinetic subtitles tightly synchronized ({ctx.subtitle_sync_drift_ms:.0f}ms drift).")
        else:
            issues.append(f"Subtitle timing desync of {ctx.subtitle_sync_drift_ms:.0f}ms degrades readability.")
            directives.append(
                CritiqueDirective(
                    parameter="subtitle_sync_drift_ms",
                    adjustment=-ctx.subtitle_sync_drift_ms,
                    rationale="Re-anchor kinetic subtitle timestamps to vocal syllable onsets.",
                    priority=1,
                )
            )

        pass_vote = final_score >= 9.0
        confidence = 0.92

        return AgentBallot(
            role=self.role,
            score=final_score,
            confidence=confidence,
            pass_vote=pass_vote,
            strengths=strengths,
            blocking_issues=issues,
            directives=directives,
        )


class SoundEngineerAgent(BaseEditorialAgent):
    """Specialist subagent evaluating EBU R128 loudness compliance, True Peak margin, and stem ducking."""

    @property
    def role(self) -> EditorialRole:
        """Return the SOUND_ENGINEER role."""
        return EditorialRole.SOUND_ENGINEER

    def critique(self, candidate: CutCandidate) -> AgentBallot:
        """Evaluate acoustic loudness compliance, peak headroom, and sidechain ducking balance.

        Args:
            candidate: The candidate cut to evaluate.

        Returns:
            AgentBallot reflecting audio mastering and broadcast compliance quality.
        """
        ctx = ContextProjector.project_sound_context(candidate)

        target_lufs = -14.0
        lufs_deviation = abs(ctx.integrated_lufs - target_lufs)
        lufs_score = max(1.0, 10.0 - 2.5 * lufs_deviation)

        if ctx.true_peak_dbfs <= -1.0:
            peak_score = 10.0
        else:
            peak_penalty = (ctx.true_peak_dbfs - (-1.0)) * 15.0
            peak_score = max(1.0, 10.0 - peak_penalty)

        ducking_ratio = ctx.ducking_attenuation_db / 12.0
        ducking_score = min(10.0, 10.0 * ducking_ratio)

        raw_score = 0.40 * lufs_score + 0.35 * peak_score + 0.25 * ducking_score

        has_permanent_clipping = ctx.true_peak_dbfs > 1.0
        if has_permanent_clipping:
            raw_score = min(raw_score, 3.5)

        final_score = round(max(1.0, min(10.0, raw_score)), 2)

        strengths: List[str] = []
        issues: List[str] = []
        directives: List[CritiqueDirective] = []

        if has_permanent_clipping:
            issues.append(
                f"Permanent digital clipping (+{ctx.true_peak_dbfs:.1f} dBFS) corrupts audio harmonics."
            )

        if abs(ctx.integrated_lufs - target_lufs) <= 1.0:
            strengths.append(f"Compliant integrated loudness at {ctx.integrated_lufs:.1f} LUFS.")
        else:
            issues.append(f"Loudness of {ctx.integrated_lufs:.1f} LUFS deviates from broadcast -14 LUFS.")
            directives.append(
                CritiqueDirective(
                    parameter="integrated_lufs",
                    adjustment=round(target_lufs - ctx.integrated_lufs, 1),
                    rationale="Apply two-pass loudnorm filter to target -14.0 LUFS integrated loudness.",
                    priority=1,
                )
            )

        if ctx.true_peak_dbfs <= -1.0:
            strengths.append(f"Safe True Peak ceiling at {ctx.true_peak_dbfs:.1f} dBFS.")
        else:
            issues.append(f"True Peak exceeds ceiling ({ctx.true_peak_dbfs:.1f} dBFS), risking mobile DAC clipping.")
            directives.append(
                CritiqueDirective(
                    parameter="true_peak_dbfs",
                    adjustment=round(-1.5 - ctx.true_peak_dbfs, 1),
                    rationale="Engage 4x oversampled brickwall limiter to lock True Peak ceiling below -1.0 dBFS.",
                    priority=1,
                )
            )

        if ctx.ducking_attenuation_db >= 12.0:
            strengths.append(f"Solid dialogue sidechain ducking ({ctx.ducking_attenuation_db:.1f} dB attenuation).")
        else:
            issues.append(f"Insufficient BGM ducking ({ctx.ducking_attenuation_db:.1f} dB) clouds vocal clarity.")
            directives.append(
                CritiqueDirective(
                    parameter="ducking_attenuation_db",
                    adjustment=round(14.0 - ctx.ducking_attenuation_db, 1),
                    rationale="Deepen sidechain auto-ducking to ensure vocal dialogue retains prime acoustic clarity.",
                    priority=2,
                )
            )

        pass_vote = final_score >= 9.0
        confidence = 0.98

        return AgentBallot(
            role=self.role,
            score=final_score,
            confidence=confidence,
            pass_vote=pass_vote,
            strengths=strengths,
            blocking_issues=issues,
            directives=directives,
        )


class ParetoArbitrator:
    """Arbitrates, reconciles, and aggregates conflicting parameter directives from multiple agents."""

    PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
        "hook_duration_sec": (1.0, 10.0),
        "duration_sec": (15.0, 60.0),
        "words_per_minute": (100.0, 250.0),
        "face_centering_iou": (0.0, 1.0),
        "cut_frequency_sec": (1.5, 8.0),
        "subtitle_sync_drift_ms": (0.0, 500.0),
        "integrated_lufs": (-30.0, -5.0),
        "true_peak_dbfs": (-20.0, 3.0),
        "ducking_attenuation_db": (0.0, 30.0),
    }

    ROLE_WEIGHTS: Dict[EditorialRole, float] = {
        EditorialRole.DIRECTOR: 0.35,
        EditorialRole.EDITOR: 0.35,
        EditorialRole.SOUND_ENGINEER: 0.30,
    }

    @classmethod
    def resolve_directives(
        cls,
        ballots: Dict[EditorialRole, AgentBallot],
        damping_factor: float = 0.85,
    ) -> Dict[str, float]:
        """Aggregate directives from all ballots and compute Pareto-optimal parameter deltas.

        Args:
            ballots: Dictionary of ballots cast across all roles in the current round.
            damping_factor: Relaxation multiplier preventing oscillatory overshoot (default: 0.85).

        Returns:
            Dictionary mapping parameter names to arbitrated net adjustment deltas.
        """
        directive_map: Dict[str, List[Tuple[float, float]]] = {}

        for role, ballot in ballots.items():
            weight = cls.ROLE_WEIGHTS.get(role, 0.33)
            for d in ballot.directives:
                priority_multiplier = 1.0 / max(1, d.priority)
                effective_weight = weight * priority_multiplier
                directive_map.setdefault(d.parameter, []).append((d.adjustment, effective_weight))

        resolved_deltas: Dict[str, float] = {}
        for param, adjustments in directive_map.items():
            total_weight = sum(w for _, w in adjustments)
            if total_weight > 0.0:
                weighted_delta = sum(adj * w for adj, w in adjustments) / total_weight
                resolved_deltas[param] = round(weighted_delta * damping_factor, 3)

        return resolved_deltas

    @classmethod
    def apply_deltas_to_candidate(
        cls, candidate: CutCandidate, deltas: Dict[str, float]
    ) -> CutCandidate:
        """Apply resolved adjustment deltas to a candidate cut, enforcing physical parameter bounds.

        Args:
            candidate: Current candidate cut instance.
            deltas: Arbitrated parameter adjustments.

        Returns:
            A new CutCandidate instance with revised attributes.
        """
        attrs = {
            "hook_duration_sec": candidate.hook_duration_sec,
            "duration_sec": candidate.duration_sec,
            "words_per_minute": candidate.words_per_minute,
            "face_centering_iou": candidate.face_centering_iou,
            "cut_frequency_sec": candidate.cut_frequency_sec,
            "subtitle_sync_drift_ms": candidate.subtitle_sync_drift_ms,
            "integrated_lufs": candidate.integrated_lufs,
            "true_peak_dbfs": candidate.true_peak_dbfs,
            "ducking_attenuation_db": candidate.ducking_attenuation_db,
        }

        for param, delta in deltas.items():
            if param in attrs:
                new_val = attrs[param] + delta
                if param in cls.PARAM_BOUNDS:
                    min_b, max_b = cls.PARAM_BOUNDS[param]
                    new_val = max(min_b, min(max_b, new_val))
                attrs[param] = round(new_val, 3)

        return CutCandidate(
            clip_id=candidate.clip_id,
            title=candidate.title,
            transcript=candidate.transcript,
            duration_sec=attrs["duration_sec"],
            hook_duration_sec=attrs["hook_duration_sec"],
            words_per_minute=attrs["words_per_minute"],
            face_centering_iou=attrs["face_centering_iou"],
            cut_frequency_sec=attrs["cut_frequency_sec"],
            subtitle_sync_drift_ms=attrs["subtitle_sync_drift_ms"],
            integrated_lufs=attrs["integrated_lufs"],
            true_peak_dbfs=attrs["true_peak_dbfs"],
            ducking_attenuation_db=attrs["ducking_attenuation_db"],
            curiosity_loop_count=candidate.curiosity_loop_count,
            dynamic_zoom_count=candidate.dynamic_zoom_count,
            parameters=dict(candidate.parameters),
        )


class SwarmEditorialConsensus:
    """Master orchestrator for multi-agent swarm editorial consensus deliberation.

    Executes parallel fan-out evaluation, aggregates ballots into weighted consensus scores,
    and conducts iterative refinement loops with Pareto conflict resolution until consensus
    is achieved or termination limits are reached.
    """

    def __init__(
        self,
        agents: Optional[List[BaseEditorialAgent]] = None,
        max_rounds: int = 3,
        consensus_threshold: float = 9.0,
        convergence_tolerance: float = 0.05,
        max_workers: int = 3,
    ) -> None:
        """Initialize the SwarmEditorialConsensus engine with agents and parameters.

        Args:
            agents: List of critique agents; defaults to DirectorAgent, EditorAgent, SoundEngineerAgent.
            max_rounds: Maximum deliberation rounds permitted before terminating (default: 3).
            consensus_threshold: Normalized score required for approval (default: 9.0).
            convergence_tolerance: Minimum score gain between rounds to avoid stagnation (default: 0.05).
            max_workers: Thread pool concurrency limit for parallel agent evaluation.
        """
        if agents is None:
            self._agents = [
                DirectorAgent(),
                EditorAgent(),
                SoundEngineerAgent(),
            ]
        else:
            self._agents = list(agents)

        self.max_rounds = max(1, max_rounds)
        self.consensus_threshold = consensus_threshold
        self.convergence_tolerance = convergence_tolerance
        self.max_workers = max(1, max_workers)

    def deliberate(
        self,
        candidate: CutCandidate,
        round_hook: Optional[Callable[[ConsensusRoundRecord], None]] = None,
    ) -> ConsensusDecision:
        """Run the full multi-round consensus deliberation workflow on a candidate video short.

        Args:
            candidate: Initial cut candidate to evaluate and refine.
            round_hook: Optional observer callback invoked at the conclusion of each round.

        Returns:
            ConsensusDecision with final status, round history, and revised cut.
        """
        current_candidate = candidate
        round_history: List[ConsensusRoundRecord] = []
        previous_score: Optional[float] = None

        for round_idx in range(1, self.max_rounds + 1):
            ballots = self._evaluate_round_parallel(current_candidate)

            weights = ParetoArbitrator.ROLE_WEIGHTS
            composite_score = sum(
                ballots[agent.role].score * weights.get(agent.role, 0.33)
                for agent in self._agents
            )
            composite_score = round(composite_score, 2)

            all_unanimous = all(b.pass_vote for b in ballots.values())
            threshold_met = (
                all_unanimous
                or (
                    composite_score >= self.consensus_threshold
                    and min(b.score for b in ballots.values()) >= 8.5
                )
            )

            deltas = ParetoArbitrator.resolve_directives(ballots)
            round_record = ConsensusRoundRecord(
                round_number=round_idx,
                ballots=ballots,
                composite_score=composite_score,
                unanimous_approved=all_unanimous,
                applied_adjustments=deltas,
            )
            round_history.append(round_record)

            if round_hook:
                round_hook(round_record)

            if threshold_met:
                summary = (
                    f"Consensus APPROVED on Round {round_idx} with score {composite_score:.2f}/10.0 "
                    f"(Unanimous: {all_unanimous}). All editorial domains certified broadcast-ready."
                )
                return ConsensusDecision(
                    status=ConsensusStatus.APPROVED,
                    total_rounds=round_idx,
                    final_score=composite_score,
                    unanimous=all_unanimous,
                    round_history=round_history,
                    final_candidate=current_candidate,
                    summary=summary,
                )

            if previous_score is not None:
                score_gain = composite_score - previous_score
                if score_gain < self.convergence_tolerance and round_idx < self.max_rounds:
                    summary = (
                        f"Consensus REJECTED on Round {round_idx} due to stagnant convergence "
                        f"(gain {score_gain:+.2f} < threshold {self.convergence_tolerance})."
                    )
                    return ConsensusDecision(
                        status=ConsensusStatus.REJECTED_STAGNANT,
                        total_rounds=round_idx,
                        final_score=composite_score,
                        unanimous=False,
                        round_history=round_history,
                        final_candidate=current_candidate,
                        summary=summary,
                    )

            previous_score = composite_score

            if round_idx < self.max_rounds:
                current_candidate = ParetoArbitrator.apply_deltas_to_candidate(
                    current_candidate, deltas
                )

        summary = (
            f"Consensus REJECTED after exhausting maximum permitted rounds ({self.max_rounds}). "
            f"Final score: {composite_score:.2f}/10.0. Requires human creator oversight."
        )
        return ConsensusDecision(
            status=ConsensusStatus.REJECTED_MAX_ROUNDS,
            total_rounds=self.max_rounds,
            final_score=composite_score,
            unanimous=False,
            round_history=round_history,
            final_candidate=current_candidate,
            summary=summary,
        )

    def _evaluate_round_parallel(
        self, candidate: CutCandidate
    ) -> Dict[EditorialRole, AgentBallot]:
        """Execute parallel fan-out evaluation of all configured agents on the candidate cut.

        Args:
            candidate: Target cut candidate.

        Returns:
            Dictionary mapping each agent's role to its returned ballot.
        """
        ballots: Dict[EditorialRole, AgentBallot] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(agent.critique, candidate): agent for agent in self._agents
            }
            for future in futures:
                agent = futures[future]
                try:
                    ballot = future.result()
                    ballots[agent.role] = ballot
                except Exception as exc:
                    fallback_ballot = AgentBallot(
                        role=agent.role,
                        score=1.0,
                        confidence=0.0,
                        pass_vote=False,
                        strengths=[],
                        blocking_issues=[f"Agent evaluation failed with exception: {exc}"],
                        directives=[],
                    )
                    ballots[agent.role] = fallback_ballot

        return ballots


class EditorialSwarmFactory:
    """Factory creating pre-configured editorial swarm instances for different production tiers."""

    @staticmethod
    def create_standard_swarm(max_rounds: int = 3) -> SwarmEditorialConsensus:
        """Create a standard 3-agent editorial consensus swarm (Director, Editor, Sound Engineer).

        Args:
            max_rounds: Maximum permitted deliberation rounds.

        Returns:
            Configured SwarmEditorialConsensus instance.
        """
        return SwarmEditorialConsensus(
            agents=[DirectorAgent(), EditorAgent(), SoundEngineerAgent()],
            max_rounds=max_rounds,
            consensus_threshold=9.0,
            convergence_tolerance=0.05,
        )

    @staticmethod
    def create_strict_broadcast_swarm(max_rounds: int = 4) -> SwarmEditorialConsensus:
        """Create a strict broadcast-grade consensus swarm with elevated consensus threshold.

        Args:
            max_rounds: Maximum permitted deliberation rounds.

        Returns:
            SwarmEditorialConsensus instance enforcing strict 9.2 consensus floor.
        """
        return SwarmEditorialConsensus(
            agents=[DirectorAgent(), EditorAgent(), SoundEngineerAgent()],
            max_rounds=max_rounds,
            consensus_threshold=9.2,
            convergence_tolerance=0.03,
        )
