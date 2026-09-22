"""Metadata for one independently versioned DecisionBench task."""

# Keep the canonical task catalog one row per task so it remains directly auditable.
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from decision_bench.data import DatasetSpec, decode_storage_row, load_rows
from decision_bench.schemas import DecisionExample, Primitive


class TaskMetadata(BaseModel):
    """Describe one task, its taxonomy, and its pinned dataset.

    Like MTEB's ``TaskMetadata``, this metadata belongs to the task rather than
    to a benchmark. Many tasks may point at the same dataset revision, but no
    benchmark owns or republishes their rows.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    dataset: DatasetSpec
    license: str = Field(min_length=1)
    languages: tuple[str, ...] = Field(min_length=1)
    primitive: Primitive
    family: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    source_task_name: str | None = None
    reference: str | None = None
    citation: str | None = None
    is_public: bool = True
    superseded_by: str | None = None

class DecisionTask:
    """One independently versioned dataset-backed evaluation task.

    Contributors normally subclass this class, declare ``metadata``, and only
    override :meth:`dataset_transform` when their source dataset does not
    already use the :class:`DecisionExample` schema.
    """

    metadata: TaskMetadata

    def dataset_transform(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        """Transform one source row into the normalized DecisionExample shape."""

        return decode_storage_row(dict(row))

    def includes(self, example: DecisionExample) -> bool:
        """Return whether a normalized row belongs to this task."""

        return (
            example.task_name == (self.metadata.source_task_name or self.metadata.name)
            and example.primitive is self.metadata.primitive
            and example.family == self.metadata.family
            and example.domain == self.metadata.domain
        )

    def load_data(
        self,
        *,
        project_root: Path | None = None,
        rows: Iterable[Mapping[str, Any] | object] | None = None,
    ) -> tuple[DecisionExample, ...]:
        """Load, transform, filter, and validate this task's evaluation rows."""

        source_rows = (
            load_rows(self.metadata.dataset, project_root=project_root)
            if rows is None
            else rows
        )
        examples: list[DecisionExample] = []
        for row in source_rows:
            if not isinstance(row, Mapping):
                raise TypeError(f"dataset row is not a mapping: {type(row)!r}")
            transformed = self.dataset_transform(row)
            example = DecisionExample.model_validate(transformed)
            if self.includes(example):
                examples.append(example)
        if not examples:
            raise ValueError(f"task {self.metadata.name!r} selected no rows")
        return tuple(examples)


TaskFactory = Callable[[], DecisionTask]
_TASK_REGISTRY: dict[str, TaskFactory] = {}


def register_task[TaskType: type[DecisionTask]](task_type: TaskType) -> TaskType:
    """Register a DecisionTask subclass by its metadata name."""

    metadata = task_type.metadata
    if metadata.name in _TASK_REGISTRY:
        raise ValueError(f"task is already registered: {metadata.name}")
    _TASK_REGISTRY[metadata.name] = task_type
    return task_type


def get_task(name: str) -> DecisionTask:
    """Return one freshly initialized registered task."""

    try:
        factory = _TASK_REGISTRY[name]
    except KeyError as error:
        available = ", ".join(sorted(_TASK_REGISTRY))
        raise KeyError(f"unknown task {name!r}; available tasks: {available}") from error
    return factory()


def get_tasks(
    task_names: Sequence[str] | None = None,
    *,
    languages: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    primitives: Sequence[Primitive | str] | None = None,
    exclude_superseded: bool = True,
) -> list[DecisionTask]:
    """Return registered tasks selected by metadata, following MTEB's shape."""

    if task_names is not None:
        return [get_task(name) for name in task_names]

    normalized_primitives = (
        {Primitive(primitive) for primitive in primitives} if primitives is not None else None
    )
    selected: list[DecisionTask] = []
    for name in sorted(_TASK_REGISTRY):
        task = get_task(name)
        metadata = task.metadata
        if exclude_superseded and metadata.superseded_by is not None:
            continue
        if languages is not None and not set(languages).intersection(metadata.languages):
            continue
        if families is not None and metadata.family not in families:
            continue
        if domains is not None and metadata.domain not in domains:
            continue
        if normalized_primitives is not None and metadata.primitive not in normalized_primitives:
            continue
        selected.append(task)
    return selected


class _CanonicalDecisionBenchTask(DecisionTask):
    def __init__(self, metadata: TaskMetadata) -> None:
        self.metadata = metadata


_CANONICAL_DATASET = DatasetSpec(
    path="Hanno-Labs/decision-bench",
    revision="b7c8107e01ecb1aee7c7eaf5caee4a3ba9f59443",
    split="eval",
)

# Public task name, stored task_name, family, domain, primitive. The existing
# consolidated dataset stays intact; each task owns the same pinned dataset
# reference and its exact row selector.
_CANONICAL_TASKS: tuple[tuple[str, str, str, str, Primitive], ...] = (
    ("AgentCallRetention", "agent-call-retention", "context_management", "coding_agents", Primitive.BINARY_CLASSIFICATION),
    ("AgentReadinessAssessment", "agent-readiness-assessment", "completion_assessment", "coding_agents", Primitive.CANDIDATE_SELECTION),
    ("AgentResultRetention", "agent-result-retention", "context_management", "coding_agents", Primitive.BINARY_CLASSIFICATION),
    ("AgentSkillRanking", "agent-skill-ranking", "routing", "agent_control", Primitive.CANDIDATE_SELECTION),
    ("BoundedValue", "bounded_value", "bounded_extraction", "privacy", Primitive.CANDIDATE_SELECTION),
    ("BrowserActionSelection", "browser-action-selection", "action_selection", "browser_automation", Primitive.CANDIDATE_SELECTION),
    ("BrowserTargetSelection", "browser-target-selection", "target_selection", "browser_automation", Primitive.CANDIDATE_SELECTION),
    ("CanonicalEntity", "canonical_entity", "entity_alignment", "public_sector", Primitive.CANDIDATE_SELECTION),
    ("ClaimEvidenceVerification", "claim-evidence-verification", "verification", "evidence_reasoning", Primitive.CANDIDATE_SELECTION),
    ("CodeChangeRisk", "code-change-risk", "review_scoring", "software_engineering", Primitive.ORDINAL_SCORING),
    ("CodebaseResultRanking", "codebase-result-ranking", "retrieval_ranking", "code_search", Primitive.CANDIDATE_SELECTION),
    ("ContainsThreat", "contains_threat", "guardrails_moderation", "online_safety", Primitive.BINARY_CLASSIFICATION),
    ("DatabaseRowClassification", "database-row-classification", "record_classification", "databases", Primitive.CANDIDATE_SELECTION),
    ("DomAdDetection", "dom-ad-detection", "content_moderation", "web_content", Primitive.BINARY_CLASSIFICATION),
    ("DroneTacticalAction", "drone-tactical-action", "action_selection", "robotics", Primitive.CANDIDATE_SELECTION),
    ("FinQANumericalReasoning", "finqa-numerical-reasoning", "reasoning", "finance", Primitive.ORDINAL_SCORING),
    ("FolioLogicalInference", "folio-logical-inference", "reasoning", "logic", Primitive.CANDIDATE_SELECTION),
    ("GameGoalSelection", "game-goal-selection", "action_selection", "games", Primitive.CANDIDATE_SELECTION),
    ("GraphEdgeSelection", "graph-edge-selection", "navigation", "knowledge_graphs", Primitive.CANDIDATE_SELECTION),
    ("HomeAlertTriage", "home-alert-triage", "triage", "home_automation", Primitive.CANDIDATE_SELECTION),
    ("MuSiQueEvidenceSufficiency", "musique-evidence-sufficiency", "reasoning", "multi_hop_search", Primitive.BINARY_CLASSIFICATION),
    ("MuSiQueMultihopAnswer", "musique-multihop-answer", "reasoning", "multi_hop_search", Primitive.CANDIDATE_SELECTION),
    ("PatentSection", "patent_section", "document_record_classification", "technical", Primitive.CANDIDATE_SELECTION),
    ("PlatformerControlSelection", "platformer-control-selection", "action_selection", "games", Primitive.CANDIDATE_SELECTION),
    ("Relevance", "relevance", "retrieval_verification", "web", Primitive.BINARY_CLASSIFICATION),
    ("RelevanceScore", "relevance_score", "rubric_scoring_prioritization", "ecommerce", Primitive.CANDIDATE_SELECTION),
    ("RobotSkillSelection", "robot-skill-selection", "action_selection", "robotics", Primitive.CANDIDATE_SELECTION),
    ("RouteFinancial", "route", "routing_triage", "financial", Primitive.CANDIDATE_SELECTION),
    ("RouteGeneralAssistant", "route", "routing_triage", "general_assistant", Primitive.CANDIDATE_SELECTION),
    ("RuntimeOutcomeVerification", "runtime-outcome-verification", "verification", "agent_control", Primitive.CANDIDATE_SELECTION),
    ("SearchPlanRouting", "search-plan-routing", "routing", "search", Primitive.CANDIDATE_SELECTION),
    ("SemanticLineMatch", "semantic-line-match", "semantic_filtering", "code_search", Primitive.BINARY_CLASSIFICATION),
    ("SocialPostFiltering", "social-post-filtering", "content_classification", "social_media", Primitive.CANDIDATE_SELECTION),
    ("StrategyCommandSelection", "strategy-command-selection", "action_selection", "games", Primitive.CANDIDATE_SELECTION),
    ("TaxDocumentPageClassification", "tax-document-page-classification", "document_classification", "tax_documents", Primitive.CANDIDATE_SELECTION),
    ("ToolActionImpact", "tool-action-impact", "risk_scoring", "agent_control", Primitive.ORDINAL_SCORING),
    ("ToolCallRouting", "tool-call-routing", "routing", "agent_control", Primitive.CANDIDATE_SELECTION),
    ("ToolSafetyGate", "tool-safety-gate", "safety_gating", "agent_control", Primitive.BINARY_CLASSIFICATION),
    ("ToolRoute", "tool_route", "function_agent_skill_routing", "software_agents", Primitive.CANDIDATE_SELECTION),
    ("ToxicitySeverity", "toxicity_severity", "guardrails_moderation", "online_safety", Primitive.ORDINAL_SCORING),
    ("TradingActionSelection", "trading-action-selection", "action_selection", "finance", Primitive.CANDIDATE_SELECTION),
    ("WorkflowDecisionBoolean", "workflow_decision", "document_workflows", "legal", Primitive.BINARY_CLASSIFICATION),
    ("WorkflowDecisionChoice", "workflow_decision", "document_workflows", "legal", Primitive.CANDIDATE_SELECTION),
)


def _canonical_task_factory(metadata: TaskMetadata) -> TaskFactory:
    def factory() -> DecisionTask:
        return _CanonicalDecisionBenchTask(metadata)

    return factory


def _register_canonical_tasks() -> None:
    for name, source_task_name, family, domain, primitive in _CANONICAL_TASKS:
        metadata = TaskMetadata(
            name=name,
            description=f"{source_task_name} decision task from DecisionBench 1.0.",
            dataset=_CANONICAL_DATASET,
            license="multiple",
            languages=("eng-Latn",),
            primitive=primitive,
            family=family,
            domain=domain,
            source_task_name=source_task_name,
            reference="https://huggingface.co/datasets/Hanno-Labs/decision-bench",
        )
        _TASK_REGISTRY[name] = _canonical_task_factory(metadata)


_register_canonical_tasks()

BUILTIN_TASK_NAMES = tuple(name for name, *_ in _CANONICAL_TASKS)
