"""Immutable public contracts for the unified 3 x 2 benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

PipelineName = Literal["naive", "two-stage", "full"]
GeneratorName = Literal["cloud", "medswin"]

PIPELINES: tuple[PipelineName, ...] = ("naive", "two-stage", "full")
GENERATORS: tuple[GeneratorName, ...] = ("cloud", "medswin")
CONTRACT_VERSION = "2.0"
METRICS_PROTOCOL_VERSION = "1.0"
SAMPLING_ALGORITHM_VERSION = "bioasq13b-stratified-sha256-v1"


@dataclass(frozen=True)
class PipelineContract:
    name: PipelineName
    query_normalization: bool
    hybrid_retrieval: bool
    reranker: bool
    mac: bool
    gate: bool
    retrieve_more: bool


PIPELINE_CONTRACTS = {
    "naive": PipelineContract("naive", False, False, False, False, False, False),
    "two-stage": PipelineContract("two-stage", True, True, True, False, True, True),
    "full": PipelineContract("full", True, True, True, True, True, True),
}


@dataclass(frozen=True)
class GeneratorContract:
    name: GeneratorName
    backend: str
    model: str
    cloud_mode: bool = True
    embedding_model: str = "embed-v-4-0"
    reranker_model: str = "Cohere-rerank-v4.0-fast"


GENERATOR_CONTRACTS = {
    "cloud": GeneratorContract("cloud", "foundry", "gpt-5.4"),
    "medswin": GeneratorContract("medswin", "medswin_local", "MedSwin/MedSwin-DaRE-TIES-KD-0.7"),
}


@dataclass(frozen=True)
class MatrixCell:
    pipeline: PipelineName
    generator: GeneratorName

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_pipelines(value: str) -> tuple[PipelineName, ...]:
    if value == "all":
        return PIPELINES
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = set(selected) - set(PIPELINES)
    if invalid or not selected:
        raise ValueError("--pipeline must be naive, two-stage, full, or all")
    return selected  # type: ignore[return-value]


def parse_generators(value: str) -> tuple[GeneratorName, ...]:
    if value == "all":
        return GENERATORS
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = set(selected) - set(GENERATORS)
    if invalid or not selected:
        raise ValueError("--generator must be cloud, medswin, or all")
    return selected  # type: ignore[return-value]


def expand_matrix(
    pipelines: Iterable[PipelineName] = PIPELINES,
    generators: Iterable[GeneratorName] = GENERATORS,
    *,
    custom: str | None = None,
) -> tuple[MatrixCell, ...]:
    if custom is not None:
        if custom != "combination-september":
            raise ValueError("--custom supports only combination-september")
        pipelines = ("naive", "two-stage")
        generators = GENERATORS
    return tuple(MatrixCell(pipeline, generator) for generator in generators for pipeline in pipelines)


def pipeline_contract_payload() -> dict[str, dict[str, object]]:
    return {name: asdict(contract) for name, contract in PIPELINE_CONTRACTS.items()}


LOCAL_EMBEDDING_MODEL = "MedEmbed-large-v0.1"
LOCAL_RERANKER_MODEL = "bge-reranker-v2-m3"
LOCAL_GENERATOR_MODEL = "MedSwin/MedSwin-DaRE-TIES-KD-0.7"
BEDROCK_GENERATOR_MODEL = "global.openai.gpt-6-luna"
BEDROCK_EMBEDDING_MODEL = "global.cohere.embed-v4:0"
BEDROCK_RERANKER_MODEL = "cohere.rerank-v3-5:0"


def resolve_generator_contract(generator: str, *, local_retrieval: bool = False) -> GeneratorContract:
    """Return the generator contract for the active cloud provider.

    ``local_retrieval=True`` selects the full local three-model stack:
    MedEmbed, bge-reranker-v2-m3, and MedSwin. The static
    ``GENERATOR_CONTRACTS`` map stays on the Foundry defaults.
    """
    if generator not in GENERATOR_CONTRACTS:
        raise KeyError(f"Unknown generator {generator}")
    base = GENERATOR_CONTRACTS[generator]
    if local_retrieval:
        return GeneratorContract(
            name=base.name,
            backend="medswin_local",
            model=LOCAL_GENERATOR_MODEL,
            cloud_mode=False,
            embedding_model=LOCAL_EMBEDDING_MODEL,
            reranker_model=LOCAL_RERANKER_MODEL,
        )
    from app.core.config import settings

    if settings.cloud_provider() == "bedrock":
        if generator == "cloud":
            return GeneratorContract(
                "cloud",
                "bedrock",
                BEDROCK_GENERATOR_MODEL,
                cloud_mode=True,
                embedding_model=BEDROCK_EMBEDDING_MODEL,
                reranker_model=BEDROCK_RERANKER_MODEL,
            )
        return GeneratorContract(
            "medswin",
            "medswin_local",
            LOCAL_GENERATOR_MODEL,
            cloud_mode=True,
            embedding_model=BEDROCK_EMBEDDING_MODEL,
            reranker_model=BEDROCK_RERANKER_MODEL,
        )
    return base
