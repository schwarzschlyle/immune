from immune.organs.base import Organ, OrganSet
from immune.organs.organs import AgentOrgan, BusinessOrgan, CareOrgan, CodingOrgan, PipelineOrgan
from immune.reflexes import ReflexSuite
from immune.state.backend import StateBackend


def default_organs(reflexes: ReflexSuite, backend: StateBackend | None = None) -> OrganSet:
    return OrganSet(
        [
            AgentOrgan(reflexes, backend),
            CodingOrgan(reflexes),
            PipelineOrgan(),
            BusinessOrgan(reflexes),
            CareOrgan(),
        ]
    )


__all__ = [
    "AgentOrgan",
    "BusinessOrgan",
    "CareOrgan",
    "CodingOrgan",
    "Organ",
    "OrganSet",
    "PipelineOrgan",
    "default_organs",
]
