from dataclasses import dataclass, field
from typing import List


@dataclass
class Metric:
    name: str
    type: str
    value: float


@dataclass
class Task:
    name: str
    type: str
    metrics: List[Metric] = field(default_factory=list)


@dataclass
class Evaluation:
    results: List[Task] = field(default_factory=list)
