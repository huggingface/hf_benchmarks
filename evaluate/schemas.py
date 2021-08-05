from typing import List, Optional, TypedDict, Union


class Metric(TypedDict):
    name: str
    type: str
    value: Union[float, Optional[dict]]


class Task(TypedDict):
    name: str
    type: str
    metrics: List[Metric]


class Result(TypedDict):
    task: Task


class Evaluation(TypedDict):
    results: List[Result]
