from dataclasses import dataclass, field, asdict


@dataclass
class Event:
    sim_hour: float
    kind: str          # observation | prediction | alert | event | control
    message: str
    data: dict = field(default_factory=dict)
    def to_dict(self): return asdict(self)
