
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    round_decimal: int = field(default=3)
    sampling_frequency: float = field(default=32)
    channel_head_count: int = field(default=10)
    init_channel_count: int = field(default=10)
    default_colors: List[str] = field(default_factory=lambda: [
        "red", "green", "blue", "orange",
        "purple", "brown", "gray", "cyan", "magenta",
        "lime", "olive", "teal", "maroon", "navy", "silver", "gold", "violet", "yellow" "pink"
    ])

