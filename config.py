# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:46:46 2023

@author: admin
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DefaultParam:
    round_decimal: int = field(default=3)
    sampling_frequency: float = field(default=4096)
    channel_head_count: int = field(default=9)
    init_channel_count: int = field(default=9)
    default_colors: List[str] = field(default_factory=lambda: [
        "black", "red", "green", "blue", "yellow", "orange", "pink",
        "purple", "brown", "gray", "lightgray", "darkgray", "cyan", "magenta",
        "lime", "olive", "teal", "maroon", "navy", "silver", "gold", "violet"
    ])


config = DefaultParam()
