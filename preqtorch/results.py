from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class EncoderResult:
    """Structured output for encoder runs."""
    model: Any
    code_length: float
    history: List[float]
    ema_params: Optional[Any] = None
    beta: Optional[Any] = None
    replay: Optional[Any] = None
