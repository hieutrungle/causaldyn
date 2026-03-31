from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pacing_controller import PIDPacingController
    from .threshold_dispatcher import ThresholdDispatcher

__all__ = ["ThresholdDispatcher", "PIDPacingController"]


def __getattr__(name: str) -> Any:
    if name == "ThresholdDispatcher":
        from .threshold_dispatcher import ThresholdDispatcher

        return ThresholdDispatcher
    if name == "PIDPacingController":
        from .pacing_controller import PIDPacingController

        return PIDPacingController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

