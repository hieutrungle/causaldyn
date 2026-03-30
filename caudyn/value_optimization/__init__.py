from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .lp_solver import ValueOptimizer
    from .unit_selector import UnitSelector

__all__ = ["UnitSelector", "ValueOptimizer"]


def __getattr__(name: str) -> Any:
    if name == "UnitSelector":
        from .unit_selector import UnitSelector

        return UnitSelector
    if name == "ValueOptimizer":
        from .lp_solver import ValueOptimizer

        return ValueOptimizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
