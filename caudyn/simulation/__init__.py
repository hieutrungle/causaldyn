from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["MarketplaceSimulation"]

if TYPE_CHECKING:
    from .system_orchestrator import MarketplaceSimulation


def __getattr__(name: str) -> Any:
    if name == "MarketplaceSimulation":
        from .system_orchestrator import MarketplaceSimulation

        return MarketplaceSimulation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
