"""Compatibility facade for LinUCB OCO components.

This module keeps historical import paths stable while the implementation is
split into agent definitions and experiment runners.
"""

from .linucb_agents import FastLinUCBAgent, LinUCBAgent
from .experiments import (
    extract_rlearner_linear_priors_from_csv,
    run_agent_simulation,
    run_default_demo,
    run_multi_seed_comparison,
    run_static_warm_start_multi_seed,
)

__all__ = [
    "LinUCBAgent",
    "FastLinUCBAgent",
    "extract_rlearner_linear_priors_from_csv",
    "run_agent_simulation",
    "run_multi_seed_comparison",
    "run_static_warm_start_multi_seed",
    "run_default_demo",
]


if __name__ == "__main__":
    run_default_demo()
