from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

from .contracts import CausalPipelineResult

DEFAULT_CAUSAL_ARTIFACT_FILENAME = "causal_pipeline_result.pkl"


def save_causal_pipeline_result(
    result: CausalPipelineResult,
    *,
    artifact_dir: str | Path,
    file_name: str = DEFAULT_CAUSAL_ARTIFACT_FILENAME,
    logger: logging.Logger | None = None,
) -> Path:
    """Persist Step 1-5 output for reuse in Step 6/7 experiments."""
    log = logger or logging.getLogger(__name__)

    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / file_name

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }

    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("Saved causal pipeline artifact to %s", artifact_path.as_posix())
    return artifact_path


def load_causal_pipeline_result(
    *,
    artifact_dir: str | Path,
    file_name: str = DEFAULT_CAUSAL_ARTIFACT_FILENAME,
    logger: logging.Logger | None = None,
) -> CausalPipelineResult:
    """Load persisted Step 1-5 output and skip expensive retraining."""
    log = logger or logging.getLogger(__name__)

    artifact_path = Path(artifact_dir) / file_name
    if not artifact_path.exists():
        raise FileNotFoundError(
            "Causal artifact not found at "
            f"{artifact_path.as_posix()}. Run once with --save-causal-artifacts first."
        )

    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, CausalPipelineResult):
        result = payload
    elif isinstance(payload, dict) and "result" in payload:
        result = payload["result"]
    else:
        raise ValueError(
            "Unexpected causal artifact format. Remove the artifact and regenerate it with "
            "--save-causal-artifacts."
        )

    if not isinstance(result, CausalPipelineResult):
        raise ValueError(
            "Loaded causal artifact is not a CausalPipelineResult instance. "
            "Regenerate it with --save-causal-artifacts."
        )

    log.info("Loaded causal pipeline artifact from %s", artifact_path.as_posix())
    return result
