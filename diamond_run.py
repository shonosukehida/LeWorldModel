"""Local and W&B identity for a DIAMOND training run."""
from itertools import count
from pathlib import Path
from uuid import uuid4
import warnings


def prepare_run(config, previous, state, started):
    """Reserve a fresh output atomically; restart attempts share the logical run identity."""
    prior = previous or {}
    inherited = "run_timestamp" in prior
    stamp = prior["run_timestamp"] if inherited else started.strftime("%Y%m%d_%H%M%S")
    run_id = prior.get("run_id") or state.get("run_id") or uuid4().hex
    name = config["wandb"]["name"] or "diamond"
    legacy_online = not inherited and bool(state.get("run_id"))
    if previous is not None and not inherited:
        warnings.warn("Checkpoint has no run timestamp; using this restart's Tokyo time")
    for attempt in count(1):
        suffix = "" if attempt == 1 else f"_{attempt:02d}"
        label = stamp + suffix
        folder = label if previous is None else f"{stamp}_resume{suffix}"
        out = Path(config["output"]).expanduser() if config["output"] else Path("outputs/diamond") / folder
        try:
            out.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if config["output"]:
                raise
            continue
        break
    if inherited:
        run_name = prior["run_name"]
    elif legacy_online:
        # None tells the SDK to retain a server-generated legacy name.
        run_name = prior.get("wandb", {}).get("name")
    else:
        run_name = f"{prior.get('wandb', {}).get('name') or name}_{label}"
    if state.get("run_id"):
        for key in ("project", "entity"):
            if key in prior.get("wandb", {}):
                config["wandb"][key] = prior["wandb"][key]
    config.update(output=str(out), run_timestamp=stamp, started_at=prior.get("started_at", started.isoformat()),
                  run_id=run_id, run_name=run_name)
    return out
