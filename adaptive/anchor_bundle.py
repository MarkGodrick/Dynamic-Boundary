"""Anchor bundle: the calibrated artifact that adaptive evaluation consumes.

Bundle contents:
  - panel_theta: dict[model_name -> θ̂] from the panel-wide Rasch fit
  - items: list of (item_id, prompt, metadata, β̂) on the calibrated logit scale
  - filter_spec, M, N: identifying metadata for reproducibility

Built once per category by scripts/build_anchor_bundle.py from the panel
response matrix + filter selection; consumed by adaptive.run_adaptive_eval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AnchorItem:
    item_id: str
    prompt: str
    metadata: dict
    beta: float


@dataclass
class AnchorBundle:
    category: str
    panel_models: list[str]
    panel_theta: dict[str, float]
    items: list[AnchorItem]
    filter_spec: str
    M: int
    N: int

    @property
    def J(self) -> int:
        return len(self.items)

    def betas(self) -> list[float]:
        return [it.beta for it in self.items]


def load_anchor_bundle(path: str | Path) -> AnchorBundle:
    with open(path) as f:
        d = json.load(f)
    items = [AnchorItem(**it) for it in d["items"]]
    return AnchorBundle(
        category=d["category"],
        panel_models=d["panel_models"],
        panel_theta=d["panel_theta"],
        items=items,
        filter_spec=d["filter_spec"],
        M=d["M"],
        N=d["N"],
    )


def save_anchor_bundle(bundle: AnchorBundle, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "category": bundle.category,
        "panel_models": bundle.panel_models,
        "panel_theta": bundle.panel_theta,
        "filter_spec": bundle.filter_spec,
        "M": bundle.M,
        "N": bundle.N,
        "items": [asdict(it) for it in bundle.items],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
