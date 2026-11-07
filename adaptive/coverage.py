"""Coverage / trigger logic for adaptive evaluation (paper §method-adaptive).

Trigger condition (Algorithm 2): the current θ̂ has drifted outside the panel
θ̂ range (extended by a small pad τ_pad). This signals that the target model
is out-of-panel and anchor cells alone cannot calibrate it — Step-2 SGBS
must be invoked to mint new boundary items.

Hysteresis: trigger only after the iteration guard t ≥ t_min AND L
consecutive iterations of coverage failure (suppresses early-iteration noise
from a small pilot).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoverageState:
    consecutive_failures: int = 0   # 'c' in Algorithm 2

    def record(self, failed: bool) -> None:
        self.consecutive_failures = self.consecutive_failures + 1 if failed else 0

    def reset(self) -> None:
        self.consecutive_failures = 0


def coverage_failed(theta_hat: float,
                    panel_theta_min: float,
                    panel_theta_max: float,
                    tau_pad: float) -> bool:
    """θ̂ falls outside [panel_min - tau_pad, panel_max + tau_pad]."""
    return theta_hat < panel_theta_min - tau_pad or theta_hat > panel_theta_max + tau_pad


def should_trigger(t: int, t_min: int, c: int, L: int) -> bool:
    """Iteration guard t ≥ t_min AND hysteresis c ≥ L."""
    return t >= t_min and c >= L
