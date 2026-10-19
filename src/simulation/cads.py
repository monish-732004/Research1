"""Carbon-Aware Deferral Scheduler (CADS).

CADS delays deferrable jobs (those with a start-time slack > 0) until the
grid carbon intensity drops below CADS_GREEN_PERCENTILE of the rolling
distribution, subject to CADS_MAX_DEFER_MINUTES hard deadline.

Key design decisions:
  - Non-deferrable jobs (latency-critical) are never delayed.
  - Deferral budget is bounded to avoid SLO violations.
  - Carbon intensity forecast uses a simple 1-hour persistence model
    (upgradeable to a learned forecast without changing the interface).
"""

import simpy
from src.config import (
    SIMPY_SEED, CADS_MAX_DEFER_MINUTES, CADS_GREEN_PERCENTILE,
)


class CADSScheduler:
    """Carbon-aware deferral on top of the baseline scheduler."""

    def __init__(
        self,
        env: simpy.Environment,
        n_gpus: int,
        intensity_forecast: dict[int, float],  # timestamp → gCO₂eq/kWh
    ) -> None:
        self.env = env
        self.n_gpus = n_gpus
        self.gpu_pool = simpy.Resource(env, capacity=n_gpus)
        self.intensity_forecast = intensity_forecast
        self.metrics: list[dict] = []

    def _is_green(self, ts: int) -> bool:
        """Return True if the forecast intensity at ts is below the green threshold."""
        # TODO: implement rolling percentile check
        raise NotImplementedError

    def submit_job(
        self,
        job_id: str,
        gpu_req: int,
        duration_s: float,
        deferrable: bool = True,
    ) -> None:
        # TODO: implement deferral logic + SimPy process
        raise NotImplementedError
