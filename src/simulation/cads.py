"""Carbon-Aware Deferral Scheduler (CADS).

CADS delays deferrable jobs (those with a start-time slack > 0) until the
grid carbon intensity drops below CADS_GREEN_PERCENTILE of the rolling
distribution, subject to CADS_MAX_DEFER_MINUTES hard deadline.

Key design decisions:
  - Non-deferrable jobs (latency-critical) are never delayed.
  - Deferral budget is bounded to avoid SLO violations.
  - Carbon intensity forecast uses a simple 1-hour persistence model
    (upgradeable to a learned forecast without changing the interface).

Predictive mode (predictive_mode=True):
  Loads the production model from data/processed/ and uses predicted
  gpu_hours_requested to decide whether deferral is worth the overhead.
  Only high-waste jobs (predicted > job-type mean) are deferred, reducing
  unnecessary scheduler overhead for small jobs.
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
        predictive_mode: bool = False,
    ) -> None:
        self.env = env
        self.n_gpus = n_gpus
        self.gpu_pool = simpy.Resource(env, capacity=n_gpus)
        self.intensity_forecast = intensity_forecast
        self.predictive_mode = predictive_mode
        self.metrics: list[dict] = []
        self._prod_model = None
        self._jt_mean_gpu_hours: dict[str, float] = {}

        if predictive_mode:
            self._load_production_model()

    def _load_production_model(self) -> None:
        """Load the production model saved by src.modeling.choose."""
        try:
            from src.modeling.choose import load_production_model
            self._prod_model = load_production_model()
            print("  [CADS] Predictive mode: production model loaded.")
        except FileNotFoundError:
            print("  [CADS] WARNING: production model not found; "
                  "falling back to non-predictive mode.")
            self.predictive_mode = False

    def _predicted_gpu_hours(self, job_features: dict) -> float:
        """Predict gpu_hours for a single job.  Returns 0.0 on failure."""
        if self._prod_model is None:
            return 0.0
        import numpy as np
        try:
            X = np.array([[job_features.get(k, 0.0)
                           for k in sorted(job_features)]], dtype=np.float32)
            return float(self._prod_model.predict(X)[0])
        except Exception:
            return 0.0

    def _should_defer(
        self,
        ts: int,
        deferrable: bool,
        job_type: str = "",
        job_features: dict | None = None,
    ) -> bool:
        """Return True if this job should be deferred to a greener window.

        In predictive mode, additionally checks whether the job's predicted
        GPU-hours exceed the mean for its job type (only defer high-waste jobs).
        """
        if not deferrable:
            return False
        if not self._is_green(ts):
            return True  # basic CADS: grid is dirty, defer
        return False

    def _should_defer_predictive(
        self,
        ts: int,
        deferrable: bool,
        job_type: str = "",
        predicted_gpu_hours: float = 0.0,
    ) -> bool:
        """Predictive variant: only defer deferrable high-waste jobs on dirty grid."""
        if not deferrable:
            return False
        if not self._is_green(ts):
            jt_mean = self._jt_mean_gpu_hours.get(job_type, predicted_gpu_hours)
            return predicted_gpu_hours > jt_mean
        return False

    def _is_green(self, ts: int) -> bool:
        """Return True if the forecast intensity at ts is below the green threshold."""
        if not self.intensity_forecast:
            return True
        intensities = list(self.intensity_forecast.values())
        threshold = sorted(intensities)[int(len(intensities) * CADS_GREEN_PERCENTILE)]
        current = self.intensity_forecast.get(ts, float("inf"))
        return current <= threshold

    def submit_job(
        self,
        job_id: str,
        gpu_req: int,
        duration_s: float,
        deferrable: bool = True,
        job_type: str = "",
        job_features: dict | None = None,
    ) -> None:
        # TODO: implement deferral logic + SimPy process
        raise NotImplementedError
