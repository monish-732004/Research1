"""Baseline scheduler: replay the original Alibaba trace without modification.

Feeds jobs into the SimPy environment in arrival order and records actual
GPU utilisation, idle time, and carbon cost for comparison against CADS.
"""

import simpy
from src.config import SIMPY_SEED, RANDOM_SEED


class BaselineScheduler:
    """Replay jobs exactly as they appear in the trace."""

    def __init__(self, env: simpy.Environment, n_gpus: int) -> None:
        self.env = env
        self.n_gpus = n_gpus
        self.gpu_pool = simpy.Resource(env, capacity=n_gpus)
        self.metrics: list[dict] = []

    def submit_job(self, job_id: str, gpu_req: int, duration_s: float) -> None:
        # TODO: implement job process coroutine
        raise NotImplementedError
