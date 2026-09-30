from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from immune.core.calls import SensingPlan, SensorResults
from immune.core.pipeline import CallPipeline
from immune.errors import SensorUnavailable
from immune.sensing.loop import SensorLoop
from immune.sensing.sensor import Sensor
from immune.telemetry.throttle import ThrottledLog

_LOGGER = logging.getLogger("immune")
_THROTTLED = ThrottledLog(_LOGGER)
PendingSensing = concurrent.futures.Future[SensorResults] | None


class SensingDispatch:
    def __init__(self, pipeline: CallPipeline, loop: SensorLoop, sensor: Sensor, budget_s: float) -> None:
        self._pipeline = pipeline
        self._loop = loop
        self._sensor = sensor
        self._budget_s = budget_s

    @property
    def budget_s(self) -> float:
        return self._budget_s

    def start(self, plan: SensingPlan) -> PendingSensing:
        if plan.is_empty:
            return None
        return self._loop.submit(self._pipeline.sense(plan))

    def wait(self, plan: SensingPlan, pending: PendingSensing) -> SensorResults:
        if pending is None:
            return {}
        try:
            return pending.result(timeout=self._budget_s)
        except TimeoutError:
            pending.cancel()
            return self._expired(plan)
        except Exception as error:
            return plan.unavailable(error)

    def collect(self, plan: SensingPlan) -> SensorResults:
        return self.wait(plan, self.start(plan))

    async def gather(self, plan: SensingPlan) -> SensorResults:
        if plan.is_empty:
            return {}
        try:
            return await asyncio.wait_for(self._loop.run(self._pipeline.sense(plan)), self._budget_s)
        except TimeoutError:
            return await asyncio.to_thread(self._expired, plan)
        except Exception as error:
            return plan.unavailable(error)

    def _expired(self, plan: SensingPlan) -> SensorResults:
        _THROTTLED.warning("deadline", "immune: screening exceeded %.2fs; continuing without Jev", self._budget_s)
        if self._loop.recover():
            self._sensor.reset()
        return plan.unavailable(SensorUnavailable(f"screening exceeded {self._budget_s:.2f}s"))
