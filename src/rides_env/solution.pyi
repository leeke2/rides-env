import numpy as np
from .entities import (
    AllStopService as AllStopService,
    LimitedStopService as LimitedStopService,
)
from .instance import LSSDPInstance as LSSDPInstance
from .utils import calculate_stats as calculate_stats, trip_time as trip_time

class LSSDPSolution:
    def __init__(self, inst: LSSDPInstance) -> None: ...
    @property
    def stats(
        self,
    ) -> dict[str, tuple[np.floating, np.floating, np.floating, np.floating]]: ...
    def terminate(self) -> None: ...
    def toggle(self, stop: int) -> None: ...
    def add_bus(self) -> None: ...
