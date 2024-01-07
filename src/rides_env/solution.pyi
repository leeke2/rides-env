import numpy as np
import numpy.typing as npt

from .entities import AllStopService as AllStopService
from .entities import LimitedStopService as LimitedStopService
from .instance import LSSDPInstance as LSSDPInstance
from .utils import calculate_stats as calculate_stats
from .utils import trip_time as trip_time

class LSSDPSolution:
    _inst: LSSDPInstance
    _ass: AllStopService
    _lss: LimitedStopService
    _obj: float
    _prev_obj: float
    _ttd: npt.NDArray[np.floating]
    _flow: npt.NDArray[np.floating]
    def __init__(self, inst: LSSDPInstance) -> None: ...
    @property
    def stats(
        self,
    ) -> dict[str, tuple[np.floating, np.floating, np.floating, np.floating]]: ...
    def terminate(self) -> None: ...
    def toggle(self, stop: int) -> None: ...
    def add_bus(self) -> None: ...
