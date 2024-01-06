from typing import Annotated

import numpy as np
import numpy.typing as npt

from .entities import AllStopService as AllStopService
from .network import SGNetwork as SGNetwork
from .utils import calculate_stats as calculate_stats
from .utils import trip_time as trip_time

class LSSDPInstance:
    travel_time: npt.NDArray[np.floating]
    demand: npt.NDArray[np.floating]
    nbuses: int
    capacity: float
    congested: bool
    base_ttd: npt.NDArray[np.floating]
    base_flow: npt.NDArray[np.floating]
    base_obj: float
    max_iters: int
    name: str | None
    _oris: AllStopService
    def __init__(
        self,
        travel_time: npt.NDArray[np.floating],
        demand: npt.NDArray[np.floating],
        nbuses: int,
        capacity: float,
        congested: bool,
        base_ttd: npt.NDArray[np.floating] = ...,
        base_flow: npt.NDArray[np.floating] = ...,
        base_obj: float = 0.0,
        max_iters: int = 10000,
        name: str | None = None,
    ) -> None: ...
    def visualise(self) -> None: ...
    def calculate_id(self) -> None: ...
    def print_summary(self) -> None: ...
    @property
    def nstops(self) -> int: ...
    @staticmethod
    def from_network(
        network: SGNetwork,
        nstops: int | list[int],
        min_headway: float,
        max_headway: float,
        speed: Annotated[float, "kmh"],
        dwell_time: Annotated[float, "min"],
        demand_npeaks_max: int,
        demand_peak_conc: float,
        demand_peak_size: float,
        demand_factor: float,
        congested: bool,
        capacity: float,
        truncate: bool,
        max_iters: int,
        rng: np.random.Generator,
    ) -> LSSDPInstance: ...
    def __hash__(self) -> int: ...
