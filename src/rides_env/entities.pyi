import abc
from abc import ABC, abstractmethod
from dataclasses import dataclass as dataclass

import numpy.typing as npt

class Service(ABC, metaclass=abc.ABCMeta):
    def __init__(
        self, nstops: int, nbuses: int, travel_time_mat: npt.NDArray, capacity: float
    ) -> None: ...
    @property
    @abc.abstractmethod
    def stops(self) -> list[int]: ...
    @abc.abstractproperty
    def _invehicle_flow_indices(self) -> list[int]: ...
    @property
    def nbuses(self) -> int: ...
    @property
    def trip_time(self) -> float: ...
    @property
    def frequency(self) -> float: ...
    @property
    def max_load(self) -> float: ...
    @property
    def last_stop(self) -> int: ...
    @property
    def nstops(self) -> int: ...
    def is_valid(self) -> bool: ...
    @abstractmethod
    def is_serving(self, stop: int) -> bool: ...
    def convert_invehicle_flow_to_mat(self) -> npt.NDArray: ...

class AllStopService(Service):
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def stops(self) -> list[int]: ...
    def is_serving(self, stop: int) -> bool: ...
    def not_serving_any_stops(self) -> bool: ...
    def remove_bus(self) -> None: ...

class LimitedStopService(Service):
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def stops(self) -> list[int]: ...
    @property
    def stops_binary(self) -> list[bool]: ...
    def is_serving(self, stop: int) -> bool: ...
    def not_serving_any_stops(self) -> bool: ...
    def add_bus(self) -> None: ...
    def toggle(self, stop: int) -> None: ...
