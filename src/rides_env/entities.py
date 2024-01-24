from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass

import numpy.typing as npt
import numpy as np

# @dataclass
# class StopSequence:
#     stops: list[int]

#     def __len__(self) -> int:
#         return self.stops.__len__()

#     def __hash__(self) -> int:
#         return tuple(self.stops).__hash__()

#     def __getitem__(self, index):
#         return self.stops.__getitem__(index)


class Service(ABC):
    def __init__(
        self, nstops: int, nbuses: int, travel_time_mat: npt.NDArray, capacity: float
    ):
        self._nbuses = nbuses
        self._nstops = nstops

        self._travel_time = travel_time_mat
        self._capacity = capacity

        self._last_stop = -1

    @abstractproperty
    def stops(self) -> list[int]:
        raise NotImplementedError()

    @abstractproperty
    def _invehicle_flow_indices(self) -> list[int]:
        raise NotImplementedError()

    @property
    def nbuses(self) -> int:
        return self._nbuses

    @property
    def trip_time(self) -> float:
        return sum(
            self._travel_time[from_][to_]
            for from_, to_ in zip(self.stops[:-1], self.stops[1:])
        )

    @property
    def frequency(self) -> float:
        return self.nbuses / self.trip_time

    @property
    def max_load(self) -> float:
        return self.frequency * self._capacity

    @property
    def last_stop(self) -> int:
        return self._last_stop

    @property
    def nstops(self) -> int:
        return len(self.stops)

    def is_valid(self) -> bool:
        return self.nstops >= 2 and self.nbuses >= 1

    @abstractmethod
    def is_serving(self, stop: int) -> bool:
        raise NotImplementedError()

    def convert_invehicle_flow_to_mat(
        self, flow: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        if not self.is_valid():
            return np.array([[0.0]], dtype=np.float32)

        nstops = self.stops[-1] + 1
        out = np.zeros((nstops, nstops), dtype=np.float32)
        values = flow[self._invehicle_flow_indices]
        indices = np.array(self.stops[:-1]) * nstops + np.array(self.stops[1:])
        out.put(indices, values)

        return out

    def remove_bus(self) -> None:
        self._nbuses -= 1

    def add_bus(self) -> None:
        self._nbuses += 1


class AllStopService(Service):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._stops = list(range(self._nstops))

    @property
    def stops(self) -> list[int]:
        return self._stops

    def is_serving(self, stop: int) -> bool:
        return stop < self._nstops

    def not_serving_any_stops(self) -> bool:
        return False

    @property
    def _invehicle_flow_indices(self) -> list[int]:
        return list(range(1, (self.nstops - 1) * 3, 3))


class LimitedStopService(Service):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._stops_binary = [False for stop in range(self._nstops)]

    @property
    def stops(self) -> list[int]:
        return [i for i, served in enumerate(self._stops_binary) if served]

    @property
    def stops_binary(self) -> list[bool]:
        return self._stops_binary

    def is_serving(self, stop: int) -> bool:
        return self._stops_binary[stop]

    def not_serving_any_stops(self) -> bool:
        return sum(self._stops_binary) == 0

    def toggle(self, stop: int) -> None:
        self._stops_binary[stop] = False if self._stops_binary[stop] else True

        if stop >= self._last_stop:
            if self._stops_binary[stop]:
                self._last_stop = stop
            elif sum(self._stops_binary) == 0:
                self._last_stop = -1
        else:
            for i in range(self._last_stop, -1, -1):
                if self._stops_binary[i]:
                    self._last_stop = i
                    break

    @property
    def _invehicle_flow_indices(self) -> list[int]:
        return [self._nstops + i for i in range(1, (self.nstops - 1) * 3, 3)]
