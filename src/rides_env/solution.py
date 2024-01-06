import numpy.typing as npt
import numpy as np
from .utils import trip_time, calculate_stats

from .instance import LSSDPInstance
from .entities import Service

from tram import mat_linear_assign, mat_linear_congested_assign


class LSSDPSolution:
    def __init__(self, inst: LSSDPInstance) -> None:
        self._inst = inst

        self._lss = Service(nstops=inst.nstops, nbuses=1)
        self._prev_obj = 1.0
        self._obj = 1.0
        self._ttd = inst.base_ttd
        self._flow = inst.base_flow
        self._rel_ttd = np.triu(np.ones_like(self._ttd), 1)

    @property
    def _capacities(self) -> npt.NDArray[np.floating]:
        if not self._lss.is_valid():
            return np.array(
                [self._inst.nbuses / self._inst.ass_trip_time * self._inst.capacity]
                * 3
                * (self._inst.nstops - 1)
            )

        return np.array(
            (
                [
                    (self._inst.nbuses - self._lss.nbuses)
                    / trip_time(self._inst.travel_time, self._lss.stops)
                    * self._inst.capacity
                ]
                * 3
                * (self._inst.nstops - 1)
            )
            + (
                [self._lss.nbuses / self._inst.ass_trip_time * self._inst.capacity]
                * 3
                * (len(self._lss.stops) - 1)
            )
        )

    @property
    def stats(
        self,
    ) -> dict[str, tuple[np.floating, np.floating, np.floating, np.floating]]:
        if not self._lss.is_valid():
            per_flow_exp = 0.0
        else:
            per_flow_exp = (
                self._flow[3 * (self._inst.nstops - 1) :].sum() / self._flow.sum()
            )

        return {
            "ttd": calculate_stats(self._ttd),
            "lf": calculate_stats(
                np.divide(self._flow, self._capacities)
                if self._inst.congested
                else [float("nan")]
            ),
            "per_flow_exp": calculate_stats([per_flow_exp]),
        }

    def terminate(self) -> None:
        self._prev_obj = self._obj

    def toggle(self, stop: int) -> None:
        self._lss.toggle(stop)
        self._calculate_objective()

    def add_bus(self) -> None:
        self._lss.add_bus()
        self._calculate_objective()

    def _calculate_objective(self) -> None:
        if not self._lss.is_valid():
            self._prev_obj = self._obj
            self._obj = 1.0
            self._ttd = self._inst.base_ttd
            self._rel_ttd = np.triu(np.ones_like(self._ttd), 1)

            return

        alignments = [self._inst.ass_stops.stops, self._lss.stops.stops]
        frequencies = [
            1.0 / self._inst.ass_trip_time * (self._inst.nbuses - self._lss.nbuses),
            1.0 / trip_time(self._inst.travel_time, self._lss.stops) * self._lss.nbuses,
        ]

        if self._inst.congested:
            out = mat_linear_congested_assign(
                alignments,
                frequencies,
                self._inst.travel_time,
                self._inst.demand,
                self._inst.capacity,
                max_iters=self._inst.max_iters,
            )
        else:
            out = mat_linear_assign(
                alignments,
                frequencies,
                self._inst.travel_time,
                self._inst.demand,
            )

        self._prev_obj = self._obj
        self._obj = out[2] / self._inst.base_obj
        self._ttd = np.asarray(out[0], dtype=np.float32)
        self._flow = np.asarray(out[1], dtype=np.float32)
        self._rel_ttd = np.divide(
            self._ttd,
            self._inst.base_ttd,
            out=np.zeros_like(self._inst.base_ttd),
            where=(self._inst.base_ttd != 0),
        )
