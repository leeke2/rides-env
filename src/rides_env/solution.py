import numpy as np
import numpy.typing as npt
from tram import mat_linear_assign, mat_linear_congested_assign

from .entities import AllStopService, LimitedStopService
from .instance import LSSDPInstance
from .utils import calculate_stats, trip_time


class LSSDPSolution:
    def __init__(self, inst: LSSDPInstance) -> None:
        self._inst = inst

        self._ass = AllStopService(
            inst.nstops, inst.nbuses - 1, inst.travel_time, inst.capacity
        )
        self._lss = LimitedStopService(inst.nstops, 1, inst.travel_time, inst.capacity)

        self._prev_obj = 1.0
        self._obj = 1.0
        self._ttd = inst.base_ttd
        self._flow = inst.base_flow

    @property
    def _lf(self) -> npt.NDArray[np.floating]:
        return (
            np.divide(self._flow, self._capacities)
            if self._inst.congested
            else np.array([float("nan")], dtype=np.float32)
        )

    @property
    def _ass_load_factor(self) -> npt.NDArray[np.floating]:
        return (
            self._ass.convert_invehicle_flow_to_mat(self._lf)
            if self._inst.congested
            else np.array([float("nan")], dtype=np.float32)
        )

    @property
    def _lss_load_factor(self) -> npt.NDArray[np.floating]:
        return (
            self._lss.convert_invehicle_flow_to_mat(self._lf)
            if self._inst.congested
            else np.array([float("nan")], dtype=np.float32)
        )

    @property
    def _ass_flow_mat(self) -> npt.NDArray[np.floating]:
        return self._ass.convert_invehicle_flow_to_mat(self._flow)

    @property
    def _lss_flow_mat(self) -> npt.NDArray[np.floating]:
        return self._lss.convert_invehicle_flow_to_mat(self._flow)

    @property
    def _capacities(self) -> npt.NDArray[np.floating]:
        if not self._lss.is_valid():
            return np.array([self._inst._oris.max_load] * 3 * (self._inst.nstops - 1))

        return np.array(
            ([self._ass.max_load] * 3 * (self._ass.nstops - 1))
            + ([self._lss.max_load] * 3 * (self._lss.nstops - 1))
        )

    @property
    def stats(
        self,
    ) -> dict[str, tuple[np.floating, np.floating, np.floating, np.floating]]:
        if not self._lss.is_valid():
            per_flow_exp = 0.0
        else:
            per_flow_exp = (
                self._flow[3 * (self._ass.nstops - 1) :].sum() / self._flow.sum()
            )

        return {
            "ttd": calculate_stats(self._ttd),
            "lf": calculate_stats(self._lf),
            "per_flow_exp": calculate_stats([per_flow_exp]),
        }

    def terminate(self) -> None:
        self._prev_obj = self._obj

    def toggle(self, stop: int) -> None:
        self._lss.toggle(stop)
        self._calculate_objective()

    def add_bus(self) -> None:
        self._ass.remove_bus()
        self._lss.add_bus()
        self._calculate_objective()

    def remove_bus(self) -> None:
        self._ass.add_bus()
        self._lss.remove_bus()
        self._calculate_objective()

    def _calculate_objective(self) -> None:
        if not self._lss.is_valid():
            self._prev_obj = self._obj
            self._obj = 1.0
            self._ttd = self._inst.base_ttd

            return

        alignments = [self._ass.stops, self._lss.stops]
        frequencies = [self._ass.frequency, self._lss.frequency]

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
