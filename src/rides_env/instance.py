import hashlib
from math import ceil, floor
from typing import Annotated

import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal  # type: ignore
from tram import mat_linear_assign, mat_linear_congested_assign

from .entities import AllStopService
from .network import SGNetwork
from .utils import calculate_stats, trip_time


class LSSDPInstance:
    def __init__(
        self,
        travel_time: npt.NDArray[np.floating],
        demand: npt.NDArray[np.floating],
        nbuses: int,
        capacity: float,
        congested: bool,
        base_ttd: npt.NDArray[np.floating] = np.array([]),
        base_flow: npt.NDArray[np.floating] = np.array([]),
        base_obj: float = 0.0,
        max_iters: int = 10000,
        name: str | None = None,
    ):
        self.travel_time = travel_time
        self.demand = demand
        self.nbuses = nbuses
        self.capacity = capacity
        self.congested = congested
        self.base_ttd = base_ttd
        self.base_flow = base_flow
        self.base_obj = base_obj
        self.max_iters = max_iters
        self.name = name

        self._oris = AllStopService(self.nstops, nbuses, travel_time, capacity)

        self._id: str = ""

    def visualise(self) -> None:
        import math

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.axes_divider import (
            make_axes_locatable,
        )  # type: ignore

        def nanlower(arr):
            arr[np.tril_indices(arr.shape[0])] = np.nan
            return arr

        plt.rcParams.update(
            {
                "image.cmap": "RdYlBu",
                "axes.grid": True,
                "axes.grid.which": "major",
                "grid.color": "white",
                "grid.linewidth": 2,
                "axes.spines.left": False,
                "axes.spines.bottom": False,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "xtick.labeltop": False,
                "xtick.labelbottom": False,
                "ytick.labelleft": False,
                "ytick.labelright": False,
                "xtick.top": False,
                "xtick.bottom": False,
                "ytick.left": False,
                "ytick.right": False,
            }
        )

        fig = plt.figure(figsize=(20, 10))
        gs = mpl.gridspec.GridSpec(6, 12, wspace=0.25, hspace=0.25)

        arr = fig.add_subplot(gs[0, :4])
        dep = fig.add_subplot(gs[1:5, 4])
        dem = fig.add_subplot(gs[1:5, :4])
        tt = fig.add_subplot(gs[1:5, 6:10])

        demand = self.demand
        demand_dep = demand.sum(axis=1)
        demand_arr = demand.sum(axis=0)
        demand_max = max(demand_dep.max(), demand_arr.max())
        interval = 5

        nstops = demand.shape[0]
        n_intervals = math.ceil(nstops / interval)

        dep.plot(demand_dep, range(nstops - 1, -1, -1))
        dep.set_ylim(-0.5, nstops - 0.5)
        dep.set_xlim(0, demand_max)

        arr.plot(demand_arr)
        arr.set_xlim(-0.5, nstops - 0.5)
        arr.set_ylim(0, demand_max)

        dem.imshow(nanlower(demand.copy()), cmap="OrRd", vmin=0)
        _ = dem.set_title("Demand", y=-0.1)
        _ = dem.set_xticks(np.arange(-0.5, n_intervals * interval - 0.5, interval))
        _ = dem.set_yticks(np.arange(-0.5, n_intervals * interval - 0.5, interval))
        for i in range(1, nstops + 1, interval):
            # _ = dem.plot([i-1, i-2], [i-1, i], 'k', lw=0.25)
            _ = dem.text(
                i - 2,
                i,
                str(i),
                rotation=-45,
                horizontalalignment="center",
                verticalalignment="center",
            )

        tt.imshow(nanlower(self.travel_time.copy()), cmap="OrRd", vmin=0)
        _ = tt.set_title("Travel time", y=-0.1)
        _ = tt.set_xticks(np.arange(-0.5, n_intervals * interval - 0.5, interval))
        _ = tt.set_yticks(np.arange(-0.5, n_intervals * interval - 0.5, interval))
        for i in range(1, nstops + 1, interval):
            # _ = tt.plot([i-1, i-2], [i-1, i], 'k', lw=0.25)
            _ = tt.text(
                i - 2,
                i,
                str(i),
                rotation=-45,
                horizontalalignment="center",
                verticalalignment="center",
            )

        plt.show()

    def calculate_id(self) -> None:
        h = hashlib.sha256()

        h.update(str(hash(tuple(self.travel_time.flatten().tolist()))).encode("utf8"))
        h.update(str(hash(tuple(self.demand.flatten().tolist()))).encode("utf8"))
        h.update(str(self.nbuses).encode("utf8"))
        h.update(str(self.capacity).encode("utf8"))

        self._id = h.hexdigest()[:10]

    def print_summary(self) -> None:
        if self.congested:
            congested_str = "\033[32mTrue\033[0m"
        else:
            congested_str = "\033[31mFalse\033[0m"

        print(
            f"  Instance  : {self._id}\n"
            f"  Name      : {self.name}\n"
            f"  Buses     : {self.nbuses}\n"
            f"  Stops     : {self.travel_time.shape[0]}\n"
            f"  Headway   : {(1.0 / self._oris.frequency):.1f} min\n"
            f"  Capacity  : {self.capacity}\n"
            f"  Congested : {congested_str}\n"
            f"  Objective : {self.base_obj:.4f}\n"
        )

        print("  \033[4mStats (min/avg/max/sum)\033[0m")
        print(
            "  OD Demand         : {:7.2f} / {:7.2f} / {:7.2f} / {:8.2f}\n"
            "  Dep Demand        : {:7.2f} / {:7.2f} / {:7.2f} / {:8.2f}\n"
            "  Arr Demand        : {:7.2f} / {:7.2f} / {:7.2f} / {:8.2f}\n"
            "  Link Travel time  : {:7.2f} / {:7.2f} / {:7.2f} / {:8.2f} (min)\n"
            "  Time to dest      : {:7.2f} / {:7.2f} / {:7.2f} / {:8.2f} (min)\n"
            "  Load factor       : {:7.2f} / {:7.2f} / {:7.2f} / {:8.2f} (%)\n".format(
                *calculate_stats(self.demand),
                *calculate_stats(self.demand.sum(axis=1)),
                *calculate_stats(self.demand.sum(axis=0)),
                *calculate_stats(self.travel_time, sum=False),
                *calculate_stats(self.base_ttd),
                *calculate_stats(
                    (self.base_flow / (self._oris.frequency * self.capacity) * 100)
                    if self.congested
                    else [float("nan")],
                    sum=False,
                ),
            )
        )

    @property
    def nstops(self) -> int:
        return self.travel_time.shape[0]

    @staticmethod
    def from_network(
        network: SGNetwork,
        nstops: int | list[int],
        min_headway: float,
        max_headway: float,
        nbuses_full_min: int,
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
    ) -> "LSSDPInstance":
        # Generate travel time matrix
        # _, distance, info = network.sample_test_route()
        if isinstance(nstops, int):
            _, distance, info = network.sample_real_route(
                nstops, truncate=truncate, rng=rng
            )
        elif isinstance(nstops, list):
            _, distance, info = network.sample_real_route(
                nstops[0], max_num_nodes=nstops[1], truncate=truncate, rng=rng
            )

        travel_time = np.triu(distance / speed / 1000 * 60 + dwell_time, 1)
        inst_nstops = travel_time.shape[0]

        ass_trip_time = trip_time(travel_time, list(range(inst_nstops)))

        min_nbuses = ceil(ass_trip_time / max_headway)
        max_nbuses = floor(ass_trip_time / min_headway) + 1

        if min_nbuses < nbuses_full_min + 1:
            min_nbuses = nbuses_full_min + 1

        if max_nbuses <= min_nbuses:
            max_nbuses = min_nbuses + 1

        inst_nbuses = rng.integers(
            min_nbuses,
            high=max_nbuses,
        )

        # Generate demand matrix
        demand = rng.random(size=(inst_nstops, inst_nstops))

        if demand_npeaks_max > 1:
            for _ in range(rng.integers(1, high=demand_npeaks_max + 1)):
                mean = sorted(rng.integers(1, high=inst_nstops + 1, size=(2,)))
                var = np.diag(
                    rng.random(size=(2,))
                    * inst_nstops
                    / rng.random()
                    / demand_peak_conc
                )

                x, y = np.mgrid[1 : inst_nstops + 1, 1 : inst_nstops + 1]
                pos = np.dstack((x, y))

                rv = multivariate_normal(mean, var, allow_singular=True)
                demand += rng.random() * rv.pdf(pos) * demand_peak_size

        demand = np.triu(demand, 1)
        demand /= inst_nstops + 1 - np.arange(inst_nstops).reshape(-1, 1)
        demand /= np.sum(demand)
        demand *= (
            inst_nbuses
            / ass_trip_time
            * capacity
            * demand_factor
            * (rng.random() * 0.5 + 0.5)
        )

        inst = LSSDPInstance(
            travel_time=travel_time.astype(np.float32),
            demand=demand.astype(np.float32),
            nbuses=inst_nbuses,
            capacity=capacity,
            congested=congested,
        )

        # Calculate objective
        # out = mat_linear_assign(
        #     [inst.ass_stops.stops],
        #     [1 / inst.ass_trip_time * inst_nbuses],
        #     travel_time,
        #     demand,
        # )

        if congested:
            out = mat_linear_congested_assign(
                [inst._oris.stops],
                [inst._oris.frequency],
                travel_time,
                demand,
                capacity,
                max_iters=max_iters,
            )
        else:
            out = mat_linear_assign(
                [inst._oris.stops],
                [inst._oris.frequency],
                travel_time,
                demand,
            )

        inst.base_ttd = np.asarray(out[0], dtype=np.float32)
        inst.base_flow = np.asarray(out[1], dtype=np.float32)
        inst.base_obj = out[2]
        inst.name = info["name"]

        inst.calculate_id()

        return inst

    def __hash__(self) -> int:
        return self._id.__hash__()
