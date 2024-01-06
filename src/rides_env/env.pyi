from typing import Annotated, Any

from gymnasium import Env, spaces

from .instance import LSSDPInstance as LSSDPInstance
from .network import SGNetwork as SGNetwork
from .solution import LSSDPSolution as LSSDPSolution

ActType = int
ObsType = dict[str, Any]

class RidesEnv(Env):
    metadata: dict[str, Any]
    observation_space: spaces.Space
    action_space: spaces.Space
    render_mode: str | None
    def __init__(
        self,
        nstops: int | list[int] = [30, 45],
        min_headway: Annotated[float, "min"] = 3.0,
        max_headway: Annotated[float, "min"] = 15.0,
        nbuses_full_min: int = 1,
        truncate: bool = True,
        capacity: float | None = 90.0,
        congested: bool = True,
        speed: Annotated[float, "kmh"] = 17.0,
        dwell_time: Annotated[float, "min"] = 0.5,
        allow_retrospect: bool = False,
        demand_npeaks_max: int = 50,
        demand_peak_conc: float = 2.0,
        demnad_peak_size: float = 150.0,
        demand_factor: float = 1.25,
        max_iters: int = 10000,
        render_mode: str | None = None,
    ) -> None: ...
    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]: ...
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]: ...
    def render(self) -> None: ...
    @property
    def sol(self) -> LSSDPSolution: ...
    @property
    def inst(self) -> LSSDPInstance: ...
