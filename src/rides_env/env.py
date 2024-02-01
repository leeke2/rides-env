from typing import Annotated, Any

import random
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import Env, spaces

from .instance import LSSDPInstance
from .network import SGNetwork  # type: ignore
from .solution import LSSDPSolution

ActType = int
ObsType = dict[str, Any]


class RidesEnv(Env):
    r"""The main Gymnasium class for implementing Reinforcement Learning Agents environments.

    The class encapsulates an environment with arbitrary behind-the-scenes dynamics through the :meth:`step` and :meth:`reset` functions.
    An environment can be partially or fully observed by single agents. For multi-agent environments, see PettingZoo.

    The main API methods that users of this class need to know are:

    - :meth:`step` - Updates an environment with actions returning the next agent observation, the reward for taking that actions,
      if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
    - :meth:`reset` - Resets the environment to an initial state, required before calling step.
      Returns the first agent observation for an episode and information, i.e. metrics, debug info.
    - :meth:`render` - Renders the environments to help visualise what the agent see, examples modes are "human", "rgb_array", "ansi" for text.
    - :meth:`close` - Closes the environment, important when external software is used, i.e. pygame for rendering, databases

    Environments have additional attributes for users to understand the implementation

    - :attr:`action_space` - The Space object corresponding to valid actions, all valid actions should be contained within the space.
    - :attr:`observation_space` - The Space object corresponding to valid observations, all valid observations should be contained within the space.
    - :attr:`spec` - An environment spec that contains the information used to initialize the environment from :meth:`gymnasium.make`
    - :attr:`metadata` - The metadata of the environment, i.e. render modes, render fps
    - :attr:`np_random` - The random number generator for the environment. This is automatically assigned during
      ``super().reset(seed=seed)`` and when assessing :attr:`np_random`.

    .. seealso:: For modifying or extending environments use the :py:class:`gymnasium.Wrapper` class

    Note:
        To get reproducible sampling of actions, a seed can be set with ``env.action_space.seed(123)``.
    """

    # # Set these in ALL subclasses
    # action_space: spaces.Space[ActType]
    # observation_space: spaces.Space[ObsType]

    # Created
    _np_random: np.random.Generator | None = None

    metadata = {"render_modes": ["human"], "render_fps": 1}

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
    ):
        self._nstops = nstops
        self._min_headway = min_headway
        self._max_headway = max_headway
        self._nbuses_full_min = nbuses_full_min
        self._truncate = truncate
        self._speed = speed
        self._dwell_time = dwell_time
        self._congested = congested
        self._capacity = capacity
        self._allow_retrospect = allow_retrospect

        self._demand_npeaks_max = demand_npeaks_max
        self._demand_peak_conc = demand_peak_conc
        self._demand_peak_size = demnad_peak_size
        self._demand_factor = demand_factor

        self._max_iters = max_iters

        self._network = SGNetwork()

        _mat = lambda: spaces.Box(
            0.0, float("inf"), shape=(self._nstops_max, self._nstops_max)
        )
        _oh = lambda size: spaces.Box(0, 1, (size,), dtype=bool)

        self.observation_space = spaces.Dict(
            {
                # Instance
                "od_demand": _mat(),
                "link_travel_time": _mat(),
                "base_od_travel_time": _mat(),
                # Solution
                "stops_lss": _oh(self._nstops_max),
                "nbuses_lss": spaces.Discrete(
                    100,
                    start=self._nbuses_full_min,
                ),
                # Metric
                "od_travel_time": _mat(),
                "base_invehicle_flow": _mat(),
                "invehicle_flow_ass": _mat(),
                "invehicle_flow_lss": _mat(),
                **(
                    {"load_factor_ass": _mat(), "load_factor_lss": _mat()}
                    if self._congested
                    else {}
                ),
            }
        )

        self.action_space = spaces.Discrete(self._nactions)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._inst: LSSDPInstance
        self._sol: LSSDPSolution

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        .. versionchanged:: 0.26

            The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
            to users when the environment had terminated or truncated which is critical for reinforcement learning
            bootstrapping algorithms.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        terminated = self._execute(action)

        # if self._sol._lss.is_valid():
        #     print(
        #         [
        #             1.0
        #             / self._inst.ass_trip_time
        #             * (self._inst.nbuses - self._sol._lss.nbuses),
        #             1.0
        #             / self._inst.trip_time(self._sol._lss.stops)
        #             * self._sol._lss.nbuses,
        #         ]
        #     )
        #     print(self._inst.ass_trip_time, self._inst.nbuses - self._sol._lss.nbuses)
        #     print(self._inst.ass_stops.stops)
        #     print(self._inst.trip_time(self._sol._lss.stops), self._sol._lss.nbuses)
        #     print(self._sol._lss.stops.stops)

        reward = self._sol._prev_obj - self._sol._obj

        if self.render_mode == "human":
            alignment = "".join(
                [
                    ("●" if selected else "○")
                    + (" " if (i + 1) % 5 == 0 and i != 0 else "")
                    for i, selected in enumerate(self._sol._lss.stops_binary)
                ]
            )

            if reward == 0:
                creward = f"{reward:9.4f}"
            elif reward > 0:
                creward = f"\033[32m{reward:9.4f}\033[0m"
            else:
                creward = f"\033[31m{reward:9.4f}\033[0m"

            stats = self._sol.stats
            _, ttd_mean, ttd_max, _ = stats["ttd"]
            _, lf_mean, lf_max, _ = stats["lf"]
            per_flow_exp, *_ = stats["per_flow_exp"]

            print(
                f" {self._step:2d} "
                f" {self._sol._obj:9.4f} "
                f" {creward} "
                f" {lf_mean:8.4f} / {lf_max:8.4f} "
                f" {ttd_mean:8.4f} / {ttd_max:8.4f} "
                f" {per_flow_exp: 6.4f} "
                f" {self._sol._lss.nbuses:3d} "
                f" {action:3d} "
                f" {alignment}"
            )

            if terminated:
                print("-" * 142)

        self._step += 1

        return self._observation, reward, terminated, False, self._info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. versionchanged:: v0.25

            The ``return_info`` parameter was removed and now info is expected to be returned.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed)

        if (
            options is None
            or "fix_problem" not in options
            or not options["fix_problem"]
        ):
            self._make_instance()

        self._sol = LSSDPSolution(self._inst)
        self._step = 1

        if self.render_mode == "human":
            spec_id = self.spec.id if self.spec is not None else ""

            print(f"  Env       : {spec_id}\n")
            self._inst.print_summary()

            div = "-" * 142
            print(
                f"{div}\n  #  Objective     Reward                   "
                f"LF                  TTD    % Exp  🚌  Act  Alignment\n{div}"
            )

        return self._observation, self._info

    def render(self):  # type: ignore
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        The environment's :attr:`metadata` render modes (`env.metadata["render_modes"]`) should contain the possible
        ways to implement the render modes. In addition, list versions for most render modes is achieved through
        `gymnasium.make` which automatically applies a wrapper to collect rendered frames.

        Note:
            As the :attr:`render_mode` is known during ``__init__``, the objects used to render the environment state
            should be initialised in ``__init__``.

        By convention, if the :attr:`render_mode` is:

        - None (default): no render is computed.
        - "human": The environment is continuously rendered in the current display or terminal, usually for human consumption.
          This rendering should occur during :meth:`step` and :meth:`render` doesn't need to be called. Returns ``None``.
        - "rgb_array": Return a single frame representing the current state of the environment.
          A frame is a ``np.ndarray`` with shape ``(x, y, 3)`` representing RGB values for an x-by-y pixel image.
        - "ansi": Return a strings (``str``) or ``StringIO.StringIO`` containing a terminal-style text representation
          for each time step. The text can include newlines and ANSI escape sequences (e.g. for colors).
        - "rgb_array_list" and "ansi_list": List based version of render modes are possible (except Human) through the
          wrapper, :py:class:`gymnasium.wrappers.RenderCollection` that is automatically applied during ``gymnasium.make(..., render_mode="rgb_array_list")``.
          The frames collected are popped after :meth:`render` is called or :meth:`reset`.

        Note:
            Make sure that your class's :attr:`metadata` ``"render_modes"`` key includes the list of supported modes.

        .. versionchanged:: 0.25.0

            The render function was changed to no longer accept parameters, rather these parameters should be specified
            in the environment initialised, i.e., ``gymnasium.make("CartPole-v1", render_mode="human")``
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return None

        return None

    ##### User defined
    @property
    def _observation(self) -> ObsType:
        pad = self._pad

        out = {
            # Instance
            "od_demand": pad(self._inst.demand),
            "link_travel_time": pad(self._inst.travel_time),
            "base_od_travel_time": pad(self._inst.base_ttd),
            # Solution
            "stops_lss": pad(np.asarray(self._sol._lss.stops_binary)),
            "nbuses_lss": self._sol._lss.nbuses,
            # Metric
            "od_travel_time": pad(self._sol._ttd),
            "base_invehicle_flow": pad(
                self._sol._ass.convert_invehicle_flow_to_mat(self._inst.base_flow)
            ),
            "invehicle_flow_ass": pad(self._sol._ass_flow_mat),
            "invehicle_flow_lss": pad(self._sol._lss_flow_mat),
        }

        if self._congested:
            return {
                **out,
                "load_factor_ass": pad(self._sol._ass_load_factor),
                "load_factor_lss": pad(self._sol._lss_load_factor),
            }

        return out

    @property
    def _action_mask(self):
        mask = np.zeros(self._nactions, dtype=np.bool_)

        mask[0] = True

        if self._allow_retrospect:
            mask[1 : 1 + self._inst.nstops] = ~np.array(self._sol._lss.stops_binary)
        else:
            mask[2 + self._sol._lss.last_stop : 1 + self._inst.nstops] = True

        if self._sol._lss.is_valid():
            mask[-1] = self._sol._lss.nbuses < self._inst.nbuses - self._nbuses_full_min

        return mask

    @property
    def _info(self) -> dict[str, Any]:
        return {
            "nstops": self._inst.nstops,
            "nbuses": self._inst.nbuses,
            "action_mask": self._action_mask,
            "capacity": self._inst.capacity,
        }

    @property
    def _nstops_max(self) -> int:
        if isinstance(self._nstops, int):
            return self._nstops

        return self._nstops[1]

    @property
    def _add_bus_action(self) -> int:
        return self._nactions - 1

    @property
    def _terminate_action(self) -> int:
        return 0

    @property
    def _nactions(self) -> int:
        return self._nstops_max + 2

    def _pad(self, mat: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if mat.shape == (self._nstops_max,) * mat.ndim:
            return mat

        pad_width = self._nstops_max - mat.shape[0]
        return np.pad(mat, ((0, pad_width),) * mat.ndim)

    def _make_instance(self):
        self._inst = LSSDPInstance.from_network(
            self._network,
            self._nstops,
            self._min_headway,
            self._max_headway,
            self._nbuses_full_min,
            self._speed,
            self._dwell_time,
            self._demand_npeaks_max,
            self._demand_peak_conc,
            self._demand_peak_size,
            self._demand_factor,
            self._congested,
            self._capacity,
            self._truncate,
            self._max_iters,
            self.np_random,
        )

        self._sol = LSSDPSolution(self._inst)

    def _execute(self, action: int) -> bool:
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")

        if action == self._terminate_action:
            self._sol.terminate()
            return True

        if action == self._add_bus_action:
            if not self._sol._lss.is_valid():
                raise ValueError(
                    "Cannot increase LSS allocation before having valid LSS"
                )

            if self._sol._lss.nbuses >= self._inst.nbuses - self._nbuses_full_min:
                raise ValueError("Allocation to limited stop service exceeded limit")

            self._sol.add_bus()
            return False

        if action - 1 >= self._inst.nstops:
            raise ValueError("Invalid action")

        if self._sol._lss.is_serving(action - 1):
            raise ValueError("Stop already served by limited stop service")

        if not self._allow_retrospect:
            if (
                not self._sol._lss.not_serving_any_stops()
                and action - 1 <= self._sol._lss.last_stop
            ):
                raise ValueError("Retrospective adding of stops not allowed")

        self._sol.toggle(action - 1)
        return False

    @property
    def sol(self) -> LSSDPSolution:
        return self._sol

    @property
    def inst(self) -> LSSDPInstance:
        return self._inst
