from typing import Optional

import numpy as np

def idx_max(lst, skip: int = 0): ...

class SGNetwork:
    path_to_data: str
    def __init__(self, path_to_data: str = ".rides_sgnetwork") -> None: ...
    def generate_gmap_link(self, route): ...
    def generate_trunk_route(
        self,
        min_num_nodes: int,
        max_num_nodes: Optional[int] = None,
        rng: np.random.Generator | None = None,
        min_terminal_distance: float = 5000.0,
        sample_candidate_threshold_distance: float = 5000.0,
        max_path_detour: float = 1.2,
        max_link_detour: float = 1.3,
        min_search_radius: float = 0.0,
        max_search_radius: float = ...,
        search_distance_factor: float = 0.2,
    ): ...
    def sample_test_route(self): ...
    def sample_real_route(
        self,
        min_num_nodes: int,
        max_num_nodes: Optional[int] = None,
        truncate: bool = True,
        rng: np.random.Generator | None = None,
    ): ...
