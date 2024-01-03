import math
import os
import re
import urllib.request
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

# def plot_route(stops, search_area, candidates):
#     wgs84 = pyproj.CRS("EPSG:4326")
#     utm = pyproj.CRS("EPSG:3414")

#     project = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True).transform

#     stops = stops.to_crs("EPSG:4326")
#     #     candidates = candidates.to_crs('EPSG:4326')

#     lon, lat = stops.dissolve().centroid.iloc[0].xy
#     m = folium.Map(location=[lat[0], lon[0]], tiles="cartodbpositron", zoom_start=12)

#     folium.PolyLine(
#         list(zip(stops.geometry.y.tolist(), stops.geometry.x.tolist())), c="blue"
#     ).add_to(m)

#     for lat, lon in stops[["Latitude", "Longitude"]].values:
#         folium.CircleMarker(
#             location=[lat, lon],
#             radius=4,
#             stroke=True,
#             weight=2.5,
#             fill=True,
#             fill_color="#ffffff",
#             fill_opacity=1,
#         ).add_to(m)

#     x, y = transform(project, search_area).exterior.xy
#     folium.Polygon(list(zip(y, x)), fill=True, stroke=False, fill_opacity=0.5).add_to(m)

#     for lat, lon in candidates[["Latitude", "Longitude"]].values:
#         folium.CircleMarker(
#             location=[lat, lon],
#             radius=3,
#             color="#000000",
#             stroke=True,
#             weight=1,
#             fill=True,
#             fill_color="#ff0000",
#             fill_opacity=1,
#         ).add_to(m)

#     display(m)


def idx_max(lst, skip=0):
    assert skip < len(lst)
    return list(sorted(enumerate(lst), key=lambda x: x[1]))[-1 - skip][0]


class SGNetwork:
    _STOPS_TO_IGNORE = ["46211", "46219", "46239", "47711"]
    _ROUTES_TO_IGNORE = [
        "160",
        "170",
        "170X",
        "950",
        "101",
        "102",
        "107M",
        "11",
        "110",
        "111",
        "112",
        "113",
        "115",
        "116",
        "119",
        "120",
        "121",
        "122",
        "123M",
        "125",
        "127",
        "134",
        "138",
        "140",
        "142",
        "143M",
        "15",
        "150",
        "158",
        "160",
        "160M",
        "162M",
        "171",
        "173",
        "177",
        "179",
        "179A",
        "180",
        "181",
        "181M",
        "182",
        "182M",
        "183",
        "184",
        "189",
        "191",
        "194",
        "195",
        "199",
        "20",
        "200",
        "201",
        "222",
        "225G",
        "225W",
        "228",
        "229",
        "23",
        "231",
        "232",
        "235",
        "238",
        "24",
        "240",
        "241",
        "242",
        "243G",
        "243W",
        "246",
        "247",
        "248",
        "248M",
        "249",
        "251",
        "252",
        "253",
        "254",
        "255",
        "257",
        "258",
        "261",
        "262",
        "265",
        "269",
        "27",
        "272",
        "273",
        "282",
        "284",
        "285",
        "29",
        "291",
        "292",
        "293",
        "298",
        "300",
        "301",
        "302",
        "307",
        "315",
        "317",
        "324",
        "325",
        "329",
        "333",
        "334",
        "335",
        "34",
        "35",
        "354",
        "358",
        "359",
        "35M",
        "36",
        "37",
        "371",
        "372",
        "374",
        "381",
        "382",
        "382G",
        "382W",
        "384",
        "386",
        "40",
        "400",
        "401",
        "403",
        "405",
        "41",
        "410G",
        "410W",
        "42",
        "43M",
        "47",
        "49",
        "502",
        "518",
        "53",
        "60",
        "62",
        "63",
        "63M",
        "64",
        "66",
        "68",
        "69",
        "70M",
        "71",
        "73",
        "79",
        "800",
        "801",
        "803",
        "804",
        "805",
        "806",
        "807",
        "81",
        "811",
        "811T",
        "812",
        "812T",
        "82",
        "825",
        "83",
        "84",
        "857",
        "858",
        "859",
        "859A",
        "859B",
        "860",
        "882",
        "883",
        "883M",
        "89",
        "90",
        "900",
        "900A",
        "901",
        "901M",
        "903",
        "903M",
        "904",
        "91",
        "911",
        "912",
        "913",
        "92",
        "920",
        "922",
        "925M",
        "927",
        "92M",
        "94",
        "941",
        "944",
        "945",
        "947",
        "95",
        "950",
        "96",
        "962",
        "964",
        "965",
        "966",
        "972",
        "972M",
        "973",
        "974",
        "975",
        "979",
        "98",
        "983",
        "98M",
        "990",
    ]
    _WGS84 = "EPSG:4326"
    _SVY21 = "EPSG:3414"
    _BUS_STOPS_CSV_FN = "singapore_stops_grouped.csv"
    _BUS_ROUTES_CSV_FN = "singapore_routes.csv"
    _TTMAT_FN = "ttmat.npy"

    _TERMINAL_KW = ["INT", "TER"]
    _TERMINAL_REGEX_PATTERN = (
        "^(?!OPP|BEF|AFT)(?!.*FERRY|.*CARGO|.*LARKIN|.*JB|.*MARINE).* (?:"
        + "|".join(_TERMINAL_KW)
        + ")$"
    )
    _TERMINAL_PATTERN = (
        "^((?!(FERRY|POLICE|CARGO|CALTEX|FIRE|POWER|MARINE|RAILWAY)).)* "
        "(STN|INT|TER)((\/| EXIT).*|$)"
    )
    _EXTRACT_PATTERN = "((?:(?<!(?:OPP|BEF|AFT))[^/])*(?:STN|TER|INT))"

    def __init__(self, path_to_data: str = ".rides_sgnetwork") -> None:
        self.path_to_data = path_to_data

        if not os.path.exists(self.path_to_data):
            os.mkdir(self.path_to_data)
            print(f"Downloading {self.path_to_data}{os.path.sep}singapore_routes.csv")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/rides-env/singapore_routes.csv",
                f"{self.path_to_data}{os.path.sep}singapore_routes.csv",
            )
            print(
                f"Downloading {self.path_to_data}{os.path.sep}singapore_stops_grouped.csv"
            )
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/rides-env/singapore_stops_grouped.csv",
                f"{self.path_to_data}{os.path.sep}singapore_stops_grouped.csv",
            )
            print(f"Downloading {self.path_to_data}{os.path.sep}ttmat.npy")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/rides-env/ttmat.npy",
                f"{self.path_to_data}{os.path.sep}ttmat.npy",
            )

        self._load_bus_stops()
        self._load_bus_routes()
        self._load_ttmat()

        self._find_terminals()
        # self._group_bus_stops()

    def _load_bus_stops(self) -> None:
        df = pd.read_csv(os.path.join(self.path_to_data, SGNetwork._BUS_STOPS_CSV_FN))

        self._bus_stops = (
            gpd.GeoDataFrame(
                df,
                crs=SGNetwork._WGS84,
                geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
            )
            .to_crs(SGNetwork._SVY21)[
                lambda x: ~x.BusStopCode.isin(SGNetwork._STOPS_TO_IGNORE)
            ]
            .set_index("BusStopCode")
        )

    def _load_bus_routes(self) -> None:
        self._bus_routes = pd.read_csv(
            os.path.join(self.path_to_data, SGNetwork._BUS_ROUTES_CSV_FN)
        )
        self._bus_routes = self._bus_routes[
            lambda x: ~x.ServiceNo.isin(SGNetwork._ROUTES_TO_IGNORE)
        ]
        self._services = (
            self._bus_routes[["ServiceNo", "Direction"]]
            .drop_duplicates()
            .values.tolist()
        )

    def _load_ttmat(self) -> None:
        with open(os.path.join(self.path_to_data, SGNetwork._TTMAT_FN), "rb") as f:
            self._ttmat = np.load(f)

    def _find_terminals(self) -> None:
        self._terminals = self._bus_stops[
            lambda x: x.Description.str.contains(SGNetwork._TERMINAL_REGEX_PATTERN)
        ]

    def _group_bus_stops(self) -> None:
        bus_stops = self._bus_stops

        bus_stops = bus_stops.assign(
            Parent=lambda x: x.Description.str.extract(SGNetwork._EXTRACT_PATTERN)
        )

        terminals = bus_stops.Description.str.match(SGNetwork._TERMINAL_PATTERN)
        bus_stops.loc[~terminals, "Parent"] = np.nan

        replace_bus_int_ter = lambda x: x.str.replace(" (STN|INT|TER)$", "", regex=True)

        for i, stop in list(bus_stops[lambda x: x.Parent.isna()].iterrows()):
            if not np.isnan(stop.Parent):
                continue

            matches = bus_stops[bus_stops.within(stop.geometry.buffer(100))]

            if sum(matches.Parent.isna()) == 0:
                continue

            if len(matches) == 1:
                bus_stops.loc[i, "Parent"] = (
                    re.sub(r"^(OPP|AFT|BEF) ", "", stop.Description)
                    + " ("
                    + stop.RoadName
                    + ")"
                )
            elif sum(~matches.Parent.isna()) != 0:
                if matches.Parent.nunique() == 1:
                    bus_stops.loc[matches.index, "Parent"] = matches[
                        lambda x: ~x.Parent.isna()
                    ].Parent.iloc[0]
                else:
                    desc = replace_bus_int_ter(matches.Parent)[lambda x: ~x.isna()]

                    if desc.nunique() != 1:
                        raise ValueError

                    parent_name = (
                        desc.unique()[0]
                        + " "
                        + "/".join(
                            matches.Parent.str.replace(f"{desc.unique()[0]} ", "")[
                                lambda x: ~x.isna()
                            ]
                            .unique()
                            .tolist()
                        )
                    )
                    bus_stops.loc[matches.index, "Parent"] = parent_name
            else:
                parent_name = " / ".join(
                    matches.Description.str.replace("^(OPP|AFT|BEF)", "", regex=True)
                    .str.strip()
                    .unique()
                    .tolist()
                )

                if matches.RoadName.nunique() == 1:
                    parent_name += " (" + matches.RoadName.iloc[0] + ")"

                if parent_name in bus_stops.Parent.tolist():
                    bus_stops.loc[lambda x: x.Parent == parent_name, "Parent"] = (
                        parent_name + " [A]"
                    )
                    parent_name += " [B]"

                bus_stops.loc[matches.index, "Parent"] = parent_name

        self._bus_stops = bus_stops

    def _cache(self, route, distances, parents, circles) -> None:
        self._cached_route = route
        self._cached_distances = distances
        self._cached_parents = parents
        self._cached_circles = circles

    def _sample_terminal(self, n=1, min_distance=None, rng=None) -> List[str]:
        if rng is None:
            sampled = self._terminals.sample(n=n).index.tolist()
        else:
            idx = rng.choice(range(len(self._terminals)), replace=False, size=2)
            sampled = self._terminals.iloc[idx].index.tolist()

        if min_distance is None:
            return sampled

        assert n == 2

        if self._get_distance(*sampled) < min_distance:
            return self._sample_terminal(n=n, min_distance=min_distance, rng=rng)

        return sampled

    def _get_distance(self, from_: Union[str, List[str]], to_: Union[str, List[str]]):
        if isinstance(from_, list):
            from_ = self._bus_stops.loc[from_]["index"].tolist()
        else:
            from_ = self._bus_stops.loc[from_]["index"]

        if isinstance(to_, list):
            to_ = self._bus_stops.loc[to_]["index"].tolist()
        else:
            to_ = self._bus_stops.loc[to_]["index"]

        return self._ttmat[from_, to_]

    def _calc_search_radius(
        self,
        distance: float,
        max_search_radius=float("inf"),
        search_distance_factor=0.2,
    ) -> float:
        if math.isinf(max_search_radius):
            return distance * search_distance_factor

    def _get_search_area(
        self,
        from_: str,
        to_: str,
        max_search_radius=float("inf"),
        search_distance_factor=0.2,
    ) -> shapely.geometry.Polygon:
        distance = self._get_distance(from_, to_)

        center = (
            self._bus_stops.loc[from_]
            .geometry.buffer(distance / 2)
            .buffer(1)
            .intersection(
                self._bus_stops.loc[to_].geometry.buffer(distance / 2).buffer(1)
            )
        )  # Buffer 1m for cases where two stops are on the same road, i.e. intersection = ø

        search_radius = self._calc_search_radius(
            distance,
            max_search_radius=max_search_radius,
            search_distance_factor=search_distance_factor,
        )
        search_area = center.buffer(search_radius)

        return search_area

    def _search_candidates(
        self,
        pos: int,
        from_: str,
        to_: str,
        plot: bool = True,
        max_path_detour=1.2,
        max_link_detour=1.3,
        max_search_radius=float("inf"),
        search_distance_factor=0.2,
    ) -> gpd.GeoDataFrame:
        search_area = self._get_search_area(
            from_,
            to_,
            max_search_radius=max_search_radius,
            search_distance_factor=search_distance_factor,
        )
        # for i, circle in enumerate(self._cached_circles):
        #     if i != pos:
        #         search_area = search_area - circle

        candidates = (
            self._bus_stops.loc[~self._bus_stops.index.isin(self._cached_route)][
                lambda x: x.within(search_area)
            ][lambda x: ~x["Parent"].isin(self._cached_parents)]
        ).assign(
            straight_line_dist1=lambda x: x.distance(
                self._bus_stops.loc[from_].geometry
            ),
            straight_line_dist2=lambda x: x.distance(self._bus_stops.loc[to_].geometry),
            actual_dist1=lambda x: self._get_distance(from_, x.index.tolist()),
            actual_dist2=lambda x: self._get_distance(x.index.tolist(), to_),
            detour1=lambda x: x.actual_dist1 / x.straight_line_dist1,
            detour2=lambda x: x.actual_dist2 / x.straight_line_dist2,
            detour=lambda x: (x.actual_dist1 + x.actual_dist2)
            / (x.straight_line_dist1 + x.straight_line_dist2),
            prop=lambda x: x.actual_dist1 / x.actual_dist2,
        )[
            lambda x: x.detour <= max_path_detour
        ][
            lambda x: x.detour1 <= max_link_detour
        ][
            lambda x: x.detour2 <= max_link_detour
        ]

        if plot:
            plot_route(self._bus_stops.loc[self._cached_route], search_area, candidates)

        return candidates

    def _select_and_commit(
        self,
        pos: int,
        from_: str,
        to_: str,
        candidates: gpd.GeoDataFrame,
        rng=None,
        sample_candidate_threshold_distance=5000,
    ) -> None:
        if self._cached_distances[pos] >= sample_candidate_threshold_distance:
            if rng is None:
                selected = candidates.sample().iloc[0]
            else:
                idx = rng.choice(range(len(candidates)))
                selected = candidates.iloc[idx]
        else:
            candidates = candidates.sort_values(by="detour")
            selected = candidates.iloc[0]

        self._cached_circles[pos] = self._get_circle(from_, selected.name)
        self._cached_circles.insert(pos + 1, self._get_circle(selected.name, to_))

        self._cached_distances[pos] = self._get_distance(from_, selected.name)
        self._cached_distances.insert(pos + 1, self._get_distance(selected.name, to_))

        self._cached_route.insert(pos + 1, selected.name)
        self._cached_parents.append(selected["Parent"])

        return self._cached_route

    def _get_circle(self, from_, to_):
        ctr = self._bus_stops.loc[[from_, to_]].dissolve().centroid.iloc[0]
        distance = self._get_distance(from_, to_)
        shape = ctr.buffer(distance / 2)

        return shape

    def _break_longest_link(
        self,
        route,
        rng=None,
        failed_attempts=0,
        sample_candidate_threshold_distance=5000,
        max_path_detour=1.2,
        max_link_detour=1.3,
        max_search_radius=float("inf"),
        search_distance_factor=0.2,
    ) -> List[str]:
        if "_cached_route" not in self.__dict__ or route != self._cached_route:
            distances = [
                self._get_distance(*link) for link in zip(route[:-1], route[1:])
            ]
            parents = self._bus_stops.loc[route].Parent.unique().tolist()
            circles = [self._get_circle(*route)]
            self._cache(route, distances, parents, circles)

        while True:
            # try:
            pos = idx_max(self._cached_distances, skip=failed_attempts)
            from_, to_ = self._cached_route[pos], self._cached_route[pos + 1]
            # except:
            #     raise RuntimeError

            candidates = self._search_candidates(
                pos,
                from_,
                to_,
                max_path_detour=max_path_detour,
                max_link_detour=max_link_detour,
                max_search_radius=max_search_radius,
                search_distance_factor=search_distance_factor,
            )

            if len(candidates) == 0:
                failed_attempts += 1
                continue

            return (
                self._select_and_commit(
                    pos,
                    from_,
                    to_,
                    candidates,
                    rng=rng,
                    sample_candidate_threshold_distance=sample_candidate_threshold_distance,
                ),
                failed_attempts,
            )

    def _get_ttmat(self, route):
        idx = self._bus_stops.loc[route]["index"].tolist()
        return np.triu(self._ttmat[idx][:, idx], 1)

    def generate_gmap_link(self, route):
        url = "https://www.google.com/maps/dir/"

        for lat, lon in self._bus_stops.loc[route][["Latitude", "Longitude"]].values:
            url += f"{lat},{lon}/"

        return url

    def generate_trunk_route(
        self,
        min_num_nodes: int,
        max_num_nodes: Optional[int] = None,
        rng=None,
        min_terminal_distance: float = 5000.0,
        sample_candidate_threshold_distance: float = 5000.0,
        max_path_detour: float = 1.2,
        max_link_detour: float = 1.3,
        min_search_radius: float = 0.0,
        max_search_radius: float = float("inf"),
        search_distance_factor: float = 0.2,
    ):
        if max_num_nodes is None:
            max_num_nodes = min_num_nodes

        if rng is not None:
            target_num_nodes = rng.integers(min_num_nodes, max_num_nodes, endpoint=True)
        else:
            target_num_nodes = np.random.randint(min_num_nodes, max_num_nodes + 1)

        route = self._sample_terminal(n=2, min_distance=min_terminal_distance, rng=rng)

        failed_attempts = 0
        while len(route) < target_num_nodes:
            # try:
            route, failed_attempts = self._break_longest_link(
                route,
                rng=rng,
                failed_attempts=failed_attempts,
                sample_candidate_threshold_distance=sample_candidate_threshold_distance,
                max_path_detour=max_path_detour,
                max_link_detour=max_link_detour,
                max_search_radius=max_search_radius,
                search_distance_factor=search_distance_factor,
            )
            # except:
            #     break

        if len(route) > min_num_nodes:
            return route, self._get_ttmat(route), {}

        print("Retrying...")
        return self.generate_trunk_route(min_num_nodes, max_num_nodes, rng=rng)

    def sample_test_route(self):
        should_truncate = False

        route = ["A", "B", "C"]
        ttmat = np.asarray([[0, 850, 850], [0, 0, 1700], [0, 0, 0]])
        service = "TEST"
        direction = "A"

        return (
            route,
            ttmat,
            {
                "service": service,
                "direction": direction,
                "name": f"{service}-D{direction}"
                + (" (Truncated)" if should_truncate else ""),
                "truncated": should_truncate,
            },
        )

    def sample_real_route(
        self,
        min_num_nodes: int,
        max_num_nodes: Optional[int] = None,
        truncate: bool = True,
        rng=None,
    ):
        cannot_truncate = not truncate

        if max_num_nodes is None:
            max_num_nodes = min_num_nodes

        if rng is not None:
            target_num_nodes = rng.integers(min_num_nodes, max_num_nodes, endpoint=True)
        else:
            target_num_nodes = np.random.randint(min_num_nodes, max_num_nodes + 1)

        if rng is None:
            idx = np.random.choice(range(len(self._services)))
        else:
            idx = rng.choice(range(len(self._services)))

        service, direction = self._services[idx]
        route = (
            self._bus_routes[lambda x: x.ServiceNo == service][
                lambda x: x.Direction == direction
            ]
            .sort_values(by="StopSequence")
            .BusStopCode.tolist()
        )

        should_truncate = len(route) > target_num_nodes

        if len(route) < min_num_nodes or (should_truncate and cannot_truncate):
            return self.sample_real_route(
                min_num_nodes, max_num_nodes=max_num_nodes, rng=rng, truncate=truncate
            )

        if should_truncate:
            if rng is None:
                idx = np.random.choice(
                    range(len(route)), target_num_nodes, replace=False
                )
            else:
                idx = rng.choice(range(len(route)), target_num_nodes, replace=False)

            route = [route[i] for i in sorted(idx)]

        return (
            route,
            self._get_ttmat(route),
            {
                "service": service,
                "direction": direction,
                "name": f"{service}-D{direction}"
                + (" (Truncated)" if should_truncate else ""),
                "truncated": should_truncate,
            },
        )
