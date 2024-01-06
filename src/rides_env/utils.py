import numpy as np
import numpy.typing as npt


def calculate_stats(
    arr: npt.ArrayLike, sum: bool = True
) -> tuple[np.floating, np.floating, np.floating, np.floating]:
    arr = np.array(arr)

    values = (
        arr[np.triu_indices_from(arr, k=1)]
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]
        else arr
    )

    return (
        np.min(values),
        np.mean(values),
        np.max(values),
        np.sum(values) if sum else float("nan"),
    )


def trip_time(travel_time: npt.NDArray, stops: list[int]) -> float:
    return sum(travel_time[from_][to_] for from_, to_ in zip(stops[:-1], stops[1:]))
