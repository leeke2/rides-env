from dataclasses import dataclass


@dataclass
class StopSequence:
    stops: list[int]

    def __len__(self) -> int:
        return self.stops.__len__()

    def __hash__(self) -> int:
        return tuple(self.stops).__hash__()

    def __getitem__(self, index):
        return self.stops.__getitem__(index)


class Service:
    def __init__(self, nstops: int, nbuses: int):
        self.stops_binary = [False for stop in range(nstops)]
        self.nbuses = nbuses
        self.nstops = nstops

        self.last_stop = -1

    @property
    def stops(self) -> StopSequence:
        return StopSequence([i for i, served in enumerate(self.stops_binary) if served])

    def is_valid(self) -> bool:
        return sum(self.stops_binary) >= 2 and self.nbuses >= 1

    def is_serving(self, stop: int) -> bool:
        return self.stops_binary[stop]

    def not_serving_any_stops(self) -> bool:
        return sum(self.stops_binary) == 0

    def add_bus(self) -> None:
        self.nbuses += 1

    def toggle(self, stop: int) -> None:
        self.stops_binary[stop] = False if self.stops_binary[stop] else True

        if stop >= self.last_stop:
            if self.stops_binary[stop]:
                self.last_stop = stop
            elif sum(self.stops_binary) == 0:
                self.last_stop = -1
            else:
                for i in range(self.last_stop, -1, -1):
                    if self.stops_binary[i]:
                        self.last_stop = i
                        break
