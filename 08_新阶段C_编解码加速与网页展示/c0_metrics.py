"""Non-overlapping wall-clock scopes for the C0 reference measurement harness."""
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter


class Timings:
    def __init__(self, clock=perf_counter):
        self.clock = clock
        self.stack = []
        self.inclusive = defaultdict(float)
        self.exclusive = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def scope(self, name):
        entry = [name, self.clock(), 0.0]
        self.stack.append(entry)
        try:
            yield
        finally:
            elapsed = self.clock() - entry[1]
            assert self.stack.pop() is entry
            self.inclusive[name] += elapsed
            self.exclusive[name] += elapsed - entry[2]
            self.counts[name] += 1
            if self.stack:
                self.stack[-1][2] += elapsed

    def snapshot(self):
        return {name: {"inclusive_seconds": self.inclusive[name],
                       "exclusive_seconds": self.exclusive[name],
                       "calls": self.counts[name]}
                for name in sorted(self.counts)}


def distribution(values):
    from statistics import median
    if not values:
        return None
    return {"n": len(values), "median": median(values), "min": min(values),
            "max": max(values), "values": values}
