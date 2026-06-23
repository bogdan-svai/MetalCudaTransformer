from contextlib import contextmanager
from time import perf_counter


class BuildTimings:
    def __init__(self):
        self._started = perf_counter()
        self._phases = {}
        self.cache_events = []

    @contextmanager
    def phase(self, name):
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            self._phases[name] = self._phases.get(name, 0.0) + elapsed

    def cache(self, layer, hit, key=None):
        event = {"layer": layer, "hit": bool(hit)}
        if key:
            event["key"] = key
        self.cache_events.append(event)

    def to_dict(self):
        phases = {
            name: round(seconds, 6)
            for name, seconds in sorted(self._phases.items())
        }
        phases["total"] = round(perf_counter() - self._started, 6)
        return {"phases_seconds": phases, "cache_events": list(self.cache_events)}
