"""Bounded replicas: lease an endpoint for one request, always release on failure."""

from contextlib import contextmanager
from queue import Queue


class EndpointPool:
    def __init__(self, values):
        if not values or len(set(values)) != len(values):
            raise ValueError("endpoint list must be nonempty and unique")
        self.available = Queue()
        for value in values:
            self.available.put(value)

    @contextmanager
    def lease(self):
        value = self.available.get()
        try:
            yield value
        finally:
            self.available.put(value)


class PooledClient:
    def __init__(self, clients):
        self.clients = clients
        self.pool = EndpointPool(list(clients))

    def chat(self, **kwargs):
        with self.pool.lease() as url:
            response = self.clients[url].chat(**kwargs)
            return {**response, "endpoint": url}

    @property
    def enforces_context_window(self):
        return all(getattr(c, "enforces_context_window", False) for c in self.clients.values())

    def inspect_context(self, **kwargs):
        if not self.enforces_context_window:
            raise RuntimeError("exact context inspection is unavailable")
        with self.pool.lease() as url:
            return {**self.clients[url].inspect_context(**kwargs), "endpoint": url}
