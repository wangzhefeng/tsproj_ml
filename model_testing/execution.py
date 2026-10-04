"""Ordered, bounded fit scheduling; scoring stays on the consuming thread."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ordered_bounded_map(function: Callable[[T], R], items: Iterable[T], *, workers: int) -> Iterator[R]:
    """Keep at most workers unconsumed tasks; propagate failures and cancel queued work."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        for item in items:
            yield function(item)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        try:
            for _ in range(workers):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.append(executor.submit(function, item))
            while pending:
                future = pending.popleft()
                result = future.result()
                del future
                yield result
                del result
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                pending.append(executor.submit(function, item))
        finally:
            for future in pending:
                future.cancel()
