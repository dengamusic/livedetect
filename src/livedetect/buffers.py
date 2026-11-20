"""Hand-off points between threads.

Both classes drop old data instead of queueing it. A real-time pipeline that
queues falls further behind the camera with every frame it cannot keep up with;
one that drops stays at most one frame behind.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    """A one-item mailbox: the writer never blocks, the reader gets the newest item."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._full = False
        self._closed = False

    def put(self, item: T) -> bool:
        """Store `item`. Returns True if it replaced an item nobody had read."""
        with self._cond:
            dropped = self._full
            self._item, self._full = item, True
            self._cond.notify()
            return dropped

    def get(self, timeout: float | None = None) -> T | None:
        """Wait for an item. Returns None on timeout or once the slot is closed."""
        with self._cond:
            self._cond.wait_for(lambda: self._full or self._closed, timeout)
            if not self._full:
                return None
            item, self._item, self._full = self._item, None, False
            return item

    def clear(self) -> int:
        """Discard what is waiting. Returns how many items that was."""
        with self._cond:
            waiting, self._item, self._full = int(self._full), None, False
            return waiting

    def close(self) -> None:
        """Wake every waiting reader; later `get` calls return None at once."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def size(self) -> int:
        """Items waiting to be read: 0 or 1."""
        return int(self._full)

    @property
    def closed(self) -> bool:
        return self._closed


class FifoQueue(Generic[T]):
    """A bounded first-in-first-out queue with the interface of `LatestSlot`.

    This is the usual way to join two threads, and the wrong one here: every
    item waits behind all the earlier ones, so a reader that is slower than
    the writer serves results that are `capacity` processing times old. When
    it is full, new items are turned away. The pipeline does not use it unless
    told to; it exists so the two designs can be compared on a live stream.
    """

    def __init__(self, capacity: int) -> None:
        self._cond = threading.Condition()
        self._items: deque[T] = deque()
        self._capacity = capacity
        self._closed = False

    def put(self, item: T) -> bool:
        """Append `item`. Returns True if the queue was full and the item was discarded."""
        with self._cond:
            if len(self._items) >= self._capacity:
                return True
            self._items.append(item)
            self._cond.notify()
            return False

    def get(self, timeout: float | None = None) -> T | None:
        """Wait for the oldest item. Returns None on timeout or once the queue is closed."""
        with self._cond:
            self._cond.wait_for(lambda: self._items or self._closed, timeout)
            return self._items.popleft() if self._items else None

    def clear(self) -> int:
        """Discard what is waiting. Returns how many items that was."""
        with self._cond:
            waiting = len(self._items)
            self._items.clear()
            return waiting

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def closed(self) -> bool:
        return self._closed


class Broadcast(Generic[T]):
    """Fan one stream of items out to any number of subscribers.

    Each subscriber has its own LatestSlot, so a slow one (a browser on a bad
    connection) skips items and never holds up the publisher or the others.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: set[LatestSlot[T]] = set()
        self._latest: T | None = None
        self._closed = False

    def publish(self, item: T) -> None:
        with self._lock:
            self._latest = item
            slots = list(self._slots)
        for slot in slots:
            slot.put(item)

    def subscribe(self) -> LatestSlot[T]:
        slot: LatestSlot[T] = LatestSlot()
        with self._lock:
            if self._closed:
                slot.close()
                return slot
            self._slots.add(slot)
            if self._latest is not None:
                slot.put(self._latest)  # a new viewer sees the current state at once
        return slot

    def unsubscribe(self, slot: LatestSlot[T]) -> None:
        with self._lock:
            self._slots.discard(slot)

    @contextmanager
    def subscription(self) -> Iterator[LatestSlot[T]]:
        slot = self.subscribe()
        try:
            yield slot
        finally:
            self.unsubscribe(slot)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            slots, self._slots = list(self._slots), set()
        for slot in slots:
            slot.close()

    @property
    def latest(self) -> T | None:
        return self._latest

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._slots)
