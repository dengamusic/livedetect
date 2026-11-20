import threading
import time

from livedetect.buffers import Broadcast, LatestSlot


def test_slot_returns_what_was_put():
    slot = LatestSlot()
    assert slot.put("a") is False
    assert slot.get(timeout=0) == "a"


def test_slot_keeps_only_the_newest_item_and_reports_the_drop():
    slot = LatestSlot()
    slot.put("old")
    assert slot.put("new") is True
    assert slot.get(timeout=0) == "new"
    assert slot.get(timeout=0) is None


def test_slot_get_times_out_with_none():
    started = time.monotonic()
    assert LatestSlot().get(timeout=0.05) is None
    assert time.monotonic() - started >= 0.04


def test_slot_put_wakes_a_waiting_reader():
    slot = LatestSlot()
    received = []
    reader = threading.Thread(target=lambda: received.append(slot.get(timeout=2)))
    reader.start()
    slot.put(42)
    reader.join(timeout=2)
    assert received == [42]


def test_slot_close_wakes_a_waiting_reader():
    slot = LatestSlot()
    received = []
    reader = threading.Thread(target=lambda: received.append(slot.get(timeout=5)))
    reader.start()
    slot.close()
    reader.join(timeout=2)
    assert received == [None]
    assert slot.closed


def test_broadcast_delivers_to_every_subscriber():
    hub = Broadcast()
    a, b = hub.subscribe(), hub.subscribe()
    hub.publish("x")
    assert a.get(timeout=0) == "x"
    assert b.get(timeout=0) == "x"


def test_broadcast_slow_subscriber_skips_ahead_without_blocking_the_publisher():
    hub = Broadcast()
    slow = hub.subscribe()
    for i in range(1000):  # nobody is reading; this must not block or grow
        hub.publish(i)
    assert slow.get(timeout=0) == 999


def test_broadcast_new_subscriber_gets_the_latest_item_at_once():
    hub = Broadcast()
    hub.publish("current")
    assert hub.subscribe().get(timeout=0) == "current"
    assert hub.latest == "current"


def test_broadcast_subscription_cleans_up_after_itself():
    hub = Broadcast()
    with hub.subscription():
        assert hub.subscriber_count == 1
    assert hub.subscriber_count == 0


def test_broadcast_close_releases_subscribers():
    hub = Broadcast()
    slot = hub.subscribe()
    hub.close()
    assert slot.closed
    assert hub.subscribe().closed
