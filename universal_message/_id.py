import os
import random
import struct
import threading
import time

_lock = threading.Lock()
_counter = random.randint(0, 0xFFFFFF)  # start counter at random point


def generate_object_id() -> str:
    """Generate unique 24-character hex object ID."""
    global _counter
    timestamp = struct.pack(">I", int(time.time()))  # 4 bytes, big-endian
    random5 = os.urandom(5)  # 5 cryptographically-secure random bytes
    with _lock:  # thread-safe 3-byte counter
        _counter = (_counter + 1) % 0xFFFFFF
        counter3 = struct.pack(">I", _counter)[1:]  # drop the MSB so we keep 3 bytes
    return (timestamp + random5 + counter3).hex()
