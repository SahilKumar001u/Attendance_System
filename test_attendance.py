"""Self-check: python test_attendance.py"""
import tempfile
from pathlib import Path

import numpy as np

import attendance as a

# mark() logs each student once
log = Path(tempfile.mkdtemp()) / "log.csv"
assert a.mark("Asha", log) and a.mark("Ravi", log)
assert not a.mark("Asha", log)
assert log.read_text().splitlines()[0] == "name,date,time"
assert len(log.read_text().splitlines()) == 3

# identify() accepts a close match and rejects a stranger
embs = np.eye(3)
assert a.identify(np.array([1.0, 0, 0]), ["A", "B", "C"], embs)[0] == "A"
weak = np.array([0.3, 0.3, 0.9])
assert a.identify(weak / np.linalg.norm(weak) * 0.3, ["A", "B", "C"], embs)[0] is None

print("ok")
