"""Decide which source query an annotator sees next.

Rules:
  * An annotator never sees a query twice (rewritten or skipped).
  * Prefer queries that are closest to their target count, so pairs of
    rewrites complete early (the go/no-go gate needs ~200 complete pairs).
    Ties go to the fixed random `priority` from select_sources.py.
  * Target is 2 rewrites per query from 2 different annotators. Once every
    live query has 2, the cap rises to 3.
  * A query skipped by RETIRE_AFTER_SKIPS different people is retired:
    it is probably unclear or broken in the source data.
  * A query on screen for someone is "reserved" for a short time, so two
    people working at once do not both get the last slot.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass

TARGET = 2
MAX_CAP = 3
RETIRE_AFTER_SKIPS = 3
RESERVE_SECONDS = 15 * 60


@dataclass(frozen=True)
class Source:
    source_id: str
    lang: str
    role: str
    text: str
    priority: int


class Assigner:
    def __init__(self, sources: list[Source], clock=time.monotonic):
        self.sources = {s.source_id: s for s in sources}
        self.done_by: dict[str, set[str]] = defaultdict(set)  # anon_id -> source ids
        self.rewrites: dict[str, set[str]] = defaultdict(set)  # source -> anon ids
        self.skips: dict[str, set[str]] = defaultdict(set)  # source -> anon ids
        self.reserved: dict[str, dict[str, float]] = defaultdict(dict)  # source -> {anon: t}
        self._lock = threading.Lock()
        self._clock = clock

    # ---- state updates ----
    def record(self, anon_id: str, source_id: str, skipped: bool) -> None:
        with self._lock:
            self.done_by[anon_id].add(source_id)
            (self.skips if skipped else self.rewrites)[source_id].add(anon_id)
            self.reserved[source_id].pop(anon_id, None)

    def retired(self, source_id: str) -> bool:
        return len(self.skips[source_id]) >= RETIRE_AFTER_SKIPS

    def _live_reservations(self, source_id: str) -> int:
        now = self._clock()
        res = self.reserved[source_id]
        for a, t in list(res.items()):
            if now - t > RESERVE_SECONDS:
                del res[a]
        return len(res)

    def _load(self, source_id: str) -> int:
        return len(self.rewrites[source_id]) + self._live_reservations(source_id)

    # ---- assignment ----
    def next_for(self, anon_id: str) -> Source | None:
        with self._lock:
            # A query already reserved for this person is shown again (page reload).
            for sid, res in self.reserved.items():
                if anon_id in res and sid not in self.done_by[anon_id]:
                    return self.sources[sid]

            live = [s for s in self.sources.values() if not self.retired(s.source_id)]
            cap = TARGET
            if live and all(len(self.rewrites[s.source_id]) >= TARGET for s in live):
                cap = MAX_CAP

            best: Source | None = None
            best_key = None
            for s in live:
                if s.source_id in self.done_by[anon_id]:
                    continue
                load = self._load(s.source_id)
                if load >= cap:
                    continue
                key = (-load, s.priority)  # fuller first, then fixed priority
                if best_key is None or key < best_key:
                    best, best_key = s, key
            if best is not None:
                self.reserved[best.source_id][anon_id] = self._clock()
            return best

    # ---- reporting ----
    def complete_pairs(self, role: str | None = None) -> int:
        return sum(
            1
            for s in self.sources.values()
            if (role is None or s.role == role) and len(self.rewrites[s.source_id]) >= TARGET
        )

    def total_done(self, anon_id: str) -> int:
        return len(self.done_by[anon_id])


def load_sources(path) -> list[Source]:
    import json

    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append(Source(r["source_id"], r["lang"], r["role"], r["text"], r["priority"]))
    return out
