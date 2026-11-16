"""Append-only JSONL storage for collected rows.

Local mode (no HR_DATA_REPO env var): rows go to form/collected/ only.
Space mode: rows are also pushed to a private HF dataset every few minutes
with huggingface_hub.CommitScheduler, and rows already in that dataset are
downloaded at startup so assignment counts survive a Space restart.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Iterator


class Store:
    def __init__(self, local_dir: Path, repo_id: str | None = None, token: str | None = None):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        # one file per process, so restarts never rewrite an uploaded file
        self.path = self.local_dir / f"rows-{uuid.uuid4().hex}.jsonl"
        self.repo_id = repo_id
        self.scheduler = None
        self._existing_dir: Path | None = None
        self._lock = threading.Lock()
        if repo_id:
            from huggingface_hub import CommitScheduler, snapshot_download

            self._existing_dir = self.local_dir.parent / "_existing"
            try:
                snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    local_dir=self._existing_dir,
                    allow_patterns=["rows/*.jsonl"],
                    token=token,
                )
            except Exception as e:  # repo may not exist yet on first start
                print(f"no existing rows downloaded: {e!r}")
            self.scheduler = CommitScheduler(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=self.local_dir,
                path_in_repo="rows",
                every=2,  # minutes
                private=True,
                token=token,
            )

    def append(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        lock = self.scheduler.lock if self.scheduler else self._lock
        with lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line)

    def all_rows(self) -> Iterator[dict]:
        """Rows from earlier runs (downloaded) and from local files."""
        dirs = [self.local_dir]
        if self._existing_dir is not None:
            dirs.append(self._existing_dir / "rows")
        seen_names: set[str] = set()
        for d in dirs:
            if not d.exists():
                continue
            for p in sorted(d.glob("rows-*.jsonl")):
                if p.name in seen_names:
                    continue
                seen_names.add(p.name)
                with p.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
