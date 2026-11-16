"""Download the form's Google Sheet, import it, and print a progress and speed check.

    python scripts/pull_sheet.py

Needs a clasp login on this machine (`npx.cmd @google/clasp@3.4.1 login`,
with the Google Drive permission ticked). It reuses that login's refresh token
from ~/.clasprc.json to export the Sheet through the Drive API; the token can
only reach files clasp created, which here is this one Sheet.

Steps: export to data/collected/sheet-latest.xlsx, convert with
scripts/import_sheet.py, then print per-annotator counts and typing speed so
bulk or automated submissions stand out.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHEET_ID = "REDACTED_SHEET_ID"
XLSX = ROOT / "data" / "collected" / "sheet-latest.xlsx"
EXPORT_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
FAST_SECONDS = 5.0  # median below this is not plausible for reading and typing


def access_token() -> str:
    rc = json.loads((Path.home() / ".clasprc.json").read_text(encoding="utf-8"))
    t = rc["tokens"]["default"]
    body = urllib.parse.urlencode({
        "client_id": t["client_id"], "client_secret": t["client_secret"],
        "refresh_token": t["refresh_token"], "grant_type": "refresh_token",
    }).encode()
    with urllib.request.urlopen("https://oauth2.googleapis.com/token", data=body) as r:
        return json.load(r)["access_token"]


def export(token: str) -> None:
    url = (f"https://www.googleapis.com/drive/v3/files/{SHEET_ID}/export?"
           + urllib.parse.urlencode({"mimeType": EXPORT_MIME}))
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    XLSX.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req) as r:
        XLSX.write_bytes(r.read())
    print(f"exported {XLSX.stat().st_size:,} bytes to {XLSX}")


def speed_report() -> None:
    path = ROOT / "data" / "collected" / "rows" / "sheet-export.jsonl"
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    by: dict[str, list[float]] = {}
    for r in rows:
        if r["type"] == "rewrite" and not r["skipped"] and r.get("seconds") is not None:
            by.setdefault(r["anon_id"], []).append(r["seconds"])
    print("\nannotator | answers | median seconds | flag")
    for aid, s in sorted(by.items(), key=lambda x: -len(x[1])):
        med = statistics.median(s)
        flag = "FAST: check before use" if med < FAST_SECONDS else ""
        print(f"{aid[:8]} | {len(s)} | {med:.1f} | {flag}")


def main() -> None:
    export(access_token())
    py = sys.executable
    subprocess.run([py, str(ROOT / "scripts" / "import_sheet.py"), str(XLSX)], check=True)
    subprocess.run([py, str(ROOT / "scripts" / "pull_rewrites.py")], check=True)
    speed_report()


if __name__ == "__main__":
    main()
