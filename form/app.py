"""Hinglish query collection form (Gradio).

Shows one MIRACL source query at a time and asks the person to type it the
way they would type it to a friend or in a search bar. Nothing personal is
collected: a random id lives in the browser's local storage.

Run locally:   python form/app.py
On a Space:    set HR_DATA_REPO (private dataset id) and HF_TOKEN (secret).
"""

from __future__ import annotations

import html
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

from assign import Assigner, load_sources
from store import Store

APP_VERSION = "0.1.0"
CONSENT_VERSION = "2026-11-12"
HERE = Path(__file__).resolve().parent
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

SOURCES = load_sources(HERE / "sources.jsonl")
STORE = Store(
    Path(os.environ.get("HR_LOCAL_DIR", HERE / "collected")),
    repo_id=os.environ.get("HR_DATA_REPO") or None,
    token=os.environ.get("HF_TOKEN") or None,
)
ASSIGNER = Assigner(SOURCES)
for _row in STORE.all_rows():
    if _row.get("type") == "rewrite":
        ASSIGNER.record(_row["anon_id"], _row["source_id"], bool(_row.get("skipped")))

INSTR_HI = (
    "**Type this the way you would type it to a friend or in a search bar.** "
    "Use English letters. There is no right spelling."
)
INSTR_EN = (
    "**How would you type this in Hinglish, to a friend or in a search bar?** "
    "Use English letters and mix Hindi and English the way you normally do."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _next_view(anon_id: str, status: str = ""):
    """Outputs: cur, progress, instr, source, box, status, task_col, done_col."""
    s = ASSIGNER.next_for(anon_id)
    n = ASSIGNER.total_done(anon_id)
    progress = f"Done so far: **{n}**. Stop whenever you like, thank you!"
    if s is None:
        return (
            None, progress, "", "", gr.update(value=""), "",
            gr.update(visible=False), gr.update(visible=True),
        )
    cur = {"source_id": s.source_id, "shown_at": _now()}
    instr = INSTR_HI if s.lang == "hi" else INSTR_EN
    return (
        cur, progress, instr, f"<div class='src' lang='{s.lang}'>{html.escape(s.text)}</div>",
        gr.update(value=""), status, gr.update(visible=True), gr.update(visible=False),
    )


def on_load(bs):
    bs = dict(bs or {})
    if not bs.get("anon_id"):
        bs["anon_id"] = uuid.uuid4().hex
    if bs.get("consented"):
        return (bs, gr.update(visible=False), *_next_view(bs["anon_id"]))
    return (
        bs, gr.update(visible=True), None, "", "", "", gr.update(value=""), "",
        gr.update(visible=False), gr.update(visible=False),
    )


def on_start(bs, freq, fluency):
    bs = dict(bs or {})
    if not bs.get("anon_id"):
        bs["anon_id"] = uuid.uuid4().hex
    bs["consented"] = True
    STORE.append(
        {
            "type": "consent",
            "anon_id": bs["anon_id"],
            "consent_version": CONSENT_VERSION,
            "hindi_latin_typing_freq": freq,
            "devanagari_reading": fluency,
            "at": _now(),
            "app_version": APP_VERSION,
        }
    )
    return (bs, gr.update(visible=False), *_next_view(bs["anon_id"]))


def _record(bs, cur, text: str, skipped: bool):
    anon = bs["anon_id"]
    STORE.append(
        {
            "type": "rewrite",
            "anon_id": anon,
            "source_id": cur["source_id"],
            "rewrite_raw": None if skipped else text,
            "skipped": skipped,
            "shown_at": cur["shown_at"],
            "submitted_at": _now(),
            "app_version": APP_VERSION,
        }
    )
    ASSIGNER.record(anon, cur["source_id"], skipped)


def on_submit(bs, cur, text):
    if not cur:
        return _next_view(bs["anon_id"])
    if not text or not text.strip():
        return (cur, gr.skip(), gr.skip(), gr.skip(), gr.update(value=text),
                "Type something, or press Skip.", gr.skip(), gr.skip())
    if DEVANAGARI.search(text):
        return (cur, gr.skip(), gr.skip(), gr.skip(), gr.update(value=text),
                "Please use English letters (a to z) only.", gr.skip(), gr.skip())
    _record(bs, cur, text, skipped=False)  # stored exactly as typed
    return _next_view(bs["anon_id"])


def on_skip(bs, cur):
    if cur:
        _record(bs, cur, "", skipped=True)
    return _next_view(bs["anon_id"])


CSS = """
.src { font-size: 1.6rem; line-height: 1.5; padding: 0.75rem 0; }
"""

with gr.Blocks(title="Hinglish queries", analytics_enabled=False) as demo:
    browser = gr.BrowserState({"anon_id": None, "consented": False}, storage_key="hinglish_form_v1")
    cur = gr.State(None)

    with gr.Column(visible=False) as consent_col:
        gr.Markdown((HERE / "consent.md").read_text(encoding="utf-8"))
        freq = gr.Radio(
            ["Daily", "Sometimes", "Rarely", "Never"],
            label="Optional: how often do you type Hindi in English letters?",
        )
        fluency = gr.Radio(
            ["Easily", "Slowly", "Not really"],
            label="Optional: can you read Hindi in Devanagari script?",
        )
        start = gr.Button("I agree, start", variant="primary")

    with gr.Column(visible=False) as task_col:
        progress = gr.Markdown()
        instr = gr.Markdown()
        source = gr.HTML()
        box = gr.Textbox(
            label="Your version",
            placeholder="Type here in English letters",
            lines=1,
            max_lines=3,
            html_attributes=gr.InputHTMLAttributes(
                autocapitalize="off", autocorrect="off", spellcheck=False, autocomplete="off"
            ),
        )
        status = gr.Markdown()
        with gr.Row():
            submit = gr.Button("Submit", variant="primary")
            skip = gr.Button("Skip this one")

    with gr.Column(visible=False) as done_col:
        gr.Markdown("### That is everything for now. Thank you for helping!")

    view = [cur, progress, instr, source, box, status, task_col, done_col]
    demo.load(on_load, [browser], [browser, consent_col, *view])
    start.click(on_start, [browser, freq, fluency], [browser, consent_col, *view])
    submit.click(on_submit, [browser, cur, box], view)
    box.submit(on_submit, [browser, cur, box], view)
    skip.click(on_skip, [browser, cur], view)

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        footer_links=["gradio"],  # no "Use via API" link on a public form
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),  # Spaces sets 0.0.0.0
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    )
