import csv
from datetime import datetime
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

from bsp3.config import ROOT_DIR, load_models_config, resolve_path

# ARC-AGI-3 16-color palette (indexes 0..15 from arc_agi.rendering.COLOR_MAP).
PALETTE = [
    "#FFFFFF",
    "#CCCCCC",
    "#999999",
    "#666666",
    "#333333",
    "#000000",
    "#E53AA3",
    "#FF7BCC",
    "#F93C31",
    "#1E93FF",
    "#88D8F1",
    "#FFDC00",
    "#FF851B",
    "#921231",
    "#4FCC30",
    "#A356D6",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def _events_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for step in steps:
        call = step.get("model_call")
        if call:
            events.append(
                {
                    "event": "agent_plan",
                    "step": step.get("step"),
                    "provider": call.get("provider"),
                    "model": call.get("model"),
                    "actions": [p.get("action") for p in call.get("plan", [])],
                    "reason": call.get("reason", ""),
                }
            )
        diff = (step.get("state") or {}).get("diff") or {}
        events.append(
            {
                "event": "step",
                "step": step.get("step"),
                "action": step.get("action"),
                "action_data": step.get("action_data"),
                "changed_pixels": diff.get("changed_pixels"),
                "game_state": step.get("game_state"),
            }
        )
    return events


def list_runs(runs_dir: Path | str, limit: int = 50) -> list[dict[str, Any]]:
    base = resolve_path(runs_dir)
    if not base.exists():
        return []

    dirs = sorted(
        (p for p in base.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    runs: list[dict[str, Any]] = []
    for run_path in dirs:
        live = _read_json(run_path / "live.json") or {}
        summary = _read_json(run_path / "summary.json") or {}
        if not live and not summary and not (run_path / "steps.jsonl").exists():
            continue
        status = live.get("status") or ("finished" if summary else "running")
        runs.append(
            {
                "run_id": run_path.name,
                "status": status,
                "game_id": live.get("game_id") or summary.get("game_id"),
                "level": live.get("level", summary.get("level")),
                "condition": live.get("condition") or summary.get("condition"),
                "model": live.get("model_profile") or live.get("model") or summary.get("exact_model"),
                "won": live.get("won", summary.get("won")),
            }
        )
        if len(runs) >= limit:
            break
    return runs


def read_monitor_state(runs_dir: Path | str, run_id: str | None = None) -> dict[str, Any]:
    """Return recent run metadata plus the active run's live board and event log."""
    base = resolve_path(runs_dir)
    runs = list_runs(base)

    selected = run_id if run_id and (base / run_id).is_dir() else None
    if selected is None:
        latest_path = base / "latest.txt"
        if latest_path.exists():
            try:
                candidate = latest_path.read_text(encoding="utf-8").strip()
            except OSError:
                candidate = ""
            if candidate and (base / candidate).is_dir():
                selected = candidate
    if selected is None and runs:
        selected = runs[0]["run_id"]

    if selected is None:
        return {"runs": runs, "active": None, "events": []}

    run_path = base / selected
    live = _read_json(run_path / "live.json") or {}
    summary = _read_json(run_path / "summary.json") or {}
    active = {**summary, **live, "run_id": selected}
    if "status" not in active:
        active["status"] = "finished" if summary else "unknown"
    if "model" not in active and "exact_model" in summary:
        active["model"] = summary["exact_model"]

    events = _read_jsonl(run_path / "events.jsonl")
    if not events:
        steps = _read_jsonl(run_path / "steps.jsonl")
        if steps:
            events = _events_from_steps(steps)
            if "step" not in active:
                active["step"] = steps[-1].get("step")

    return {"runs": runs, "active": active, "events": events[-300:]}


INDEX_HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BSP3 Live Monitor</title>
<style>
  :root {{
    color-scheme: dark;
    --bg: #0f1217;
    --panel: #171b22;
    --border: #282f3b;
    --text: #e6edf3;
    --muted: #8b949e;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }}
  header {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    padding: 10px 16px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{ font-size: 15px; margin: 0; letter-spacing: 0.02em; }}
  .controls {{
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 8px 16px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }}
  .controls .row {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
  .controls label {{ display: flex; gap: 6px; align-items: center; }}
  .controls input[type="number"] {{ width: 72px; }}
  .step {{ display: flex; gap: 8px; align-items: center; }}
  .step .idx {{ width: 18px; }}
  .step .s-level-label {{ width: 28px; }}
  .step .s-level {{ width: 64px; }}
  .step .s-remove {{ padding: 4px 9px; }}
  #start-btn {{ background: #4ade80; border-color: #4ade80; color: #ffffff; }}
  #stop-btn {{ background: #f87171; border-color: #f87171; color: #ffffff; }}
  button:disabled {{ opacity: 0.45; cursor: not-allowed; }}
  select, button {{
    background: #0d1015;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 13px;
  }}
  .badge {{
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    background: #232a35;
  }}
  .status-running {{ background: #133928; color: #ffffff; }}
  .status-extracting_knowledge {{ background: #2c2544; color: #ffffff; }}
  .status-finished {{ background: #1e293b; color: #ffffff; }}
  .status-error {{ background: #451a1d; color: #ffffff; }}
  .status-stopped {{ background: #3f3f46; color: #ffffff; }}
  main {{
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: minmax(340px, 560px) 1fr;
    gap: 12px;
    padding: 12px;
  }}
  @media (max-width: 900px) {{
    body {{ height: auto; }}
    main {{ grid-template-columns: 1fr; }}
    .board-wrap {{ min-height: 70vh; }}
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
  }}
  .meta {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px 14px;
    padding: 12px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }}
  .meta .k {{ color: var(--text); font-size: 11px; text-transform: uppercase; }}
  .meta .v {{ font-family: ui-monospace, SFMono-Regular, monospace; word-break: break-word; }}
  .board-wrap {{
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 14px;
    gap: 10px;
    background: #0b0e13;
  }}
  canvas {{
    max-width: 100%;
    max-height: 100%;
    image-rendering: pixelated;
    border: 1px solid var(--border);
    background: #111;
  }}
  .hint {{ color: var(--text); font-size: 12px; text-align: center; min-height: 18px; }}
  .hint.waiting {{ color: #fbbf24; }}
  .log-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }}
  .log-body {{
    flex: 1;
    overflow-y: auto;
    padding: 8px 12px;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 12px;
    line-height: 1.5;
  }}
  .ev {{
    display: grid;
    grid-template-columns: 64px 126px 1fr;
    gap: 8px;
    padding: 3px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  }}
  .ts {{ color: var(--muted); }}
  .tag {{ font-weight: 600; }}
  .t-api_request {{ color: #60a5fa; }}
  .t-api_response {{ color: #4ade80; }}
  .t-api_wait {{ color: #fbbf24; }}
  .t-api_http_error, .t-api_error, .t-error {{ color: #f87171; }}
  .t-provider_fallback, .t-agent_fallback {{ color: #fb923c; }}
  .t-agent_plan {{ color: #c084fc; }}
  .t-step {{ color: #e2e8f0; }}
  .t-run_start, .t-run_end, .t-knowledge_start, .t-knowledge_done {{ color: #2dd4bf; }}
</style>
</head>
<body>
<header>
  <h1>BSP3 Live Monitor</h1>
  <select id="run-select"></select>
  <button id="follow-btn" type="button">Follow latest: ON</button>
  <span id="status-badge" class="badge">idle</span>
  <span id="poll-info" class="hint" style="margin-left:auto"></span>
</header>
<div class="controls">
  <div class="row">
    <strong>Chain</strong>
    <label>Game <select id="c-game"></select></label>
    <label>Level <input id="c-level" type="number" min="0" value="0"></label>
    <div id="steps"></div>
    <button id="add-step" type="button">+ Step</button>
  </div>
  <div class="row">
    <label>Judge <select id="c-judge"></select></label>
    <label>Budget <input id="c-budget" type="number" min="1" placeholder="auto"></label>
    <span id="conds"></span>
    <button id="start-btn" type="button">Start</button>
    <button id="stop-btn" type="button">Stop</button>
    <span id="ctrl-info"></span>
  </div>
</div>
<main>
  <section class="card">
    <div class="meta">
      <div><div class="k">Model</div><div id="m-model" class="v">-</div></div>
      <div><div class="k">Game / Level</div><div id="m-game" class="v">-</div></div>
      <div><div class="k">Step / Budget</div><div id="m-step" class="v">-</div></div>
      <div><div class="k">Calls / Tokens</div><div id="m-tokens" class="v">-</div></div>
      <div style="grid-column: 1 / -1"><div class="k">Last Action / Reason</div><div id="m-action" class="v">-</div></div>
    </div>
    <div class="board-wrap">
      <canvas id="board" width="512" height="512"></canvas>
      <div id="board-note" class="hint">Waiting for a run...</div>
    </div>
  </section>
  <section class="card">
    <div class="log-head">
      <strong>API &amp; Agent Log</strong>
      <label style="font-size:12px;color:var(--text)"><input id="autoscroll" type="checkbox" checked> Auto scroll</label>
    </div>
    <div id="log" class="log-body"></div>
  </section>
</main>
<script>
const PALETTE = {json.dumps(PALETTE)};
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const runSelect = document.getElementById("run-select");
const followBtn = document.getElementById("follow-btn");
const statusBadge = document.getElementById("status-badge");
const pollInfo = document.getElementById("poll-info");
const boardNote = document.getElementById("board-note");
const logEl = document.getElementById("log");
const autoScrollEl = document.getElementById("autoscroll");
const stepsEl = document.getElementById("steps");
const addStepBtn = document.getElementById("add-step");
const cGame = document.getElementById("c-game");
const cLevel = document.getElementById("c-level");
const cJudge = document.getElementById("c-judge");
const cBudget = document.getElementById("c-budget");
const condsEl = document.getElementById("conds");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const ctrlInfo = document.getElementById("ctrl-info");

let lastBoardSize = 0;

function fitBoard() {{
  const wrap = canvas.parentElement;
  const note = document.getElementById("board-note");
  const style = getComputedStyle(wrap);
  const padX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  const padY = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
  const gap = parseFloat(style.rowGap || style.gap) || 0;
  const availW = wrap.clientWidth - padX;
  const availH = wrap.clientHeight - padY - note.offsetHeight - gap;
  const size = Math.max(120, Math.min(availW, availH, 480));
  if (size === lastBoardSize) return;
  lastBoardSize = size;
  canvas.style.width = size + "px";
  canvas.style.height = size + "px";
}}

window.addEventListener("resize", fitBoard);
window.addEventListener("load", fitBoard);

let followLatest = true;
let selectedRun = "";
let lastRunsSig = "";
let lastLogSig = "";

followBtn.onclick = () => {{
  followLatest = !followLatest;
  followBtn.textContent = "Follow latest: " + (followLatest ? "ON" : "OFF");
  if (followLatest) {{
    selectedRun = "";
    refresh();
  }}
}};

runSelect.onchange = () => {{
  selectedRun = runSelect.value;
  followLatest = false;
  followBtn.textContent = "Follow latest: OFF";
  refresh();
}};

let optionsData = {{ games: [], models: [], conditions: [] }};

function fillSelect(el, values, selected) {{
  el.textContent = "";
  for (const value of values) {{
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    if (value === selected) opt.selected = true;
    el.appendChild(opt);
  }}
}}

function renumberSteps() {{
  [...stepsEl.children].forEach((row, i) => {{
    row.querySelector(".idx").textContent = (i + 1) + ".";
  }});
}}

function gameLevels() {{
  return (optionsData.game_levels || {{}})[cGame.value] || 1;
}}

function applyLevelMax() {{
  const levels = gameLevels();
  const max = Math.max(0, levels - stepsEl.children.length);
  cLevel.max = max;
  if (Number(cLevel.value) > max) cLevel.value = max;
  addStepBtn.disabled = stepsEl.children.length >= levels;
  refreshStepLevels();
}}

function refreshStepLevels() {{
  const start = Number(cLevel.value) || 0;
  [...stepsEl.children].forEach((row, i) => {{
    row.querySelector(".s-level-label").textContent = "L" + (start + i);
  }});
}}

function addStep(model) {{
  const row = document.createElement("div");
  row.className = "step";
  const idx = document.createElement("span");
  idx.className = "idx";
  const levelLabel = document.createElement("span");
  levelLabel.className = "s-level-label";
  const modelSel = document.createElement("select");
  modelSel.className = "s-model";
  fillSelect(modelSel, optionsData.models, model);
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "s-remove";
  remove.textContent = "x";
  remove.onclick = () => {{ row.remove(); renumberSteps(); applyLevelMax(); }};
  row.append(idx, levelLabel, modelSel, remove);
  stepsEl.appendChild(row);
  renumberSteps();
  applyLevelMax();
}}

function collectSteps() {{
  const start = Number(cLevel.value) || 0;
  return [...stepsEl.children].map((row, i) => ({{
    game: cGame.value,
    level: start + i,
    model: row.querySelector(".s-model").value,
  }}));
}}

function collectConditions() {{
  return [...condsEl.querySelectorAll(".cond")].filter((box) => box.checked).map((box) => box.value);
}}

async function loadOptions() {{
  try {{
    const resp = await fetch("/api/options", {{ cache: "no-store" }});
    optionsData = await resp.json();
    fillSelect(cGame, optionsData.games || []);
    fillSelect(cJudge, optionsData.models || [], "minimax");
    for (const condition of optionsData.conditions || []) {{
      const wrap = document.createElement("label");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.className = "cond";
      box.value = condition;
      box.checked = true;
      wrap.append(box, document.createTextNode(" " + condition));
      condsEl.appendChild(wrap);
    }}
    addStep();
    addStep();
    addStep();
  }} catch (err) {{
    ctrlInfo.textContent = "options error: " + err.message;
  }}
}}

addStepBtn.onclick = () => addStep();
cGame.onchange = applyLevelMax;
cLevel.oninput = refreshStepLevels;

startBtn.onclick = async () => {{
  const steps = collectSteps();
  const conditions = collectConditions();
  if (!steps.length) {{
    ctrlInfo.textContent = "add at least one step";
    return;
  }}
  if (Number(steps[steps.length - 1].level) >= gameLevels()) {{
    ctrlInfo.textContent = "this game has only " + gameLevels() + " levels, remove a step";
    return;
  }}
  if (!conditions.length) {{
    ctrlInfo.textContent = "pick at least one condition";
    return;
  }}
  startBtn.disabled = true;
  ctrlInfo.textContent = "starting...";
  const body = {{ steps, conditions, judge: cJudge.value, budget: cBudget.value }};
  try {{
    const resp = await fetch("/api/start", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(body),
    }});
    const data = await resp.json();
    ctrlInfo.textContent = data.ok ? "started: " + data.command : "error: " + data.error;
  }} catch (err) {{
    ctrlInfo.textContent = "error: " + err.message;
  }}
  refresh();
}};

stopBtn.onclick = async () => {{
  stopBtn.disabled = true;
  ctrlInfo.textContent = "stopping...";
  try {{
    const resp = await fetch("/api/stop", {{ method: "POST" }});
    const data = await resp.json();
    ctrlInfo.textContent = data.ok ? "stopped" : "error: " + data.error;
  }} catch (err) {{
    ctrlInfo.textContent = "error: " + err.message;
  }}
  refresh();
}};

loadOptions();

function drawGrid(grid) {{
  if (!Array.isArray(grid) || !grid.length) {{
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return false;
  }}
  const rows = grid.length;
  const cols = grid[0].length;
  const cw = canvas.width / cols;
  const ch = canvas.height / rows;
  for (let r = 0; r < rows; r++) {{
    for (let c = 0; c < cols; c++) {{
      const val = grid[r][c] & 15;
      ctx.fillStyle = PALETTE[val] || "#000000";
      ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
    }}
  }}
  return true;
}}

function formatEvent(ev) {{
  const kind = ev.event || "info";
  const who = [ev.provider, ev.model].filter(Boolean).join("/");
  if (kind === "api_request") return `${{who}} key#${{ev.key_index}} try ${{ev.attempt}}/${{ev.attempts}}${{ev.stream ? " (stream)" : ""}}`;
  if (kind === "api_response") return `${{who}} 200 OK ${{ev.latency_ms}}ms, ${{ev.total_tokens}} tok (think ${{ev.thoughts_tokens || 0}})`;
  if (kind === "api_wait") return `${{who}} cooldown wait ${{ev.wait_s}}s (key#${{ev.key_index}})`;
  if (kind === "api_http_error") return `${{who}} HTTP ${{ev.status}} (try ${{ev.attempt}}): ${{ev.detail || ""}}`;
  if (kind === "api_error" || kind === "error") return `${{who ? who + " " : ""}}${{ev.error || ""}}`;
  if (kind === "provider_fallback") return `${{ev.from_provider}} -> ${{ev.to_provider}} (${{ev.to_model}}): ${{ev.reason || ""}}`;
  if (kind === "agent_fallback") return `step ${{ev.step}}: choose() fallback (${{ev.reason || ""}})`;
  if (kind === "agent_plan") return `step ${{ev.step}} [${{(ev.actions || []).join(", ")}}] ${{ev.reason || ""}}`;
  if (kind === "step") return `step ${{ev.step}}${{ev.budget ? "/" + ev.budget : ""}} ${{ev.action}} changed=${{ev.changed_pixels ?? "?"}}px [${{ev.game_state || ""}}]`;
  if (kind === "knowledge_start" || kind === "knowledge_done") return `${{(ev.conditions || []).join(", ")}} -> ${{ev.store_chain}}`;
  if (kind === "run_start") return `${{ev.game_id}} L${{ev.level}} (${{ev.condition}}, budget=${{ev.budget}}) ${{who}}`;
  if (kind === "run_end") return `${{ev.outcome}} (won=${{ev.won}}, steps=${{ev.actions_taken}}, ${{ev.elapsed_s}}s, ${{ev.total_tokens}} tok)`;
  return JSON.stringify(ev);
}}

function renderLog(events) {{
  const tail = events.length ? JSON.stringify(events[events.length - 1]) : "";
  const sig = `${{selectedRun}}|${{events.length}}|${{tail}}`;
  if (sig === lastLogSig) return;
  lastLogSig = sig;
  const prevScroll = logEl.scrollTop;
  logEl.textContent = "";
  for (const ev of events) {{
    const row = document.createElement("div");
    row.className = "ev";
    const ts = document.createElement("span");
    ts.className = "ts";
    ts.textContent = ev.ts ? ev.ts.slice(11, 19) : "--:--:--";
    const tag = document.createElement("span");
    const kind = ev.event || "info";
    tag.className = "tag t-" + kind;
    tag.textContent = kind;
    const msg = document.createElement("span");
    msg.textContent = formatEvent(ev);
    row.append(ts, tag, msg);
    logEl.appendChild(row);
  }}
  logEl.scrollTop = autoScrollEl.checked ? logEl.scrollHeight : prevScroll;
}}

const WAIT_EVENTS = ["api_request", "api_wait", "api_http_error", "provider_fallback", "agent_fallback"];

function waitingText(events, status) {{
  if (status !== "running" && status !== "extracting_knowledge") return "";
  const last = events[events.length - 1];
  if (!last || !WAIT_EVENTS.includes(last.event) || !last.ts) return "";
  const seconds = Math.max(0, Math.round((Date.now() - Date.parse(last.ts)) / 1000));
  const who = last.model || last.to_model || "model";
  const what = last.event === "api_wait" ? "cooling down" : "waiting for " + who;
  const attempt = last.attempt ? ` (try ${{last.attempt}}/${{last.attempts}})` : "";
  return `${{what}}${{attempt}}: ${{seconds}} s`;
}}

async function refresh() {{
  try {{
    const q = !followLatest && selectedRun ? "?run=" + encodeURIComponent(selectedRun) : "";
    const resp = await fetch("/api/state" + q, {{ cache: "no-store" }});
    const data = await resp.json();

    const ctrl = data.control || {{}};
    startBtn.disabled = !!ctrl.running;
    stopBtn.disabled = !ctrl.running;
    if (ctrl.running) ctrlInfo.textContent = "running: " + (ctrl.command || "");

    const runs = data.runs || [];
    const rSig = runs.map(r => `${{r.run_id}}:${{r.status}}:${{r.won}}`).join("|");
    if (rSig !== lastRunsSig) {{
      lastRunsSig = rSig;
      const prevSelected = runSelect.value;
      runSelect.textContent = "";
      for (const r of runs) {{
        const opt = document.createElement("option");
        opt.value = r.run_id;
        opt.textContent = `${{r.run_id}} [${{r.status}}] ${{r.model || ""}}`;
        runSelect.appendChild(opt);
      }}
      if (prevSelected) runSelect.value = prevSelected;
    }}
    const active = data.active;
    if (active && active.run_id) {{
      runSelect.value = active.run_id;
      selectedRun = active.run_id;
    }}

    if (!active) {{
      statusBadge.className = "badge";
      statusBadge.textContent = "idle";
      boardNote.textContent = "No runs found yet.";
      drawGrid(null);
      renderLog([]);
      return;
    }}

    const status = active.status || "unknown";
    statusBadge.className = "badge status-" + status;
    statusBadge.textContent = active.won ? status + " (WON)" : status;

    const profile = active.model_profile ? active.model_profile + " | " : "";
    document.getElementById("m-model").textContent = profile + [active.provider, active.model].filter(Boolean).join(" / ");
    document.getElementById("m-game").textContent = `${{active.game_id || "-"}} L${{active.level ?? "-"}} (${{active.condition || "-"}}, s${{active.seed ?? "-"}})`;
    document.getElementById("m-step").textContent = `${{active.step ?? active.actions_taken ?? 0}} / ${{active.budget ?? "-"}} [${{active.game_state || active.outcome || "-"}}]`;
    const tok = (active.tokens && active.tokens.total_tokens) || 0;
    document.getElementById("m-tokens").textContent = `${{active.model_calls ?? 0}} calls / ${{tok}} tok`;

    const actData = active.action_data ? " " + JSON.stringify(active.action_data) : "";
    const px = active.changed_pixels != null ? ` (${{active.changed_pixels}}px)` : "";
    const reason = active.reason ? " - " + active.reason : "";
    document.getElementById("m-action").textContent = (active.action ? active.action + actData + px : active.outcome || "-") + reason;

    const hasGrid = drawGrid(active.grid);
    const waiting = waitingText(data.events || [], status);
    boardNote.classList.toggle("waiting", !!waiting);
    boardNote.textContent = active.error
      ? "Error: " + active.error
      : waiting
        ? waiting
        : hasGrid
          ? `${{active.grid.length}}x${{active.grid[0].length}} frame @ step ${{active.step ?? 0}}`
          : "No live frame saved for this run.";

    renderLog(data.events || []);
    pollInfo.textContent = "Updated " + new Date().toLocaleTimeString();
    fitBoard();
  }} catch (err) {{
    pollInfo.textContent = "Poll error: " + err.message;
  }}
}}

fitBoard();
refresh();
setInterval(refresh, 750);
</script>
</body>
</html>
"""


CONDITIONS = ["nothing", "trajectory", "notes", "ontology"]


def _spawn_run(command: list[str], cwd: Path) -> subprocess.Popen:
    if os.name == "nt":
        return subprocess.Popen(command, cwd=str(cwd), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return subprocess.Popen(command, cwd=str(cwd), start_new_session=True)


def _kill_run(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class RunController:
    """Start and stop one agent run from the monitor page."""

    def __init__(self, root: Path, runs_dir: Path | str = "runs") -> None:
        self._root = root
        self._runs_dir = resolve_path(runs_dir)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._command: list[str] = []
        self._started_at = 0.0

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is not None:
                self._proc = None
            running = self._proc is not None
            return {"running": running, "command": " ".join(self._command) if running else ""}

    def start(self, options: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"ok": False, "error": "a run is already in progress"}
            command = self._build_command(options)
            if command is None:
                return {
                    "ok": False,
                    "error": "steps and a condition are required (trajectory is off when a level repeats)",
                }
            self._started_at = time.time()
            self._proc = _spawn_run(command, self._root)
            self._command = command
            return {"ok": True, "command": " ".join(command)}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = None
                return {"ok": False, "error": "no run in progress"}
            _kill_run(self._proc)
            self._proc = None
            # The killed process cannot finish its own logs, so close the runs it left open.
            for live_path in self._runs_dir.glob("*/live.json"):
                # File mtimes tick more coarsely than time.time(), so allow a small margin.
                if live_path.stat().st_mtime < self._started_at - 2.0:
                    continue
                live = _read_json(live_path) or {}
                if live.get("status") in ("running", "extracting_knowledge"):
                    live["status"] = "stopped"
                    live_path.write_text(json.dumps(live), encoding="utf-8")
            return {"ok": True}

    @staticmethod
    def _build_command(options: dict[str, Any]) -> list[str] | None:
        raw_steps = options.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return None
        tasks: list[dict[str, Any]] = []
        for step in raw_steps:
            if not isinstance(step, dict):
                return None
            game = str(step.get("game") or "").strip()
            model = str(step.get("model") or "").strip()
            if not game or not model:
                return None
            try:
                level = int(step.get("level") or 0)
            except (TypeError, ValueError):
                return None
            tasks.append({"game_id": game, "level": level, "model": model})

        conditions = [c for c in (options.get("conditions") or []) if c in CONDITIONS]
        # Replaying a game and level already played would hand the solution to the next agent.
        pairs = [(t["game_id"], t["level"]) for t in tasks]
        if len(set(pairs)) < len(pairs):
            conditions = [c for c in conditions if c != "trajectory"]
        if not conditions:
            return None

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spec = {
            "name": f"custom_{stamp}",
            "group": "custom",
            "description": "Chain built in the monitor",
            "tasks": tasks,
        }
        command = [
            "uv", "run", "python", "scripts/run_experiment.py",
            "--chain-json", json.dumps(spec),
            "--conditions", ",".join(conditions),
            "--seeds", "0",
            "--continue-on-error",
        ]
        judge = str(options.get("judge") or "").strip()
        if judge:
            command += ["--judge", judge]
        budget = options.get("budget")
        if budget not in (None, ""):
            try:
                command += ["--budget", str(int(budget))]
            except (TypeError, ValueError):
                return None
        return command


def _monitor_options(root: Path) -> dict[str, Any]:
    games: list[str] = []
    game_levels: dict[str, int] = {}
    csv_path = root / "data" / "survey" / "games.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                game_id = row.get("game_id")
                if not game_id:
                    continue
                games.append(game_id)
                try:
                    game_levels[game_id] = int(row.get("levels") or 0)
                except (TypeError, ValueError):
                    game_levels[game_id] = 0
    models = [name for name in load_models_config().get("models", {}) if name != "mock"]
    return {"games": games, "game_levels": game_levels, "models": models, "conditions": CONDITIONS}


class MonitorHandler(BaseHTTPRequestHandler):
    """Serve the live monitor HTML page and JSON run state."""

    def __init__(self, *args: Any, runs_dir: Path | str, controller: RunController, **kwargs: Any) -> None:
        self._runs_dir = runs_dir
        self._controller = controller
        super().__init__(*args, **kwargs)

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/options":
            self._send_json(_monitor_options(ROOT_DIR))
            return

        if parsed.path == "/api/state":
            params = parse_qs(parsed.query)
            run_id = params.get("run", [None])[0]
            payload = read_monitor_state(self._runs_dir, run_id=run_id)
            payload["control"] = self._controller.status()
            self._send_json(payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            options = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            options = {}
        if not isinstance(options, dict):
            options = {}

        if parsed.path == "/api/start":
            self._send_json(self._controller.start(options))
            return
        if parsed.path == "/api/stop":
            self._send_json(self._controller.stop())
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        return


def make_server(
    runs_dir: Path | str = "runs",
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    controller = RunController(ROOT_DIR, runs_dir)
    handler = partial(MonitorHandler, runs_dir=runs_dir, controller=controller)
    return ThreadingHTTPServer((host, port), handler)


def serve_monitor(
    runs_dir: Path | str = "runs",
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    """Start the blocking HTTP monitor server until interrupted."""
    server = make_server(runs_dir, host=host, port=port)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"BSP3 monitor listening at {url} (runs: {resolve_path(runs_dir)})", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
