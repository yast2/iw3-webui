"""
Browser UI for iw3, on top of the Phase 1 headless converter.

Deliberately does NOT import iw3.gui (wxPython desktop app) - see Phase 2
notes: the desktop GUI has no job queue, no persistence, no streaming logs,
none of what was actually asked for. This shells out to `python -m iw3`
per job instead, one job at a time, and manages the queue itself.
"""
import sys
sys.path.insert(0, "/opt/nunif")

import asyncio
import json
import os
import re
import shutil
import signal
import sqlite3
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from iw3.utils import create_parser

NUNIF_HOME = os.environ.get("NUNIF_HOME", "/config")
INPUT_ROOT = Path("/input").resolve()
OUTPUT_ROOT = Path("/output").resolve()
WEBUI_DIR = Path(NUNIF_HOME) / "webui"
LOG_DIR = WEBUI_DIR / "logs"
DB_PATH = WEBUI_DIR / "jobs.db"
NUNIF_DIR = Path("/opt/nunif")

WEBUI_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Preview mode cuts a short clip out of the middle of the source and converts
# *that* with the very settings the job carries. It used to pass iw3's
# --keyframe instead, which only ever produced stills - and stills cannot show
# the one thing a 3D preview has to answer: whether depth stays stable while
# the picture moves. The middle is taken because the head of a file is titles,
# logos and fades often enough to be unrepresentative.
PREVIEW_CLIP_SECONDS = float(os.environ.get("PREVIEW_CLIP_SECONDS", "120"))
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

# ---------------------------------------------------------------------------
# The multi-stage pipeline
#
# Everything above queues one `python -m iw3` process per job. The better
# quality levels are not one process: they run a depth model over extracted
# frames, post-process the depth maps, warp from the result and encode - five
# or six programs, each reading what the previous one wrote.
#
# Rebuilding that inside this file would mean owning a frame store, a
# per-stage resume and a temp-space policy, all of which the pipeline script
# already owns. So the script stays the unit of work and this app stays a
# queue: it execs the script with a level name and parses its stage markers.
# That also keeps the two testable apart - the script can be run by hand.
#
# The script is NOT part of this repository (it is machine-specific and much
# larger than a web UI has any business carrying). If it is not installed,
# the levels that need it are offered but disabled, with the reason shown,
# rather than silently missing.
# ---------------------------------------------------------------------------
CHAIN_SCRIPT = Path(os.environ.get("IW3_CHAIN_SCRIPT", "/opt/iw3-chain/run_chain.sh"))
CHAIN_WORK_ROOT = Path(os.environ.get("IW3_CHAIN_WORK", str(OUTPUT_ROOT / "_chain")))

# ---------------------------------------------------------------------------
# Which device to convert on
#
# There is no vendor-specific code here and there does not need to be: nunif
# resolves the backend itself for any device id >= 0 (nunif/device.py) -
#
#     cuda -> mps -> xpu, and ValueError if none of them is there
#
# so the only real choice is "an accelerator" versus "the CPU". "auto" makes
# that choice by looking, which is what lets one image run on an NVIDIA box, an
# Intel box and a machine with no GPU at all without being reconfigured.
#
# Falling back to the CPU is allowed. Falling back *quietly* is not: it is
# roughly fifty times slower, and a job that merely takes forever looks exactly
# like a job that is working. So the fallback is announced in the startup log,
# in /api/health and as a banner across the top of the web UI.
# ---------------------------------------------------------------------------
IW3_GPU_SETTING = os.environ.get("IW3_GPU", "auto")


def _detect_accelerator():
    """(name, count) of the accelerator nunif would pick, or (None, 0)."""
    try:
        import torch
    except Exception:
        return None, 0
    for name in ("cuda", "mps", "xpu"):
        backend = getattr(torch, name, None)
        if backend is None:
            continue
        try:
            if name == "mps":
                available = torch.backends.mps.is_available()
                count = 1 if available else 0
            else:
                available = backend.is_available()
                count = backend.device_count() if available else 0
        except Exception:
            continue
        if available:
            return name, count
    return None, 0


def _resolve_gpu(setting):
    """(value for --gpu, accelerator name or None, warning or None)."""
    if setting.strip().lower() != "auto":
        return setting, None, None
    name, _ = _detect_accelerator()
    if name:
        return "0", name, None
    return "-1", None, (
        "No CUDA, MPS or XPU device is visible - every conversion will run on "
        "the CPU, which is far slower than any GPU. If this machine has a GPU, "
        "the container was most likely started without access to it "
        "(--gpus all for NVIDIA, --device /dev/dri for Intel/AMD)."
    )


IW3_GPU, ACCELERATOR, DEVICE_WARNING = _resolve_gpu(IW3_GPU_SETTING)

# ---------------------------------------------------------------------------
# Settings schema - introspected from iw3's own argparse parser so defaults
# and choices never drift out of sync with the installed nunif version.
# The *set* of fields exposed (and how they're grouped into widgets) is a UI
# decision and is hardcoded; the *values* (default/choices) are read live.
# ---------------------------------------------------------------------------
_parser = create_parser(required_true=False)
_defaults = {action.dest: action.default for action in _parser._actions}
_choices = {action.dest: action.choices for action in _parser._actions}


def _field(dest, label, kind, **extra):
    f = {"dest": dest, "label": label, "kind": kind, "default": _defaults.get(dest)}
    if _choices.get(dest):
        f["choices"] = list(_choices[dest])
    f.update(extra)
    return f


SETTINGS_SCHEMA = [
    _field("depth_model", "Depth model", "select"),
    _field("divergence", "Divergence (3D strength)", "float", min=0, max=10, step=0.1,
           help="0-2 is reasonable. Higher = more pop, more eye strain."),
    _field("convergence", "Convergence (screen plane)", "float", min=0, max=1, step=0.05,
           help="0-1 reasonable. 0.5 pulls part of the scene in front of the screen."),
    # Exposed because it is not a detail: `constant` nails the screen plane to
    # the middle of the depth range, `sod_v1` looks for it per frame. Measured
    # side by side on the same clip, `constant` left two to three times as much
    # of the scene in front of the screen and carried a slow drift that
    # `sod_v1` does not have, at no difference in runtime.
    _field("convergence_mode", "Convergence mode", "select",
           help="sod_v1 finds the screen plane per frame; constant fixes it in "
                "the middle of the depth range."),
    _field("foreground_scale", "Foreground scale", "float", min=-3, max=3, step=0.1,
           help="0 disabled. Source: iw3 argparse Range(-3.0, 3.0)."),
    _field("edge_dilation", "Edge dilation (x, y)", "int_pair", default=[2, 1]),
    _field("video_codec", "Video codec", "select",
           choices=["libx264", "libx265", "libopenh264", "utvideo", "ffv1"],
           help="No VAAPI/QSV path exists in iw3 - see Phase 1 notes. All software encode."),
    _field("pix_fmt", "Pixel format", "select"),
    _field("max_fps", "Max FPS", "float", min=1, max=120, step=1),
    _field("scene_detect", "Scene detection", "bool",
           help="Recommended for VDA depth models - resets state at hard cuts."),
    _field("ema_normalize", "Flicker reduction (EMA normalize)", "bool",
           help="Recommended for VDA depth models."),
    _field("ema_decay", "  EMA decay", "float", min=0, max=1, step=0.01),
    _field("ema_buffer", "  EMA lookahead buffer (frames)", "int", min=1, max=240),
]

STEREO_FORMATS = [
    {"value": "full_sbs", "label": "Full SBS (default)", "flags": {}},
    {"value": "half_sbs", "label": "Half SBS", "flags": {"half_sbs": True}},
    {"value": "tb", "label": "Full Top-Bottom", "flags": {"tb": True}},
    {"value": "half_tb", "label": "Half Top-Bottom", "flags": {"half_tb": True}},
    {"value": "vr180", "label": "VR180", "flags": {"vr180": True}},
    {"value": "cross_eyed", "label": "Cross-eyed", "flags": {"cross_eyed": True}},
    {"value": "rgbd", "label": "RGBD", "flags": {"rgbd": True}},
    {"value": "half_rgbd", "label": "Half RGBD", "flags": {"half_rgbd": True}},
    {"value": "anaglyph", "label": "Anaglyph (Dubois)", "flags": {"anaglyph": "dubois"}},
]

# ---------------------------------------------------------------------------
# Quality levels
#
# Twelve free-form settings are the right thing for somebody who knows what
# each of them does and the wrong thing for the ordinary question, which is
# "good, or fast?". The levels below are three answers to that question, each
# one a parameter set that was actually measured rather than assembled from
# plausible-looking defaults.
#
# Every level therefore carries its own runtime model, and the UI turns that
# into a time for the video in front of the user instead of quoting the hours
# the measurement took. Two rules for what goes in `costs`:
#
#   * `fixed_sec` is per job, `sec_per_frame` scales with the frames actually
#     processed. Both come from the same two-point measurements the queue's
#     other estimates use (see "Estimating queued jobs" below).
#   * `gpu` and `cpu` are added. For a single `iw3` process the encode hides
#     behind the conversion and `cpu` is 0 - the measured wall clock is
#     already in `gpu`. For the pipeline the stages run one after another, so
#     the CPU-bound stage is time the user waits.
#
# Measured on an Intel Arc Pro B60 over 21,606 frames of 1920x1080 video
# (12 min at 29.97 fps), each figure from a run of the full length rather than
# extrapolated from a short clip. `notes` names what each number is, because
# a level that quotes a GPU lower bound and a level that quotes a measured
# wall clock are not the same kind of promise.
# ---------------------------------------------------------------------------
MEASURED_ON_FRAMES = 21606

# Depth estimation at 4K costs more per frame than at HD, and not by the
# pixel ratio: this installation's own VDA_B medians are 6.56 fps at HD
# against 2.58 at 4K, a factor of 2.54, where the pixels alone would say 4.
# So the levels are seeded at HD and scaled by the factor this machine
# measured - flagged as an extrapolation wherever it is used, because no
# level has been run end to end on a 4K source.
RESOLUTION_FACTOR_4K = 6.56 / 2.58

QUALITY_LEVELS = [
    {
        "id": "fast",
        "label": "Fast",
        "chain": False,
        "costs": {"fixed_sec": 50.0, "gpu": 0.1577, "cpu": 0.0},
        "cost_sentence": "Coarsest outlines - 39% less relief on silhouettes than "
                         "the slower levels, and about half the fine detail - but "
                         "the steadiest over time and by far the cheapest.",
        "notes": "Measured wall clock including the x265 encode: 57 min 37 s for "
                 "21,606 frames, cold and warm runs 0.6% apart.",
        # One iw3 process, so the level is fully expressible as a parameter set
        # and is stored as one: the job record then says exactly what ran.
        #
        # Two of these values are corrections rather than copies of what this
        # UI shipped with. --divergence 1.376 instead of 2.0: measured on the
        # finished side-by-side output, 2.0 carried 35.3 px of disparity where
        # every other recipe carried 24.6-25.0 px, +42%, and 24.6 px is the
        # span the 1.376 was derived against in the first place. And
        # --convergence-mode sod_v1 instead of iw3's `constant`: better in all
        # twelve frequency bands across three image regions, at no cost in
        # runtime and no change to the depth map.
        "params": {
            "depth_model": "VDA_B",
            "divergence": 1.376,
            "convergence": 0.5,
            "convergence_mode": "sod_v1",
            "foreground_scale": 0,
            "edge_dilation": [2, 1],
            "video_codec": "libx265",
            "pix_fmt": "yuv420p",
            "max_fps": 1000,
            "scene_detect": True,
            "ema_normalize": True,
            "ema_decay": 0.75,
            "ema_buffer": 30,
        },
    },
    {
        "id": "economical",
        "label": "Economical",
        "chain": True,
        "costs": {"fixed_sec": 35.0, "gpu": 0.3860, "cpu": 0.1611},
        "cost_sentence": "Softer outlines than Standard - 8% less relief on "
                         "silhouettes, and noticeably less fine structure - for "
                         "roughly three quarters of the time.",
        "notes": "GPU lower bound 2 h 19 plus a measured 58 min of CPU-bound "
                 "post-processing, over 21,606 frames.",
    },
    {
        "id": "standard",
        "label": "Standard",
        "chain": True,
        "default": True,
        "costs": {"fixed_sec": 54.0, "gpu": 0.5776, "cpu": 0.1389},
        "cost_sentence": "Keeps the full fine structure of the depth model and "
                         "slightly sharper silhouettes than Fast; the most "
                         "expensive level that is not optional extras.",
        "notes": "GPU lower bound 3 h 28 plus a measured 50 min of CPU-bound "
                 "post-processing, over 21,606 frames.",
    },
    {
        "id": "custom",
        "label": "Custom (all settings)",
        "chain": False,
        "costs": None,  # estimated from this machine's finished jobs instead
        "cost_sentence": "Every iw3 setting, exactly as this UI has always "
                         "offered them. Time estimated from finished jobs on "
                         "this machine, per depth model.",
        "notes": None,
    },
]

# Two switches rather than two more levels: they are the same recipes with one
# extra stage, they multiply with the levels, and both are off by default
# because in side-by-side viewing neither was clearly worth its hours.
QUALITY_OPTIONS = [
    {
        "id": "flow",
        "label": "Optical flow smoothing",
        "chain_only": True,
        "costs": {"fixed_sec": 0.0, "gpu": 0.1777, "cpu": 0.0},
        "cost_sentence": "Less flicker on still surfaces (-10% measured). Viewed "
                         "side by side the difference was at the edge of being "
                         "visible at all; some scenes may need it.",
        "notes": "Measured 1 h 04 over 21,606 frames.",
    },
    {
        "id": "upscale",
        "label": "Denoise and 2x upscale before the warp",
        "chain_only": True,
        "costs": {"fixed_sec": 0.0, "gpu": 0.8700, "cpu": 0.0},
        "cost_sentence": "Removes compression artefacts, but adds no detail the "
                         "source did not carry: against an 8 Mbit/s source the "
                         "entire gain sat above what the source could hold.",
        "notes": "Measured 0.870 s/frame = 5 h 13 over 21,606 frames, for the "
                 "upscaling stage alone. The warp then works on four times the "
                 "pixels, which has not been measured.",
    },
]

QUALITY_BY_ID = {lv["id"]: lv for lv in QUALITY_LEVELS}
OPTION_BY_ID = {op["id"]: op for op in QUALITY_OPTIONS}
DEFAULT_QUALITY = next(lv["id"] for lv in QUALITY_LEVELS if lv.get("default"))


def _chain_available():
    """Whether the multi-stage pipeline script is installed and executable."""
    return CHAIN_SCRIPT.is_file() and os.access(CHAIN_SCRIPT, os.X_OK)


def _level_seconds(level_id, frames, bucket, flow=False, upscale=False):
    """Estimated wall clock for `frames` at this level, or None if unmodelled.

    `bucket` is 'hd' or '4k'; see RESOLUTION_FACTOR_4K for what the 4K case is
    worth. Returns (seconds, extrapolated).
    """
    level = QUALITY_BY_ID.get(level_id)
    if not level or not level.get("costs") or not frames:
        return None, False
    parts = [level["costs"]]
    if level["chain"]:
        if flow:
            parts.append(OPTION_BY_ID["flow"]["costs"])
        if upscale:
            parts.append(OPTION_BY_ID["upscale"]["costs"])
    seconds = sum(p["fixed_sec"] + frames * (p["gpu"] + p["cpu"]) for p in parts)
    if bucket == "4k":
        return seconds * RESOLUTION_FACTOR_4K, True
    return seconds, False

# ---------------------------------------------------------------------------
# Job store (SQLite under $NUNIF_HOME so the queue survives container restarts)
# ---------------------------------------------------------------------------
_db_lock = asyncio.Lock()


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                input_path TEXT NOT NULL,
                recursive INTEGER NOT NULL DEFAULT 0,
                stereo_format TEXT NOT NULL,
                params_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error TEXT,
                position INTEGER NOT NULL DEFAULT 0,
                quality TEXT NOT NULL DEFAULT 'custom',
                opt_flow INTEGER NOT NULL DEFAULT 0,
                opt_upscale INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Queue order used to be created_at, which made "run this one next" a
        # question of falsifying a timestamp. It is an explicit column now.
        # Existing databases are seeded from the order the old sort key
        # produced, so a queue that is days deep keeps running in exactly the
        # order it had when the upgrade landed.
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        # Quality levels arrived after the queue did. Rows written before them
        # carry a hand-picked parameter set, which is exactly what 'custom'
        # means - so the default backfills them correctly and their ETAs keep
        # coming from the same measured-throughput path as before.
        for name, decl in (("quality", "TEXT NOT NULL DEFAULT 'custom'"),
                           ("opt_flow", "INTEGER NOT NULL DEFAULT 0"),
                           ("opt_upscale", "INTEGER NOT NULL DEFAULT 0")):
            if name not in columns:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")
        if "position" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN position INTEGER NOT NULL DEFAULT 0")
            seed = conn.execute(
                "SELECT id FROM jobs WHERE status IN ('running','queued') "
                "ORDER BY mode <> 'preview', created_at"
            ).fetchall()
            for i, row in enumerate(seed, start=1):
                conn.execute("UPDATE jobs SET position=? WHERE id=?", (i, row["id"]))
        # Crash recovery: a 'running' row means the container died mid-job.
        # It keeps its position, so it is picked up again first.
        conn.execute("UPDATE jobs SET status='queued', started_at=NULL "
                      "WHERE status='running'")


_init_db()

# job_id -> asyncio.subprocess.Process, for cancellation of the current job
_running_procs: dict[str, asyncio.subprocess.Process] = {}
# job_id -> set of asyncio.Queue, for SSE log tailing
_log_subscribers: dict[str, set] = {}
# job_id -> progress dict, parsed live from iw3's tqdm output. In memory only:
# progress is meaningless for a job that isn't running, and a container restart
# resets running jobs to 'queued' anyway (see _init_db).
_job_progress: dict[str, dict] = {}


def _now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Queue order
#
# `position` is the single sort key the worker, the ETA sum and the table all
# read, so what the queue shows is what the queue will do. It is rewritten
# dense (1..N) after every change rather than nudged: at a hundred-odd active
# rows renumbering costs nothing, and it rules out the drift that fractional
# or gap-based schemes accumulate after enough reorders.
#
# The running job is not part of the orderable set. It is already on the GPU
# and nothing here interrupts it - it simply always sorts first.
# ---------------------------------------------------------------------------
def _queued_ids(conn):
    return [r["id"] for r in conn.execute(
        "SELECT id FROM jobs WHERE status='queued' ORDER BY position, created_at")]


def _apply_queue_order(conn, ids):
    """Renumber the queue to `ids`, running job first."""
    running = [r["id"] for r in conn.execute(
        "SELECT id FROM jobs WHERE status='running' ORDER BY position, created_at")]
    for i, job_id in enumerate(running + list(ids), start=1):
        conn.execute("UPDATE jobs SET position=? WHERE id=?", (i, job_id))


def _next_position(conn):
    row = conn.execute("SELECT MAX(position) AS m FROM jobs "
                        "WHERE status IN ('running','queued')").fetchone()
    return (row["m"] or 0) + 1


# ---------------------------------------------------------------------------
# Progress parsing
#
# iw3 drives tqdm, which already computes everything a progress bar needs -
# there is no reason to estimate any of it ourselves. A job emits these in
# sequence:
#
#   1. the Scene Boundary Detection pre-pass, which is itself a full bar:
#        "clip.mp4: Scene Boundary Detection:  72%|###  | 9911/13778 [00:14<00:05, 675.8it/s]"
#      and finishes with an unbounded summary line:
#        "clip.mp4: Scene Boundary Detection: 13779it [00:20, 675.87it/s]"
#   2. the conversion pass, the one worth showing as *the* progress:
#        "clip.mp4:  29%|####      | 3928/13778 [12:12<23:13,  7.07it/s]"
#
# So bounded-vs-unbounded does NOT identify the pass - both passes draw a
# percentage. The description does: iw3 sets tqdm_title to
# f"{basename}: Scene Boundary Detection" for the pre-pass (iw3/utils.py:1031)
# and to the bare basename for the conversion. Hence the greedy desc capture.
#
# Two further shapes seen in real logs and handled below: the clock is
# [H:]MM:SS ("[11:19<1:30:41"), and tqdm flips to "s/it" instead of "it/s"
# once a step takes over a second ("2.94s/it").
# ---------------------------------------------------------------------------
_SCENE_DETECT_TITLE = "Scene Boundary Detection"

_TQDM_BOUNDED = re.compile(
    r"(?P<desc>.*):\s+(?P<pct>\d+)%\|[^|]*\|\s*(?P<n>\d+)/(?P<total>\d+)"
    r"\s*\[(?P<elapsed>[0-9:]+)<(?P<remaining>[0-9:?]+),\s*(?P<rate>[0-9.]+)(?P<unit>it/s|s/it)"
)
_TQDM_UNBOUNDED = re.compile(
    r"(?P<desc>.*):\s+(?P<n>\d+)it\s*\[(?P<elapsed>[0-9:]+),\s*(?P<rate>[0-9.]+)(?P<unit>it/s|s/it)"
)


def _tqdm_seconds(clock: str):
    """tqdm prints [H:]MM:SS, and '?' while it has no estimate yet."""
    if not clock or "?" in clock:
        return None
    parts = clock.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def _phase_of(desc: str):
    return "scene_detect" if desc.rstrip().endswith(_SCENE_DETECT_TITLE) else "convert"


def _rate_to_fps(value: str, unit: str):
    rate = float(value)
    # Normalise s/it to it/s so the UI only ever deals with one unit.
    if unit == "s/it":
        return 1.0 / rate if rate else 0.0
    return rate


# The pipeline announces each stage on its own line before starting it:
#
#     === 2/6 depth estimation  2026-09-21 15:04:11
#
# which is the one thing a stage runner has to say to be watchable from the
# outside. The stages themselves are ordinary iw3/torch programs and draw the
# same tqdm bars as a single-process job, so the bar below is the *stage's*
# progress and is labelled as such - a chain has no honest single percentage
# to report, because its stages are nothing like equally long.
_CHAIN_STAGE = re.compile(
    r"^===\s*(?P<stage>\d+)\s*/\s*(?P<stages>\d+)\s+(?P<label>.+?)"
    r"(?:\s\s+\d{4}-\d\d-\d\d[ T][\d:]+)?\s*$")


def _parse_chain_stage(text: str):
    m = _CHAIN_STAGE.match(text.strip("\r\n"))
    if not m:
        return None
    return {
        "phase": "chain",
        "stage": int(m.group("stage")),
        "stages": int(m.group("stages")),
        "stage_label": m.group("label").strip(),
        "percent": None, "frames": None, "total_frames": None,
        "elapsed_sec": None, "eta_sec": None, "rate_fps": None,
    }


def _merge_chain_progress(previous, parsed):
    """Keep the stage a chain is in while its inner tqdm bar moves on."""
    if not previous or previous.get("phase") != "chain":
        return parsed
    merged = dict(previous)
    for key in ("percent", "frames", "total_frames", "elapsed_sec", "rate_fps"):
        merged[key] = parsed.get(key)
    # tqdm's remaining-time describes the current stage, never the chain. The
    # job's ETA comes from the level's runtime model instead (_running_eta).
    merged["eta_sec"] = None
    return merged


def _parse_progress(text: str):
    m = _TQDM_BOUNDED.search(text)
    if m:
        total = int(m.group("total"))
        n = int(m.group("n"))
        return {
            "phase": _phase_of(m.group("desc")),
            "frames": n,
            "total_frames": total,
            "percent": round(100.0 * n / total, 1) if total else None,
            "elapsed_sec": _tqdm_seconds(m.group("elapsed")),
            "eta_sec": _tqdm_seconds(m.group("remaining")),
            "rate_fps": round(_rate_to_fps(m.group("rate"), m.group("unit")), 2),
        }
    m = _TQDM_UNBOUNDED.search(text)
    if m:
        return {
            # tqdm's unbounded form knows no total, so there is no honest
            # percentage and no ETA to report for it.
            "phase": _phase_of(m.group("desc")),
            "frames": int(m.group("n")),
            "total_frames": None,
            "percent": None,
            "elapsed_sec": _tqdm_seconds(m.group("elapsed")),
            "eta_sec": None,
            "rate_fps": round(_rate_to_fps(m.group("rate"), m.group("unit")), 2),
        }
    return None


# ---------------------------------------------------------------------------
# Estimating queued jobs
#
# A queued job has produced no tqdm output yet, so its runtime has to be
# estimated. Runtime scales with the number of frames actually processed
# (duration x effective fps), not with clip length - and the per-frame rate
# depends on the depth model and the resolution.
#
# The rates are read back out of this installation's own finished jobs, so an
# ETA describes the machine it is shown on rather than the machine this was
# written on. The seed table below only fills combinations that have never run
# here yet: one local measurement beats somebody else's median.
#
# Seed values are medians over the 34 jobs completed on an Intel Arc Pro B60 as
# of 2026-08-28. Within each group the spread was small - VDA_B/1080p ran
# 6.01-7.28 fps across 8 jobs.
# ---------------------------------------------------------------------------
SEED_THROUGHPUT_FPS = {
    ("VDA_B", "4k"): 2.58,
    ("VDA_B", "hd"): 6.56,
    ("VDA_L", "hd"): 2.97,
    ("ZoeD_Any_N", "4k"): 4.59,
    ("ZoeD_Any_N", "hd"): 9.38,
}
# Used when neither this machine nor the seed table knows the model.
# Deliberately the slowest seeded rate rather than an average: overestimating
# the wait is the less annoying error.
FALLBACK_FPS = {"4k": 2.58, "hd": 2.97}

# job_id -> (model, bucket, fps) | None. A source that cannot be probed (moved,
# deleted after conversion) stays None and is simply never a sample.
_rate_samples: dict[str, tuple | None] = {}
_calibration: dict[tuple, float] = {}
_calibration_size = -1
_calibration_checked = 0.0
# _estimate_seconds() runs once per queued job, and a queue is routinely
# hundreds of jobs deep, so anything _throughput() does per call is multiplied
# by the length of the queue on every poll of /api/jobs. Finished jobs appear
# hours apart; re-checking a few times a minute is more than enough.
CALIBRATION_RECHECK_SEC = 15.0


def _bucket(height):
    return "4k" if (height or 0) >= 1600 else "hd"


def _job_rate(row):
    """Frames per second a finished job actually achieved."""
    if row["mode"] != "convert" or row["recursive"]:
        # A preview converts an extracted clip that is deleted afterwards, and
        # a folder job is many files under one timestamp: neither is a clean
        # measurement of one file's throughput.
        return None
    if QUALITY_BY_ID.get(row["quality"], {}).get("chain"):
        # A pipeline job's elapsed time covers five programs, only one of
        # which is the depth model this table is keyed on. Counting it here
        # would make every single-process estimate several times too slow.
        return None
    if not row["started_at"] or not row["finished_at"]:
        return None
    try:
        elapsed = (datetime.fromisoformat(row["finished_at"])
                   - datetime.fromisoformat(row["started_at"])).total_seconds()
    except ValueError:
        return None
    if elapsed <= 0:
        return None
    info = _probe_video(INPUT_ROOT / row["input_path"])
    if not info or not info.get("duration_sec") or not info.get("fps"):
        return None
    params = json.loads(row["params_json"])
    max_fps = params.get("max_fps") or _defaults.get("max_fps") or 30
    frames = info["duration_sec"] * min(info["fps"], float(max_fps))
    model = params.get("depth_model") or _defaults.get("depth_model")
    return model, _bucket(info.get("height")), frames / elapsed


def _throughput():
    """Measured rates per (depth model, resolution bucket) for this machine.

    Median rather than mean: a job that was silently interrupted, or one that
    shared the GPU with something else, otherwise drags a whole group with it.
    """
    global _calibration, _calibration_size, _calibration_checked
    now = time.monotonic()
    if _calibration_size >= 0 and now - _calibration_checked < CALIBRATION_RECHECK_SEC:
        return _calibration
    _calibration_checked = now

    where = "status='done' AND mode='convert' AND recursive=0"
    with _db() as conn:
        # Count first: the row fetch and the probing behind it are only worth
        # doing when a job has actually finished since the last look.
        size = conn.execute(f"SELECT COUNT(*) FROM jobs WHERE {where}").fetchone()[0]
        if size == _calibration_size:
            return _calibration
        rows = conn.execute(f"SELECT * FROM jobs WHERE {where}").fetchall()

    groups: dict[tuple, list] = {}
    for r in rows:
        if r["id"] not in _rate_samples:
            _rate_samples[r["id"]] = _job_rate(r)
        sample = _rate_samples[r["id"]]
        if sample:
            groups.setdefault((sample[0], sample[1]), []).append(sample[2])
    _calibration = {k: statistics.median(v) for k, v in groups.items()}
    _calibration_size = len(rows)
    return _calibration

def _running_eta(job_row, progress):
    """(seconds_left, is_estimate) for the job currently running.

    tqdm's own remaining-time is only usable once the *conversion* pass is
    running. During the Scene Boundary Detection pre-pass its ETA describes
    that pre-pass alone - a few seconds - which would wildly understate the
    job. Fall back to the measured-throughput estimate until then.
    """
    if progress and progress.get("phase") == "convert" and progress.get("eta_sec") is not None:
        return progress["eta_sec"], False
    total = _estimate_seconds(job_row)
    if total is None:
        return None, True
    # An estimate is a total, and a running job wants what is *left*. For the
    # scene-detection pre-pass the difference is seconds; for a pipeline job,
    # whose stages each draw a bar that knows nothing about the other stages,
    # the fallback lasts the whole job - and an ETA that stands still for four
    # hours is worse than no ETA at all.
    if job_row["started_at"]:
        try:
            elapsed = (datetime.now(timezone.utc)
                       - datetime.fromisoformat(job_row["started_at"])).total_seconds()
        except ValueError:
            elapsed = 0.0
        total = max(0.0, total - elapsed)
    return total, True


_probe_cache: dict[tuple, dict] = {}


def _probe_video(path: Path):
    """Duration/fps/height for a source file. Cached on (path, size, mtime)."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_size, int(st.st_mtime))
    if key in _probe_cache:
        return _probe_cache[key]
    try:
        import av
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            info = {
                "duration_sec": float(container.duration / 1_000_000) if container.duration else None,
                "fps": float(stream.average_rate) if stream.average_rate else None,
                "height": stream.codec_context.height,
            }
    except Exception:
        info = None
    _probe_cache[key] = info
    return info


def _estimate_seconds(job_row):
    """Rough runtime estimate for a job that hasn't started yet."""
    if job_row["recursive"]:
        # A folder job is N unknown files; not worth a fake number.
        return None
    try:
        path = _safe_input_path(job_row["input_path"])
    except HTTPException:
        return None
    info = _probe_video(path)
    if not info or not info.get("duration_sec") or not info.get("fps"):
        return None

    params = json.loads(job_row["params_json"])
    level = QUALITY_BY_ID.get(job_row["quality"], {})
    # The pipeline extracts every frame of the source; max_fps is an iw3
    # argument and the pipeline is not one iw3 call.
    if level.get("chain"):
        effective_fps = info["fps"]
    else:
        max_fps = params.get("max_fps") or _defaults.get("max_fps") or 30
        effective_fps = min(info["fps"], float(max_fps))
    # A preview only ever converts the extracted clip, so the source length
    # beyond it costs nothing. Cutting the clip itself is a stream copy of a
    # few seconds and is not worth modelling.
    duration = info["duration_sec"]
    if job_row["mode"] == "preview":
        duration = min(duration, PREVIEW_CLIP_SECONDS)
    frames = duration * effective_fps

    bucket = _bucket(info.get("height"))
    # A level with a runtime model of its own is estimated from that model:
    # it was measured as a whole recipe, which is a better description of it
    # than a per-depth-model rate could be.
    if level.get("costs"):
        seconds, _ = _level_seconds(job_row["quality"], frames, bucket,
                                    flow=bool(job_row["opt_flow"]),
                                    upscale=bool(job_row["opt_upscale"]))
        if seconds is not None:
            return seconds
    model = params.get("depth_model") or _defaults.get("depth_model")
    rate = (_throughput().get((model, bucket))
            or SEED_THROUGHPUT_FPS.get((model, bucket))
            or FALLBACK_FPS[bucket])
    return frames / rate


def _preview_dir(job_id) -> Path:
    return OUTPUT_ROOT / "_previews" / job_id


def _preview_window(duration_sec, clip_sec):
    """(start, length) of the centred clip. Shorter sources are taken whole."""
    if not duration_sec or duration_sec <= clip_sec:
        return 0.0, None
    return (duration_sec - clip_sec) / 2.0, clip_sec


def _chain_work_dir(job_id) -> Path:
    return CHAIN_WORK_ROOT / job_id


def _build_chain_argv(job_row, input_path=None, out_dir=None):
    """The pipeline call for a multi-stage level.

    Contract with the script, kept deliberately small - one level name, one
    input, one output directory, one scratch directory, and a switch per
    option. Everything about *how* a level is produced stays in the script;
    this app only ever picks a level and waits.
    """
    if input_path is None:
        input_path = INPUT_ROOT / job_row["input_path"]
    if out_dir is None:
        out_dir = _preview_dir(job_row["id"]) if job_row["mode"] == "preview" else OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    work = _chain_work_dir(job_row["id"])
    work.mkdir(parents=True, exist_ok=True)

    argv = [str(CHAIN_SCRIPT),
            "--level", job_row["quality"],
            "-i", str(input_path),
            "-o", str(out_dir),
            "--work", str(work),
            "--gpu", IW3_GPU]
    if job_row["opt_flow"]:
        argv.append("--flow")
    if job_row["opt_upscale"]:
        argv.append("--upscale")
    fmt = next((s for s in STEREO_FORMATS if s["value"] == job_row["stereo_format"]),
               STEREO_FORMATS[0])
    argv += ["--stereo-format", fmt["value"]]
    return argv


def _build_argv(job_row, input_path=None, out_dir=None):
    if QUALITY_BY_ID.get(job_row["quality"], {}).get("chain"):
        return _build_chain_argv(job_row, input_path=input_path, out_dir=out_dir)
    params = json.loads(job_row["params_json"])
    if input_path is None:
        input_path = INPUT_ROOT / job_row["input_path"]
    if out_dir is None:
        out_dir = _preview_dir(job_row["id"]) if job_row["mode"] == "preview" else OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = ["python3", "-m", "iw3", "-i", str(input_path), "-o", str(out_dir), "-y", "--gpu", IW3_GPU]

    if job_row["recursive"]:
        argv.append("--recursive")
        argv.append("--skip-error")

    for f in SETTINGS_SCHEMA:
        dest = f["dest"]
        if dest not in params:
            continue
        val = params[dest]
        flag = "--" + dest.replace("_", "-")
        if f["kind"] == "bool":
            if val:
                argv.append(flag)
        elif f["kind"] == "int_pair":
            argv += [flag, str(val[0]), str(val[1])]
        else:
            argv += [flag, str(val)]

    fmt = next((s for s in STEREO_FORMATS if s["value"] == job_row["stereo_format"]), STEREO_FORMATS[0])
    for k, v in fmt["flags"].items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        else:
            argv += [flag, str(v)]

    return argv


async def _publish_log(job_id, line):
    for q in _log_subscribers.get(job_id, ()):
        q.put_nowait(line)


async def _run_logged(job_id, argv, logf):
    """Run a helper process, streaming its output into the job log.

    Registered in _running_procs like the conversion itself, so cancelling a
    job during clip extraction works the same way it does mid-conversion.
    """
    await _write_log(job_id, logf, f"$ {' '.join(argv)}\n")
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    _running_procs[job_id] = proc
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            await _write_log(job_id, logf, line.decode(errors="replace"))
        return await proc.wait()
    finally:
        _running_procs.pop(job_id, None)


async def _write_log(job_id, logf, text):
    logf.write(text)
    logf.flush()
    await _publish_log(job_id, text)


async def _extract_preview_clip(job_id, src: Path, dest: Path, logf):
    """Cut PREVIEW_CLIP_SECONDS out of the middle of src into dest."""
    info = _probe_video(src)
    start, length = _preview_window((info or {}).get("duration_sec"), PREVIEW_CLIP_SECONDS)
    if length is None:
        await _write_log(job_id, logf,
                          f"[preview] source is shorter than {PREVIEW_CLIP_SECONDS:g}s, using it whole\n")
    else:
        await _write_log(job_id, logf,
                          f"[preview] cutting {length:g}s from {start:.1f}s (middle of the source)\n")

    # -nostats keeps ffmpeg's \r progress out of a log viewer that is already
    # busy throttling iw3's.
    base = [FFMPEG_BIN, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "warning", "-y"]
    # -ss *before* -i is an input seek: it lands on the keyframe at or before
    # the mark, which is exactly what a stream copy needs.
    window = (["-ss", f"{start:.3f}"] if start else []) + ["-i", str(src)]
    window += ["-t", f"{length:.3f}"] if length else []
    # Only the first video and (if present) the first audio track: subtitle and
    # attachment streams carried over from a container like MKV have no place
    # in the mp4 the clip is written to.
    maps = ["-map", "0:v:0", "-map", "0:a:0?"]

    # The video is copied, never re-encoded: the clip has to look exactly like
    # the source or it cannot be used to judge the source. The audio is
    # re-encoded unconditionally - mp4 refuses plenty of the audio codecs that
    # arrive in mkv/wmv containers, and two minutes of AAC costs no measurable
    # time. That keeps the fallback below for the one case that really needs
    # it: a video codec mp4 cannot hold at all.
    rc = await _run_logged(job_id, base + window + maps +
                            ["-c:v", "copy", "-c:a", "aac",
                             "-avoid_negative_ts", "make_zero", str(dest)], logf)
    if rc < 0:
        # Killed by a signal - that is a cancellation, not a bad source.
        raise RuntimeError(f"preview extraction terminated by signal {-rc}")
    if rc == 0 and dest.exists() and dest.stat().st_size > 0:
        return

    # Re-encoding two minutes is a moment of CPU and always works.
    await _write_log(job_id, logf, "[preview] stream copy failed, re-encoding the clip instead\n")
    rc = await _run_logged(job_id, base + window + maps +
                            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                             "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest)], logf)
    if rc != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"preview clip extraction failed (ffmpeg exit code {rc})")


async def _run_job(job_id):
    with _db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        return

    log_path = LOG_DIR / f"{job_id}.log"
    is_chain = bool(QUALITY_BY_ID.get(row["quality"], {}).get("chain"))

    with _db() as conn:
        conn.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (_now(), job_id))
        conn.commit()

    clip_path = None
    with open(log_path, "a") as logf:
        if row["mode"] == "preview":
            src = _safe_input_path(row["input_path"])
            out_dir = _preview_dir(job_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            # iw3 names its output after the input's basename, so the _preview
            # suffix carries through to <name>_preview_LRF_Full_SBS.mp4 and a
            # preview can never be mistaken for a finished conversion.
            clip_path = out_dir / f"{src.stem}_preview.mp4"
            _job_progress[job_id] = {"phase": "extract", "percent": None, "frames": None,
                                     "total_frames": None, "elapsed_sec": None,
                                     "eta_sec": None, "rate_fps": None}
            try:
                await _extract_preview_clip(job_id, src, clip_path, logf)
            except Exception as e:
                # Report it as a finished-and-failed job rather than letting it
                # escape to the worker loop: an open log viewer waits for the
                # end marker below and would otherwise hang on a dead job.
                await _write_log(job_id, logf, f"\n[job failed, {e}]\n")
                with _db() as conn:
                    conn.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
                                  (_now(), str(e), job_id))
                    conn.commit()
                await _publish_log(job_id, "__EOF__")
                return
            finally:
                _job_progress.pop(job_id, None)
            argv = _build_argv(row, input_path=clip_path, out_dir=out_dir)
        else:
            argv = _build_argv(row)

        await _write_log(job_id, logf, f"$ {' '.join(argv)}\n")
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(NUNIF_DIR),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        _running_procs[job_id] = proc
        try:
            # iw3 uses carriage returns for tqdm progress - a real terminal
            # overwrites the same line on \r rather than accumulating one
            # stored line per tick. A multi-hour job can emit tens of
            # thousands of \r ticks; storing (and later replaying) every one
            # of them as a separate log line is what crashed the log viewer.
            # \n-terminated lines are real output and always flush
            # immediately; \r progress ticks are throttled to ~2/sec.
            buf = b""
            last_progress_flush = 0.0
            PROGRESS_THROTTLE_SEC = 0.5
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf or b"\r" in buf:
                    idx_n = buf.find(b"\n")
                    idx_r = buf.find(b"\r")
                    if idx_n != -1 and (idx_r == -1 or idx_n < idx_r):
                        idx, is_progress = idx_n, False
                    else:
                        idx, is_progress = idx_r, True
                    line, buf = buf[:idx], buf[idx + 1:]
                    text = line.decode(errors="replace") + "\n"
                    # Parse before the throttle: the log only needs ~2 ticks a
                    # second, but the progress bar should reflect the newest
                    # tick we have actually seen.
                    if is_chain:
                        stage = _parse_chain_stage(text)
                        if stage:
                            _job_progress[job_id] = stage
                        else:
                            parsed = _parse_progress(text)
                            if parsed:
                                _job_progress[job_id] = _merge_chain_progress(
                                    _job_progress.get(job_id), parsed)
                    else:
                        parsed = _parse_progress(text)
                        if parsed:
                            _job_progress[job_id] = parsed
                    now = time.monotonic()
                    if is_progress and now - last_progress_flush < PROGRESS_THROTTLE_SEC:
                        continue
                    last_progress_flush = now
                    logf.write(text)
                    logf.flush()
                    await _publish_log(job_id, text)
            if buf:
                text = buf.decode(errors="replace") + "\n"
                logf.write(text)
                await _publish_log(job_id, text)
            returncode = await proc.wait()
        finally:
            _running_procs.pop(job_id, None)
            _job_progress.pop(job_id, None)

    status = "done" if returncode == 0 else "failed"
    if is_chain:
        work = _chain_work_dir(job_id)
        if status == "done":
            # Tens of thousands of extracted frames and as many depth maps -
            # keeping them would fill the output volume within a few jobs.
            shutil.rmtree(work, ignore_errors=True)
        elif work.exists():
            await _publish_log(job_id, f"[pipeline] scratch kept for inspection: {work}\n")
    if clip_path is not None:
        if status == "done":
            # The clip is an intermediate, reproducible in seconds. Only the
            # stereo output next to it is worth keeping.
            try:
                clip_path.unlink()
            except OSError:
                pass
        else:
            await _publish_log(job_id, f"[preview] clip kept for inspection: {clip_path}\n")
    with _db() as conn:
        conn.execute("UPDATE jobs SET status=?, finished_at=?, error=? WHERE id=?",
                      (status, _now(), None if returncode == 0 else f"exit code {returncode}", job_id))
        conn.commit()
    await _publish_log(job_id, f"\n[job {status}, exit code {returncode}]\n")
    await _publish_log(job_id, "__EOF__")


async def _worker_loop():
    while True:
        with _db() as conn:
            # One sort key, set by create_job and by the reorder endpoints.
            # created_at only breaks ties between rows that somehow share a
            # position; it no longer decides anything on its own.
            row = conn.execute(
                "SELECT id FROM jobs WHERE status='queued' "
                "ORDER BY position, created_at LIMIT 1"
            ).fetchone()
        if row is None:
            await asyncio.sleep(1.0)
            continue
        # One job at a time - the GPU cannot be shared across concurrent conversions.
        try:
            await _run_job(row["id"])
        except Exception as e:
            with _db() as conn:
                conn.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
                              (_now(), str(e), row["id"]))
                conn.commit()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()


@app.on_event("startup")
async def _startup():
    # First thing in the log, because it is the first thing to check when
    # conversions turn out to be slow.
    if ACCELERATOR:
        print(f"[iw3-webui] device: {ACCELERATOR} (--gpu {IW3_GPU})", flush=True)
    elif DEVICE_WARNING:
        print(f"[iw3-webui] WARNING: {DEVICE_WARNING}", flush=True)
    else:
        print(f"[iw3-webui] device: --gpu {IW3_GPU} (set explicitly via IW3_GPU)", flush=True)
    asyncio.create_task(_worker_loop())


class JobCreate(BaseModel):
    mode: str  # "convert" | "preview"
    input_path: str  # relative to /input
    recursive: bool = False
    stereo_format: str = "full_sbs"
    quality: str = "custom"
    flow: bool = False
    upscale: bool = False
    # Only 'custom' reads these. A level is a measured recipe; letting a form
    # field through would make the name on the job a claim about a run that
    # never happened.
    params: dict = {}


def _safe_input_path(rel_path: str) -> Path:
    p = (INPUT_ROOT / rel_path.lstrip("/")).resolve()
    if INPUT_ROOT not in p.parents and p != INPUT_ROOT:
        raise HTTPException(400, "path escapes /input")
    return p


@app.get("/api/schema")
def get_schema():
    return {"fields": SETTINGS_SCHEMA, "stereo_formats": STEREO_FORMATS}


def _level_public(level, available):
    out = {k: level[k] for k in ("id", "label", "chain", "cost_sentence", "notes")}
    out["available"] = True if not level["chain"] else available
    out["unavailable_reason"] = None if out["available"] else (
        f"needs the multi-stage pipeline script ({CHAIN_SCRIPT}), which is not "
        f"installed in this container")
    out["settings_form"] = not level.get("params") and not level["chain"]
    out["default"] = bool(level.get("default"))
    return out


@app.get("/api/quality")
def quality():
    """The levels, their switches, and what each one honestly costs."""
    available = _chain_available()
    return {
        "levels": [_level_public(lv, available) for lv in QUALITY_LEVELS],
        "options": [{k: op[k] for k in ("id", "label", "chain_only",
                                        "cost_sentence", "notes")}
                    for op in QUALITY_OPTIONS],
        "default": DEFAULT_QUALITY if available else "fast",
        "chain_available": available,
        "chain_script": str(CHAIN_SCRIPT),
        "measured_on_frames": MEASURED_ON_FRAMES,
    }


@app.get("/api/estimate")
def estimate(path: str = "", mode: str = "convert", flow: bool = False,
             upscale: bool = False):
    """How long each level would take *for this video*.

    The measurements behind the levels were taken on one 12-minute file, and a
    fixed "3 h 28" on a button would be wrong for every other length. So the
    frame count of the selected source is what is quoted, and when nothing is
    selected the numbers say plainly which file they belong to.
    """
    info = None
    if path.strip():
        info = _probe_video(_safe_input_path(path))
    if info and info.get("duration_sec") and info.get("fps"):
        duration = info["duration_sec"]
        if mode == "preview":
            duration = min(duration, PREVIEW_CLIP_SECONDS)
        frames = duration * info["fps"]
        bucket = _bucket(info.get("height"))
        source = "selected file"
    else:
        # The reference the levels were measured on, so a first-time visitor
        # still sees the shape of the ladder before picking a file.
        frames = MEASURED_ON_FRAMES
        bucket = "hd"
        source = "reference measurement (12 min, 1080p)"

    levels = {}
    for lv in QUALITY_LEVELS:
        seconds, extrapolated = _level_seconds(lv["id"], frames, bucket,
                                               flow=flow, upscale=upscale)
        levels[lv["id"]] = {
            "seconds": round(seconds) if seconds else None,
            "extrapolated": extrapolated,
        }
    options = {}
    for op in QUALITY_OPTIONS:
        c = op["costs"]
        extra = c["fixed_sec"] + frames * (c["gpu"] + c["cpu"])
        if bucket == "4k":
            extra *= RESOLUTION_FACTOR_4K
        options[op["id"]] = {"seconds": round(extra), "extrapolated": bucket == "4k"}
    return {
        "frames": round(frames),
        "resolution_bucket": bucket,
        "basis": source,
        "levels": levels,
        "options": options,
    }


@app.get("/api/health")
def health():
    """What this container can actually reach - the question a build cannot answer.

    A `docker build` host usually has no GPU passed in, so "does torch see the
    device" is only decidable here, with the container's real devices attached.
    First stop when conversions are unexpectedly slow: an empty accelerator
    list means everything is running on the CPU.
    """
    import torch
    accelerators = {}
    for name in ("xpu", "cuda"):
        backend = getattr(torch, name, None)
        if backend is None:
            continue
        try:
            available = bool(backend.is_available())
            accelerators[name] = {
                "available": available,
                "devices": backend.device_count() if available else 0,
            }
        except Exception as e:  # a broken driver raises rather than returning False
            accelerators[name] = {"available": False, "error": str(e)}
    return {
        "torch": torch.__version__,
        "accelerators": accelerators,
        "device": ACCELERATOR or "cpu",
        "iw3_gpu": IW3_GPU,
        "iw3_gpu_setting": IW3_GPU_SETTING,
        "warning": DEVICE_WARNING,
        "ffmpeg": shutil.which(FFMPEG_BIN),
        "preview_clip_seconds": PREVIEW_CLIP_SECONDS,
        "input_root": str(INPUT_ROOT),
        "output_root": str(OUTPUT_ROOT),
    }


@app.get("/api/throughput")
def throughput():
    """The rates the queue's ETAs are built on, and where each one came from."""
    measured = _throughput()
    keys = sorted(set(measured) | set(SEED_THROUGHPUT_FPS))
    return {
        "rates": [
            {
                "depth_model": model,
                "resolution": bucket,
                "fps": round(measured.get((model, bucket), SEED_THROUGHPUT_FPS.get((model, bucket))), 2),
                "source": "measured here" if (model, bucket) in measured else "seed (Arc Pro B60)",
            }
            for model, bucket in keys
        ],
        "jobs_measured": sum(1 for s in _rate_samples.values() if s),
    }


@app.get("/api/browse")
def browse(path: str = ""):
    target = _safe_input_path(path)
    if not target.exists():
        raise HTTPException(404, "not found")
    if target.is_file():
        raise HTTPException(400, "not a directory")
    entries = []
    for entry in sorted(target.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
        rel = str(entry.relative_to(INPUT_ROOT))
        entries.append({
            "name": entry.name,
            "path": rel,
            "type": "dir" if entry.is_dir() else "file",
        })
    return {"path": str(target.relative_to(INPUT_ROOT)) if target != INPUT_ROOT else "", "entries": entries}


@app.get("/api/jobs")
def list_jobs():
    # Two blocks, because they answer different questions and want opposite
    # orders: what is going to happen (queue order, all of it - truncating the
    # thing you are about to reorder would be its own bug), then what already
    # happened (newest first, capped).
    with _db() as conn:
        active = conn.execute(
            "SELECT * FROM jobs WHERE status IN ('running','queued') "
            "ORDER BY status <> 'running', position, created_at").fetchall()
        history = conn.execute(
            "SELECT * FROM jobs WHERE status NOT IN ('running','queued') "
            "ORDER BY created_at DESC LIMIT 200").fetchall()
    rows = list(active) + list(history)

    jobs = []
    queue_index = 0
    for r in rows:
        job = dict(r)
        job["progress"] = None
        job["eta_sec"] = None
        job["eta_estimated"] = False
        job["queue_index"] = None

        if job["status"] == "running":
            progress = _job_progress.get(job["id"])
            job["progress"] = progress
            eta = _running_eta(r, progress)
            job["eta_sec"] = eta[0]
            job["eta_estimated"] = eta[1]
        elif job["status"] == "queued":
            queue_index += 1
            job["queue_index"] = queue_index
            job["eta_sec"] = _estimate_seconds(r)
            job["eta_estimated"] = job["eta_sec"] is not None
        jobs.append(job)

    return jobs


@app.get("/api/queue-eta")
def queue_eta():
    """Total time left: the running job's own ETA plus estimates for the rest.

    Sorted by position (same as _worker_loop), so the numbers line up with the
    order things will actually run in.
    """
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status IN ('running','queued') "
            "ORDER BY status <> 'running', position, created_at"
        ).fetchall()

    total = 0.0
    exact = True  # false once any part of the sum is a model-based estimate
    counted = 0
    for r in rows:
        if r["status"] == "running":
            seconds, estimated = _running_eta(r, _job_progress.get(r["id"]))
            if estimated:
                exact = False
        else:
            seconds = _estimate_seconds(r)
            exact = False
        if seconds is None:
            # Unprobeable file (or a recursive folder job): can't be counted,
            # so say so rather than silently understating the total.
            continue
        total += seconds
        counted += 1

    return {
        "jobs": len(rows),
        "jobs_counted": counted,
        "total_sec": round(total) if counted else None,
        "exact": exact,
    }


@app.post("/api/jobs")
def create_job(job: JobCreate):
    if job.mode not in ("convert", "preview"):
        raise HTTPException(400, "mode must be convert or preview")
    if not job.input_path.strip():
        raise HTTPException(400, "input_path must not be empty (refusing to default to the whole /input root)")
    target = _safe_input_path(job.input_path)
    if not target.exists():
        raise HTTPException(400, f"input path does not exist: {job.input_path}")
    if job.stereo_format not in {s["value"] for s in STEREO_FORMATS}:
        raise HTTPException(400, "unknown stereo_format")
    # A preview is one clip cut out of one file - there is no meaningful
    # "middle" of a folder.
    if job.mode == "preview" and (job.recursive or target.is_dir()):
        raise HTTPException(400, "preview works on a single video file, not a folder")

    level = QUALITY_BY_ID.get(job.quality)
    if level is None:
        raise HTTPException(400, f"unknown quality level: {job.quality}")
    if level["chain"]:
        if not _chain_available():
            raise HTTPException(
                400, f"quality level '{job.quality}' needs the multi-stage pipeline, "
                     f"and {CHAIN_SCRIPT} is not installed or not executable. Set "
                     f"IW3_CHAIN_SCRIPT, or pick a single-pass level.")
        if job.recursive:
            # The pipeline takes one video and one scratch directory. A folder
            # of them is a queue's job, not a script's.
            raise HTTPException(400, f"quality level '{job.quality}' converts one file "
                                     f"at a time; queue the files individually")
    elif job.flow or job.upscale:
        raise HTTPException(400, "the flow and upscale options are stages of the "
                                 "multi-stage pipeline; they do nothing on a "
                                 "single-pass level")
    # A named level *is* its parameter set, stored so the row keeps saying what
    # ran even if the table above is later corrected.
    params = dict(level["params"]) if level.get("params") else dict(job.params)

    job_id = str(uuid.uuid4())
    with _db() as conn:
        conn.execute(
            "INSERT INTO jobs (id, mode, input_path, recursive, stereo_format, params_json, "
            "status, created_at, position, quality, opt_flow, opt_upscale) "
            "VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)",
            (job_id, job.mode, job.input_path, int(job.recursive), job.stereo_format,
             json.dumps(params), _now(), _next_position(conn),
             job.quality, int(job.flow), int(job.upscale)),
        )
        # Previews jump the queue. The whole point of a preview is to see the
        # settings before committing the hours a full conversion costs - behind
        # a queue that is days deep it would answer the question long after the
        # question stopped mattering.
        #
        # This is now an insertion rule rather than a sort rule, and that is the
        # point: the queue has exactly one order, the one on screen. A preview
        # lands in front, and from then on it can be dragged like any other row
        # instead of being pinned there by a sort key nobody can see.
        if job.mode == "preview":
            modes = {r["id"]: r["mode"] for r in
                      conn.execute("SELECT id, mode FROM jobs WHERE status='queued'")}
            ids = [i for i in _queued_ids(conn) if i != job_id]
            at = 0
            while at < len(ids) and modes.get(ids[at]) == "preview":
                at += 1
            ids.insert(at, job_id)
            _apply_queue_order(conn, ids)
        conn.commit()
    return {"id": job_id}


class QueueOrder(BaseModel):
    order: list[str]


@app.post("/api/queue/reorder")
def reorder_queue(body: QueueOrder):
    """Apply a client-supplied queue order (the drag-and-drop endpoint).

    A merge, not an assignment. The order the browser sends is the order it was
    *showing*, which is up to a refresh interval stale: jobs may have started,
    finished, been canceled or been added since. So ids the server no longer
    has queued are dropped, and queued ids the client never saw keep their
    relative order behind the ones it did. Dragging one row can then never
    resurrect, reorder or lose a job the user could not see.
    """
    with _db() as conn:
        current = _queued_ids(conn)
        known = set(current)
        seen = set()
        wanted = []
        for job_id in body.order:
            if job_id in known and job_id not in seen:
                seen.add(job_id)
                wanted.append(job_id)
        rest = [i for i in current if i not in seen]
        _apply_queue_order(conn, wanted + rest)
        conn.commit()
        return {"order": _queued_ids(conn)}


class JobMove(BaseModel):
    to: str  # "top" | "bottom" | "up" | "down"


@app.post("/api/jobs/{job_id}/move")
def move_job(job_id: str, body: JobMove):
    """Move one job within the queue.

    Dragging is fine over a screenful. Over a queue a hundred deep, "run this
    next" is a button, not an exercise in scrolling while holding the mouse
    down - which is the case this whole feature exists for.
    """
    with _db() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "not found")
        if row["status"] != "queued":
            # The running job is already on the GPU and nothing here interrupts
            # it; a finished one is not in the queue to begin with.
            raise HTTPException(409, f"only a queued job can be moved (this one is {row['status']})")
        ids = _queued_ids(conn)
        at = ids.index(job_id)
        ids.pop(at)
        targets = {"top": 0, "bottom": len(ids),
                    "up": max(0, at - 1), "down": min(len(ids), at + 1)}
        if body.to not in targets:
            raise HTTPException(400, "to must be one of: top, bottom, up, down")
        at = targets[body.to]
        ids.insert(at, job_id)
        _apply_queue_order(conn, ids)
        conn.commit()
        return {"position": at + 1, "queued": len(ids)}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    with _db() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "not found")
        if row["status"] == "queued":
            conn.execute("UPDATE jobs SET status='canceled', finished_at=? WHERE id=?", (_now(), job_id))
            conn.commit()
            return {"ok": True}
    proc = _running_procs.get(job_id)
    if proc is not None:
        proc.send_signal(signal.SIGTERM)
        return {"ok": True, "note": "SIGTERM sent to running job"}
    raise HTTPException(409, "job is not queued or running")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with _db() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "not found")
        if row["status"] in ("queued", "running"):
            raise HTTPException(409, "cancel the job first")
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        conn.commit()
    log_path = LOG_DIR / f"{job_id}.log"
    log_path.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/log")
async def stream_log(job_id: str):
    log_path = LOG_DIR / f"{job_id}.log"

    async def gen():
        # Replay what's already on disk first. Capped and batched into a
        # handful of SSE messages, not one per line - a multi-hour job's log
        # can run into the thousands of lines, and sending each as its own
        # event/DOM append is what crashed the browser tab.
        REPLAY_MAX_LINES = 500
        REPLAY_BATCH_SIZE = 50
        if log_path.exists():
            with open(log_path) as f:
                lines = f.readlines()
            if len(lines) > REPLAY_MAX_LINES:
                omitted = len(lines) - REPLAY_MAX_LINES
                lines = [f"[... {omitted} earlier lines omitted ...]\n"] + lines[-REPLAY_MAX_LINES:]
            for i in range(0, len(lines), REPLAY_BATCH_SIZE):
                batch = "".join(lines[i:i + REPLAY_BATCH_SIZE])
                yield f"data: {json.dumps(batch)}\n\n"

        with _db() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["status"] not in ("queued", "running"):
            yield "event: eof\ndata: {}\n\n"
            return

        q = asyncio.Queue()
        _log_subscribers.setdefault(job_id, set()).add(q)
        try:
            while True:
                line = await q.get()
                if line == "__EOF__":
                    yield "event: eof\ndata: {}\n\n"
                    break
                yield f"data: {json.dumps(line)}\n\n"
        finally:
            _log_subscribers.get(job_id, set()).discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
