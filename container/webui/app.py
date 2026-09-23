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
import subprocess
import threading
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

# ---------------------------------------------------------------------------
# Preview mode
#
# A preview converts a short window of the source with the very settings the
# job carries. It used to pass iw3's --keyframe instead, which only ever
# produced stills - and stills cannot show the one thing a 3D preview has to
# answer: whether depth stays stable while the picture moves.
#
# Three numbers define the window, and only the first one is cosmetic:
#
#   PREVIEW_SECONDS   how much is worth watching. Every comparison clip cut
#                     during the recipe work was 60 s and that turned out to
#                     be enough to decide by, twice a day for a week.
#
#   PREVIEW_LEAD      frames run through the pipeline *before* the visible
#                     window and thrown away afterwards. This one is not
#                     optional: the convergence estimator carries an
#                     exponential moving average across frames, so a window
#                     started cold is not the same recipe as the full run.
#                     200 frames is what the recipe work used throughout.
#
#   PREVIEW_TAIL      frames after the window, thrown away as well. The depth
#                     band swap works over a radius of 30 frames and has no
#                     material past the end of the clip; without a tail the
#                     last second of a preview carries an edge effect the full
#                     run does not have.
#
# So a 60 s preview of a 29.97 fps source costs 200 + 1798 + 30 = 2,028 frames
# against 21,606 for the whole file - 9.4%, and that is the figure the UI
# quotes, computed the same way for whatever file is selected.
# ---------------------------------------------------------------------------
PREVIEW_SECONDS = float(os.environ.get("PREVIEW_SECONDS", "60"))
PREVIEW_LEAD_FRAMES = int(os.environ.get("PREVIEW_LEAD_FRAMES", "200"))
PREVIEW_TAIL_FRAMES = int(os.environ.get("PREVIEW_TAIL_FRAMES", "30"))
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# Hardware encoder for the two jobs this app does itself - trimming the lead
# frames off a finished preview, and building a comparison file out of two of
# them. Both are re-encodes of a minute or three of video and neither is worth
# a CPU hour. The encoder is probed once, with two seconds of test pattern,
# because `ffmpeg -encoders` lists what was compiled in, not what this
# container can actually reach: the device node has to be passed in too.
QSV_FFMPEG = os.environ.get("QSV_FFMPEG", "/usr/lib/jellyfin-ffmpeg/ffmpeg")
QSV_DEVICE = os.environ.get("QSV_DEVICE", "/dev/dri/renderD128")
QSV_GLOBAL_QUALITY = os.environ.get("QSV_GLOBAL_QUALITY", "17")
# Delivery files are read by whoever browses the share, not by this container.
OUTPUT_UID = int(os.environ.get("OUTPUT_UID", "99"))
OUTPUT_GID = int(os.environ.get("OUTPUT_GID", "100"))
# A label bar is burned into the comparison clips. It is drawn per eye, at the
# same coordinates in each, and it is opaque: a translucent one would let the
# two eyes' differing pixels through and flicker in stereo.
LABEL_BAR_HEIGHT = 72
LABEL_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]

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
# Every level therefore names a runtime model (`costs`: "single" for one iw3
# process, "chain" for the pipeline, None for custom), and the UI turns that
# into a time for the video in front of the user instead of quoting the hours
# the measurement took.
#
# The first measurements were taken on an Intel Arc Pro B60 over 21,606 frames
# of 1920x1080 video (12 min at 29.97 fps). The runtime model itself is below
# ("What a job costs"); `notes` says what each level's figure rests on.
# ---------------------------------------------------------------------------
MEASURED_ON_FRAMES = 21606

# ---------------------------------------------------------------------------
# What a job costs
#
# The first model was one rate per level, measured on one 1080p film and
# multiplied by 2.54 for anything classed as 4K. Six full pipeline runs later
# that turned out to be wrong in three separate ways:
#
#   * The stages do not scale alike. DepthPro works at a fixed 1536x1536 and
#     costs the same per frame at 1080p and 1440p (0.373-0.382 s); VDA_B
#     barely moves either. The band swap, the warp, the encode and the frame
#     count after it scale with the pixels (1.6-2.1x at 1440p).
#   * "4K" was decided by height alone, so a 1080x1920 portrait clip was
#     quoted at 4K rates. Six portrait jobs on this machine ran at exactly the
#     HD rate (0.151-0.203 s/frame). What costs is the pixel count.
#   * The encode stage ends with `ffprobe -count_frames` over the delivery
#     file, which writes nothing and is in no stage's time: 1,087 s after a
#     471 s encode on a 29,122-frame 1440p film. It is modelled here.
#
# So a pipeline job is costed stage by stage, each as a fixed start-up plus a
# per-frame rate that is a straight line in the pixel count through the two
# resolutions that have actually been run (1080p and 1440p). Outside that range
# the line is an extrapolation and says so. The per-frame figures are means
# over the full-length runs, not medians: the runs disagree by up to 25 % per
# stage (a GPU shared with other work is the likely reason, not a proven one),
# and a wait quoted too short is the more annoying error.
#
# Measured runs, 1080p: the validation film (21,606 frames), two films of
# 7,906 and 9,476 frames; 1440p: one film of 29,122 frames and one preview of
# 2,041. Fixed costs from a 300-frame run, where they dominate.
# ---------------------------------------------------------------------------
REF_PIXELS = 1920 * 1080
PX_1440 = 2560 * 1440 / REF_PIXELS          # 1.778
CHAIN_MEASURED_PX = (0.9, 1.9)              # outside this the line is extrapolated


def _px_of(info):
    """Source pixel count relative to 1080p (portrait and landscape alike)."""
    w, h = (info or {}).get("width") or 0, (info or {}).get("height") or 0
    if not w or not h:
        return 1.0
    return w * h / REF_PIXELS


def _px_line(hd, q1440):
    """s/frame(px) through the 1080p and 1440p figures, never below a quarter of HD."""
    slope = (q1440 - hd) / (PX_1440 - 1.0)
    return lambda px: max(0.25 * hd, hd + slope * (px - 1.0))


# One entry per stage the pipeline script announces, in its order. `log` is
# the file the stage's own progress bar goes to (log/<name> under the job's
# scratch directory); `match` recognises the stage from its marker label, so
# the script's wording can change without breaking the lookup.
CHAIN_STAGES = {
    "bilder":    {"match": r"^Bilder",     "log": None,          "fixed": 1.0,
                  "rate": _px_line(0.00325, 0.0054)},
    "hoch":      {"match": r"^waifu2x",    "log": "hoch.txt",    "fixed": 0.0,
                  "rate": _px_line(0.870, 0.870 * PX_1440), "unmeasured": True},
    "feinband":  {"match": r"^DepthPro",   "log": "pro.txt",     "fixed": 40.0,
                  "rate": _px_line(0.3741, 0.3820)},
    "vda":       {"match": r"^VDA_B",      "log": "vda.txt",     "fixed": 15.0,
                  "rate": _px_line(0.1145, 0.1229)},
    "band":      {"match": r"^Bandtausch", "log": "band.txt",    "fixed": 8.0,
                  "rate": _px_line(0.0523, 0.0940)},
    "fluss":     {"match": r"^Fluss",      "log": "fluss.txt",   "fixed": 0.0,
                  "rate": _px_line(0.1777, 0.1777 * PX_1440), "unmeasured": True},
    "export":    {"match": r"^Export",     "log": "exp_*.txt",   "fixed": 1.0,
                  "rate": _px_line(0.0017, 0.0015)},
    "warp":      {"match": r"^Warp",       "log": "warp_*.txt",  "fixed": 30.0,
                  "rate": _px_line(0.0986, 0.1700)},
    # The encode itself plus the ffprobe frame count after it; neither
    # writes any progress, so this stage is always the model.
    "kodierung": {"match": r"^Kodierung",  "log": None,          "fixed": 3.0,
                  "rate": _px_line(0.0084 + 0.0223, 0.0140 + 0.0375)},
}
# DepthPro_S against DepthPro on the same 300 frames: 56 s against 115 s.
# Economical has never run end to end through the pipeline, so every
# economical figure is marked as an extrapolation.
DEPTHPRO_S_FACTOR = 56.0 / 115.0
# Deleting the scratch directory after a successful job: 36 s for 29,122
# frames at 1440p, 5-6 s for 8-9k frames at 1080p.
CHAIN_CLEANUP = _px_line(0.0007, 0.0012)
# The stages that work on the warped RGB. With the upscale switch that is
# twice the width and twice the height - never measured, so extrapolated.
UPSCALED_STAGES = ("export", "warp", "kodierung")

# The single-pass level is one iw3 process, so it is one rate - but not a
# straight line in the pixels: medians over this machine's 218 finished
# single-pass VDA_B jobs, by pixel count relative to 1080p.
#     0.43 px: 0.127 s (n=9)   1.00: 0.174 (n=145)   1.78: 0.313 (n=2)   4.00: 0.375 (n=66)
# The 1440p point rests on two jobs and one preview (0.331 s); it is the
# least certain of the four.
SINGLE_PASS_POINTS = [(0.43, 0.127), (1.0, 0.1737), (PX_1440, 0.3132), (4.0, 0.375)]
SINGLE_PASS_FIXED = 50.0


def _single_pass_rate(px):
    pts = SINGLE_PASS_POINTS
    if px <= pts[0][0]:
        return max(0.08, pts[0][1] * px / pts[0][0])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if px <= x1:
            return y0 + (y1 - y0) * (px - x0) / (x1 - x0)
    return pts[-1][1] * px / pts[-1][0]


# What a preview costs on top of its frames: the motion scan over the whole
# source when the window is picked automatically (15 s for 15,410 frames at
# 1080p, 62-66 s for 29,122 at 1440p), and cutting and trimming the clip.
PREVIEW_SCAN = _px_line(0.001, 0.0022)
PREVIEW_OVERHEAD_SEC = 20.0

QUALITY_LEVELS = [
    {
        "id": "fast",
        "label": "Fast",
        "chain": False,
        "costs": "single",  # see "What a job costs"
        "uses": "One pass of iw3 with the small depth model VDA_B and its own "
                "flicker smoothing. No second depth model, no post-processing "
                "of the depth maps, nothing after the warp.",
        "good_for": "Deciding whether a film is worth converting at all, and for "
                    "material you will watch once.",
        "pros": [
            "58 minutes for a 12-minute film - a quarter of what Standard costs, "
            "and the only level that finishes inside a lunch break.",
            "Temporally the calmest of the three: 0.0382 px of flicker on still "
            "surfaces, against 0.0452 for Standard and 0.0468 for Economical.",
            "One process rather than five, so it cancels cleanly and leaves "
            "nothing behind.",
        ],
        "cons": [
            "39% less relief on silhouettes than the two pipeline levels. This is "
            "the one difference that was picked out in a headset immediately, "
            "without being told what to look for.",
            "About half the fine detail: fine-band correlation 0.45 against raw "
            "DepthPro, where the pipeline levels reach 0.85 and 0.999.",
            "Cannot take the optical-flow or upscale switches - they are stages "
            "of the pipeline and this level is not one.",
        ],
        "notes": "Measured wall clock including the x265 encode: 57 min 37 s for "
                 "21,606 frames, cold and warm runs 0.6% apart. The time shown for "
                 "a file comes from this machine's 218 finished single-pass jobs, "
                 "by the file's pixel count.",
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
        "costs": "chain",
        "fine_model": "DepthPro_S",
        "uses": "The reduced DepthPro_S for fine structure and VDA_B with "
                "flicker smoothing for the coarse band, swapped into one "
                "another over a 30-frame radius, then warped with mlbw_l2.",
        "good_for": "A long film where an hour of machine time is worth more "
                    "than the last few percent of edge sharpness.",
        "pros": [
            "An hour faster than Standard - 3 h 18 against 4 h 19 for a "
            "12-minute film - for the same pipeline and the same warp.",
            "Keeps silhouettes within 8% of Standard. Fast is 39% off; this is "
            "a different order of loss entirely.",
            "Takes both switches, so it can be pushed up rather than replaced.",
        ],
        "cons": [
            "Softer outlines than Standard: 8.0% less relief, which showed up at "
            "an elbow in side-by-side viewing rather than across the picture.",
            "Markedly less fine structure - fine band 0.8534 against Standard's "
            "0.9988. This is measured, not seen: asked directly, the difference "
            "in fine structure was not visible in a headset.",
            "Still three times the cost of Fast, and slightly more flicker on "
            "still surfaces than either other level (0.0468 px).",
        ],
        "notes": "Never run end to end through the pipeline: priced as Standard "
                 "with DepthPro_S at 0.49 of DepthPro's time (measured on 300 "
                 "frames), so every figure for it is an extrapolation.",
    },
    {
        "id": "standard",
        "label": "Standard",
        "chain": True,
        "default": True,
        "costs": "chain",
        "fine_model": "DepthPro",
        "uses": "DepthPro at full resolution for fine structure and VDA_B with "
                "flicker smoothing for the coarse band, swapped into one "
                "another over a 30-frame radius, then warped with mlbw_l2.",
        "good_for": "Anything you intend to keep. This is the recipe a week of "
                    "side-by-side viewing settled on.",
        "pros": [
            "Keeps the depth model's fine band essentially intact - 0.9988 "
            "against raw DepthPro, where Economical reaches 0.8534.",
            "The sharpest silhouettes on offer: 1.2% ahead of Fast and 8% ahead "
            "of Economical.",
            "The only level chosen by looking rather than by measuring; it was "
            "frozen on 21 September after every rival had been watched against "
            "it in a headset.",
        ],
        "cons": [
            "The most expensive level that is not an optional extra: 4 h 19 for "
            "a 12-minute film, four and a half times Fast.",
            "3 h 28 of that is GPU time the machine can do nothing else with, "
            "and the remaining 50 minutes are CPU-bound stages that run after "
            "it rather than alongside.",
            "18% more flicker on still surfaces than Fast (0.0452 against "
            "0.0382). Side by side that gap sat below the threshold of being "
            "seen at all - it is a real number that buys nothing visible.",
        ],
        "notes": "Priced stage by stage from five full pipeline runs on this "
                 "machine (1080p and 1440p). The 12-minute validation film took "
                 "4 h 13 end to end.",
    },
    {
        "id": "custom",
        "label": "Custom (all settings)",
        "chain": False,
        "costs": None,  # estimated from this machine's finished jobs instead
        "uses": "Whatever you put in the form below - every iw3 setting, exactly "
                "as this UI has always offered them.",
        "good_for": "Trying something none of the three levels covers.",
        "pros": [
            "The only way to reach settings the levels fix, such as a different "
            "depth model, a different divergence or a lossless codec.",
        ],
        "cons": [
            "Not a measured recipe, so nothing here promises anything about the "
            "result - the three levels each stand on a full-length run.",
            "The time shown is read back from finished jobs on this machine per "
            "depth model, which is a throughput figure rather than a model of "
            "this particular combination of settings.",
        ],
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
        "stage": "fluss",
        "uses": "An extra RAFT optical-flow stage that smooths each depth map "
                "against the one before it.",
        "pros": [
            "10.6% less flicker on still surfaces, measured on the same clip "
            "the levels were measured on.",
        ],
        "cons": [
            "An hour of GPU per 12-minute film for a difference that was at the "
            "edge of being visible: watched A/B/A in a headset the verdict was "
            "\"super hard to tell\".",
            "It is kept as a switch because fast motion, hard cuts and "
            "low-contrast material were never tested with it off.",
        ],
        "notes": "Measured 1 h 04 over 21,606 frames.",
    },
    {
        "id": "upscale",
        "label": "Denoise and 2x upscale before the warp",
        "chain_only": True,
        "stage": "hoch",
        "uses": "A waifu2x denoise-and-double pass over the RGB frames before "
                "the warp. The depth maps stay at source resolution - DepthPro "
                "works at a fixed 1536x1536 internally, so there is no extra "
                "depth detail to be had from upscaling them.",
        "pros": [
            "Genuinely removes compression artefacts - the top frequency band "
            "drops 15% against the source, which is the part of the picture an "
            "h264 encoder threw away.",
        ],
        "cons": [
            "5 h 13 for the upscaling stage alone, and the warp afterwards then "
            "works on four times the pixels, which has never been measured. The "
            "time shown for this switch is therefore too low, not too high.",
            "It adds no detail the source did not carry: against an 8 Mbit/s "
            "source the entire gain sat above what the source could hold, and "
            "in a headset the verdict was \"rather underwhelming\".",
            "A better source beats any upscaler, and costs nothing to convert.",
        ],
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


def _chain_plan(flow=False, upscale=False):
    """The pipeline's stages in the order the script runs and numbers them."""
    plan = ["bilder"]
    if upscale:
        plan.append("hoch")
    plan += ["feinband", "vda", "band"]
    if flow:
        plan.append("fluss")
    plan += ["export", "warp", "kodierung"]
    return plan


def _stage_seconds(key, frames, px, level=None, upscale=False):
    """Model wall clock of one pipeline stage over `frames` frames."""
    st = CHAIN_STAGES[key]
    if upscale and key in UPSCALED_STAGES:
        px = px * 4
    fixed, rate = st["fixed"], st["rate"](px)
    if key == "feinband" and level and level.get("fine_model") == "DepthPro_S":
        fixed, rate = fixed * DEPTHPRO_S_FACTOR, rate * DEPTHPRO_S_FACTOR
    return fixed + frames * rate


def _level_seconds(level_id, frames, px, flow=False, upscale=False):
    """Estimated wall clock for `frames` at this level, or None if unmodelled.

    `px` is the source's pixel count relative to 1080p. Returns
    (seconds, extrapolated) - extrapolated whenever the figure rests on
    something that has not been run on this machine at this size.
    """
    level = QUALITY_BY_ID.get(level_id)
    if not level or not level.get("costs") or not frames:
        return None, False
    if level["costs"] == "single":
        lo, hi = SINGLE_PASS_POINTS[0][0], SINGLE_PASS_POINTS[-1][0]
        return (SINGLE_PASS_FIXED + frames * _single_pass_rate(px),
                not (lo * 0.9 <= px <= hi * 1.05))
    plan = _chain_plan(flow, upscale)
    seconds = sum(_stage_seconds(k, frames, px, level, upscale) for k in plan)
    seconds += frames * CHAIN_CLEANUP(px)
    extrapolated = (not (CHAIN_MEASURED_PX[0] <= px <= CHAIN_MEASURED_PX[1])
                    or level.get("fine_model") == "DepthPro_S"
                    or any(CHAIN_STAGES[k].get("unmeasured") for k in plan)
                    or upscale)
    return seconds, extrapolated


def _preview_extra_seconds(info, px, auto_window=True):
    """What a preview costs besides its frames: the window scan and the clip work."""
    extra = PREVIEW_OVERHEAD_SEC
    if auto_window and info and info.get("duration_sec") and info.get("fps"):
        extra += info["duration_sec"] * info["fps"] * PREVIEW_SCAN(px)
    return extra

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
                opt_upscale INTEGER NOT NULL DEFAULT 0,
                preview_start_sec REAL,
                output_path TEXT
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
                           ("opt_upscale", "INTEGER NOT NULL DEFAULT 0"),
                           # Null, not 0: "no window was chosen by hand" and
                           # "the window starts at second zero" are different
                           # statements and the second one is legal.
                           ("preview_start_sec", "REAL"),
                           ("output_path", "TEXT"),
                           # The comparison a preview was queued for, if any.
                           # Older code ignores the column.
                           ("parent_id", "TEXT")):
            if name not in columns:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")
        if "position" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN position INTEGER NOT NULL DEFAULT 0")
            seed = conn.execute(
                "SELECT id FROM jobs WHERE status IN ('running','queued') "
                "AND mode<>'compare' ORDER BY mode <> 'preview', created_at"
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
# Jobs somebody pressed Cancel on while they ran. A cancelled process exits
# non-zero (143 from the pipeline, -15 from iw3), and without this record the
# job would be filed as failed - with its scratch kept "for inspection" as if
# something had gone wrong. In memory only: a restart re-queues running jobs.
_cancel_requested: set[str] = set()
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
        "SELECT id FROM jobs WHERE status='queued' AND mode<>'compare' "
        "ORDER BY position, created_at")]


def _apply_queue_order(conn, ids):
    """Renumber the queue to `ids`, running job first."""
    running = [r["id"] for r in conn.execute(
        "SELECT id FROM jobs WHERE status='running' AND mode<>'compare' "
        "ORDER BY position, created_at")]
    for i, job_id in enumerate(running + list(ids), start=1):
        conn.execute("UPDATE jobs SET position=? WHERE id=?", (i, job_id))


def _next_position(conn):
    row = conn.execute("SELECT MAX(position) AS m FROM jobs "
                        "WHERE status IN ('running','queued') "
                        "AND mode<>'compare'").fetchone()
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
    label = m.group("label").strip()
    return {
        "phase": "chain",
        "stage": int(m.group("stage")),
        "stages": int(m.group("stages")),
        "stage_label": label,
        "stage_key": _stage_key(label),
        "stage_started": time.time(),
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


# 4K by pixel count, not by one side: a 1080x1920 portrait clip has the pixels
# of 1080p and converts at the 1080p rate (six such jobs here, 0.151-0.203
# s/frame). The line sits where the old height rule sat for landscape 16:9
# (2844x1600, about 4.5 MP), so every landscape file keeps its bucket.
BUCKET_4K_PIXELS = 4_000_000


def _bucket(info):
    w, h = (info or {}).get("width") or 0, (info or {}).get("height") or 0
    if not w:
        # A probe without a width (should not happen) falls back to the old rule.
        return "4k" if h >= 1600 else "hd"
    return "4k" if w * h >= BUCKET_4K_PIXELS else "hd"


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
    return model, _bucket(info), frames / elapsed


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
    if progress and progress.get("phase") == "chain":
        left = _chain_eta(job_row, progress)
        if left is not None:
            return left, True
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


# ---------------------------------------------------------------------------
# Progress inside a pipeline stage
#
# Every stage writes its progress bar to log/<stage>.txt in the job's scratch
# directory and nothing to stdout (the script starts each one as
# `"$@" > "$_log" 2>&1 &`), so the stdout this app parses only ever carried
# the stage markers. The stage
# bar is therefore read where it is written: the tail of the current stage's
# log, a few kilobytes, on each poll. Three shapes occur -
#
#   tqdm, from every iw3/waifu2x stage   "Images:  34%|###  | 5240/15410 [32:38<1:02:20,  2.72it/s]"
#   the band swap's own counter          "  1800/2021  0.064 s/Bild  Rest 0.2 min"
#   the flow stage's bare counter        "  1950/2021"
#
# Two stages write nothing at all: frame extraction (seconds) and the encode,
# whose closing ffprobe frame count is the longest silent stretch of the job.
# Those are the model and nothing else - and marked as such.
#
# 🔴 Nothing here may tick on its own. A watcher outside this app decides
# "stuck" by whether frames/percent/elapsed move; a field that advances with
# the wall clock would make a hung stage look alive. The time since the stage
# began is therefore reported separately as `stage_elapsed_sec`.
# ---------------------------------------------------------------------------
_BAND_LINE = re.compile(r"^\s*(\d+)/(\d+)\s+([0-9.]+)\s*s/Bild\s+Rest\s+([0-9.]+)\s*min")
_COUNT_LINE = re.compile(r"^\s*(\d+)/(\d+)\s*$")
_CHAIN_FRAMES = re.compile(r"^\s+Bilder:\s+(\d+)\s*$")
LOG_TAIL_BYTES = 16384


def _stage_key(label):
    for key, st in CHAIN_STAGES.items():
        if re.match(st["match"], label or ""):
            return key
    return None


def _stage_log_progress(job_id, key, since):
    """Latest progress line from a stage's own log, or None."""
    pattern = CHAIN_STAGES.get(key, {}).get("log")
    if not pattern:
        return None
    logdir = _chain_work_dir(job_id) / "log"
    try:
        candidates = [(p.stat().st_mtime, p) for p in logdir.glob(pattern)]
    except OSError:
        return None
    if not candidates:
        return None
    mtime, path = max(candidates)
    # A log from before this stage started belongs to an earlier attempt.
    if since and mtime < since - 5:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - LOG_TAIL_BYTES))
            tail = f.read().decode(errors="replace")
    except OSError:
        return None
    for line in reversed(re.split(r"[\r\n]+", tail)):
        if not line.strip():
            continue
        parsed = _parse_progress(line)
        if parsed and parsed.get("total_frames"):
            return parsed
        m = _BAND_LINE.match(line)
        if m:
            n, total, spf, rest = int(m[1]), int(m[2]), float(m[3]), float(m[4])
            return {"phase": "convert", "frames": n, "total_frames": total,
                    "percent": round(100.0 * n / total, 1) if total else None,
                    "elapsed_sec": None, "eta_sec": rest * 60.0,
                    "rate_fps": round(1.0 / spf, 2) if spf else None}
        m = _COUNT_LINE.match(line)
        if m:
            n, total = int(m[1]), int(m[2])
            return {"phase": "convert", "frames": n, "total_frames": total,
                    "percent": round(100.0 * n / total, 1) if total else None,
                    "elapsed_sec": None, "eta_sec": None, "rate_fps": None}
    return None


def _chain_live_progress(job_id, progress):
    """The chain's progress with the current stage's own bar merged in."""
    if not progress or progress.get("phase") != "chain":
        return progress
    out = dict(progress)
    started = out.get("stage_started")
    out["stage_elapsed_sec"] = round(time.time() - started) if started else None
    out["stage_eta_sec"] = None
    out["stage_sub"] = None
    bar = _stage_log_progress(job_id, out.get("stage_key"), started)
    if bar:
        for key in ("percent", "frames", "total_frames", "elapsed_sec", "rate_fps"):
            out[key] = bar.get(key)
        if bar.get("phase") == "scene_detect":
            # VDA's pre-pass: its bar is real, its remaining time is not the stage's.
            out["stage_sub"] = "scene detection"
        elif bar.get("eta_sec") is not None:
            out["stage_eta_sec"] = bar["eta_sec"]
        elif bar.get("frames") and out["stage_elapsed_sec"]:
            # The flow counter carries no rate; the stage's own pace does.
            done, total = bar["frames"], bar["total_frames"]
            pace = out["stage_elapsed_sec"] / done
            out["rate_fps"] = round(1.0 / pace, 2) if pace else None
            out["stage_eta_sec"] = pace * (total - done)
    out["stage_source"] = "log" if out["stage_eta_sec"] is not None else "model"
    return out


def _chain_eta(job_row, progress):
    """Seconds left for a running pipeline job.

    The current stage's remaining time comes from its own progress bar when it
    has one, otherwise from the model minus the time the stage has run; the
    stages after it are the model. A stage that overruns its model is not
    quoted at zero - it is still running - but at a twentieth of its model.
    """
    try:
        info = _probe_video(_safe_input_path(job_row["input_path"]))
    except HTTPException:
        info = None
    if not info or not info.get("duration_sec") or not info.get("fps"):
        return None
    level = QUALITY_BY_ID.get(job_row["quality"], {})
    flow, upscale = bool(job_row["opt_flow"]), bool(job_row["opt_upscale"])
    px = _px_of(info)
    frames = progress.get("chain_frames") or _job_frames(job_row, info)
    plan = _chain_plan(flow, upscale)
    key = progress.get("stage_key")
    if key not in plan:
        idx = min(max((progress.get("stage") or 1) - 1, 0), len(plan) - 1)
        key = plan[idx]
    idx = plan.index(key)
    model_now = _stage_seconds(key, frames, px, level, upscale)
    if progress.get("stage_eta_sec") is not None:
        left = float(progress["stage_eta_sec"])
    else:
        ran = progress.get("stage_elapsed_sec") or 0.0
        left = max(model_now - ran, 0.05 * model_now)
    left += sum(_stage_seconds(k, frames, px, level, upscale) for k in plan[idx + 1:])
    left += frames * CHAIN_CLEANUP(px)
    if job_row["mode"] == "preview":
        left += PREVIEW_OVERHEAD_SEC / 2   # the trim after the pipeline
    return left


_probe_cache: dict[tuple, dict] = {}


def _probe_video(path: Path):
    """Duration/fps/width/height for a source file. Cached on (path, size, mtime)."""
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
                "width": stream.codec_context.width,
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

    frames = _job_frames(job_row, info)
    level = QUALITY_BY_ID.get(job_row["quality"], {})
    px = _px_of(info)
    extra = 0.0
    if job_row["mode"] == "preview":
        # The scan runs only when the window is left to the rule, and not at
        # all for a source too short to hold a window.
        auto = (job_row["preview_start_sec"] is None
                and not _preview_frames(info)["whole_source"])
        extra = _preview_extra_seconds(info, px, auto_window=auto)
    # A level with a runtime model of its own is estimated from that model:
    # it was measured as a whole recipe, which is a better description of it
    # than a per-depth-model rate could be.
    if level.get("costs"):
        seconds, _ = _level_seconds(job_row["quality"], frames, px,
                                    flow=bool(job_row["opt_flow"]),
                                    upscale=bool(job_row["opt_upscale"]))
        if seconds is not None:
            return seconds + extra
    params = json.loads(job_row["params_json"])
    bucket = _bucket(info)
    model = params.get("depth_model") or _defaults.get("depth_model")
    rate = (_throughput().get((model, bucket))
            or SEED_THROUGHPUT_FPS.get((model, bucket))
            or FALLBACK_FPS[bucket])
    return frames / rate + extra


def _job_frames(job_row, info):
    """Frames a job will push through the converter."""
    level = QUALITY_BY_ID.get(job_row["quality"], {})
    # The pipeline extracts every frame of the source; max_fps is an iw3
    # argument and the pipeline is not one iw3 call.
    if level.get("chain"):
        effective_fps = info["fps"]
    else:
        params = json.loads(job_row["params_json"])
        max_fps = params.get("max_fps") or _defaults.get("max_fps") or 30
        effective_fps = min(info["fps"], float(max_fps))
    # A preview only ever converts the extracted clip, so the source length
    # beyond it costs nothing. The lead and tail frames are counted, because
    # they are a tenth of the clip and they go through every stage.
    if job_row["mode"] == "preview":
        return _preview_frames(info, effective_fps)["total"]
    return info["duration_sec"] * effective_fps


def _preview_dir(job_id) -> Path:
    return OUTPUT_ROOT / "_previews" / job_id


def _preview_frames(info, fps=None):
    """How many frames a preview of this source costs, and how they split up.

    `visible` is what ends up in the file. `lead` and `tail` are run through
    every stage and then cut away again - see PREVIEW_LEAD_FRAMES above for why
    the lead is not optional. A source too short to hold all three is taken
    whole and gets no lead at all, which is stated in the job log.
    """
    fps = fps or info["fps"]
    total_frames = int(round(info["duration_sec"] * fps))
    visible = int(round(PREVIEW_SECONDS * fps))
    if total_frames <= visible + PREVIEW_LEAD_FRAMES + PREVIEW_TAIL_FRAMES:
        return {"visible": total_frames, "lead": 0, "tail": 0,
                "total": total_frames, "whole_source": True}
    return {"visible": visible, "lead": PREVIEW_LEAD_FRAMES,
            "tail": PREVIEW_TAIL_FRAMES,
            "total": visible + PREVIEW_LEAD_FRAMES + PREVIEW_TAIL_FRAMES,
            "whole_source": False}


# ---------------------------------------------------------------------------
# Which minute to preview
#
# The middle of the file was the old rule and it is a bad one: a preview is
# supposed to answer "does the depth hold up", and a minute in which nothing
# happens answers it falsely - every recipe looks fine on a static shot. The
# window that decided every recipe question during the measurement work was
# picked by an explicit rule instead, and this is that rule:
#
#   1. Only windows that fit a lead in front of them are candidates.
#   2. Of those, only ones that contain a cut. A cut is the hardest thing a
#      depth model has to survive, and a preview that never crosses one has
#      not been asked the question.
#   3. Of those, the ones within 1% of the highest average frame-to-frame
#      motion. Within that plateau the motion figure no longer separates
#      anything, so picking a winner by it would be an invented reason.
#   4. From the plateau, the window whose cut sits closest to the middle, so
#      both shots get roughly half a minute.
#   5. No cut anywhere in the film: drop condition 2 and take the
#      highest-motion window. The job log says which case applied.
#
# The motion figure is the mean absolute difference between consecutive
# frames, measured on a 160x90 greyscale decode. That is a proxy for the
# full-resolution measure the recipe work used, and it is a good one: on the
# reference film it finds the same single cut at frame 7394, the same 1,798
# candidate windows, and lands 46 frames - a second and a half - from the
# window that work chose by hand. The whole scan costs about 30 s for a
# 12-minute film, against 5 to 25 minutes for the preview it is choosing.
# ---------------------------------------------------------------------------
WINDOW_RULE = (
    "The default window is the minute with the most frame-to-frame motion that "
    "also contains a cut, positioned so the cut falls near its middle. A "
    "preview is there to answer whether the depth holds up, and a minute in "
    "which nothing happens answers it falsely - every recipe looks fine on a "
    "static shot. If the film has no cut at all, the highest-motion minute is "
    "taken instead and the job log says so.")

SCAN_WIDTH, SCAN_HEIGHT = 160, 90
# Both halves of the cut test matter: the ratio alone fires on a film that
# barely moves, the absolute alone fires on a hand-held one.
CUT_RATIO, CUT_ABS = 6.0, 12.0
PLATEAU = 0.99


async def _scan_motion(job_id, src: Path, logf):
    """Per-frame absolute difference over the whole source, as a numpy array."""
    import numpy as np
    argv = [FFMPEG_BIN, "-hide_banner", "-nostdin", "-loglevel", "error",
            "-i", str(src), "-an", "-sn",
            "-vf", f"scale={SCAN_WIDTH}:{SCAN_HEIGHT}:flags=fast_bilinear,format=gray",
            "-f", "rawvideo", "-"]
    await _write_log(job_id, logf, f"$ {' '.join(argv)}\n")
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    _running_procs[job_id] = proc
    frame_bytes = SCAN_WIDTH * SCAN_HEIGHT
    prev, diffs = None, []
    try:
        while True:
            buf = await proc.stdout.readexactly(frame_bytes)
            cur = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
            if prev is not None:
                diffs.append(float(np.abs(cur - prev).mean()))
            prev = cur
    except asyncio.IncompleteReadError:
        pass
    finally:
        await proc.wait()
        _running_procs.pop(job_id, None)
    return np.asarray(diffs, dtype=np.float64)


def _choose_window(diffs, visible, lead):
    """(first_visible_frame_0based, why) for the rule above, or None."""
    import numpy as np
    n_frames = len(diffs) + 1
    if n_frames < visible + lead:
        return None
    median = float(np.median(diffs)) if len(diffs) else 0.0
    cuts = np.nonzero((diffs > CUT_RATIO * median) & (diffs > CUT_ABS))[0] + 1

    starts = np.arange(lead, n_frames - visible + 1)
    # Mean motion per window, from a prefix sum - the alternative is 20,000
    # slices of a 20,000-element array.
    csum = np.concatenate([[0.0], np.cumsum(diffs)])
    lo, hi = starts, starts + visible - 1
    motion = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)

    # Does a cut fall inside [s, s+visible-1]? searchsorted rather than a loop,
    # so a film with thousands of cuts costs the same as one with a single cut.
    if len(cuts):
        first_at_or_after = np.searchsorted(cuts, starts, side="left")
        last_before_end = np.searchsorted(cuts, starts + visible - 1, side="right")
        pool = np.nonzero(last_before_end > first_at_or_after)[0]
    else:
        pool = np.array([], dtype=int)

    if len(pool):
        reason = "highest-motion window containing a cut, cut nearest its middle"
    else:
        pool = np.arange(len(starts))
        reason = ("no cut found in this film, so the highest-motion window "
                  "was taken instead")

    plateau = pool[motion[pool] >= motion[pool].max() * PLATEAU]
    if "no cut" in reason:
        return int(starts[max(plateau, key=lambda i: motion[i])]), reason

    middle = (visible - 1) / 2.0

    def cut_offset(i):
        s = starts[i]
        inside = cuts[(cuts >= s) & (cuts <= s + visible - 1)]
        return float(np.abs((inside - s) - middle).min())

    pick = min(plateau, key=lambda i: (cut_offset(i), -motion[i]))
    return int(starts[pick]), reason


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


async def _run_logged(job_id, argv, logf, shown=None):
    """Run a helper process, streaming its output into the job log.

    Registered in _running_procs like the conversion itself, so cancelling a
    job during clip extraction works the same way it does mid-conversion.
    `shown` replaces the command line in the log, for a helper whose argv is
    a whole script.
    """
    await _write_log(job_id, logf, f"$ {shown or ' '.join(argv)}\n")
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


async def _preview_plan(job_id, src: Path, job_row, logf):
    """Which frames of `src` this preview covers.

    `preview_start_sec` on the job overrides the rule: an explicit choice beats
    any heuristic, and the scan is skipped entirely, which also skips its 30
    seconds. Otherwise the rule above picks the window.
    """
    info = _probe_video(src)
    if not info or not info.get("duration_sec") or not info.get("fps"):
        raise RuntimeError("could not probe the source for duration and frame rate")
    fps = info["fps"]
    plan = _preview_frames(info)
    plan["fps"] = fps
    total_frames = int(round(info["duration_sec"] * fps))

    if plan["whole_source"]:
        plan.update(start_frame=0, chain_start_frame=0,
                    reason="source too short for a window, taken whole")
        return plan

    manual = job_row["preview_start_sec"]
    if manual is not None:
        wanted = int(round(float(manual) * fps))
        # Clamped rather than refused: the lead has to fit in front and the
        # tail behind, and a request a second outside that is a rounding
        # difference, not a different intention.
        start = _clamp_start(info, plan, manual)
        if job_row["parent_id"]:
            reason = f"the window its comparison chose, from {start / fps:.1f}s"
        else:
            reason = f"chosen by hand, from {start / fps:.1f}s"
        if start != wanted:
            reason += (f" (asked for {wanted / fps:.1f}s, moved so the "
                        f"{plan['lead']}-frame lead fits)")
        plan.update(start_frame=start, chain_start_frame=start - plan["lead"],
                    reason=reason)
        return plan

    _job_progress[job_id] = {"phase": "scan", "percent": None, "frames": None,
                             "total_frames": None, "elapsed_sec": None,
                             "eta_sec": None, "rate_fps": None}
    await _write_log(job_id, logf,
                      "[preview] scanning the source for the most telling minute "
                      f"({SCAN_WIDTH}x{SCAN_HEIGHT} greyscale, about 30 s per 12 "
                      "minutes of film)\n")
    started = time.monotonic()
    diffs = await _scan_motion(job_id, src, logf)
    chosen = _choose_window(diffs, plan["visible"], plan["lead"])
    await _write_log(job_id, logf,
                      f"[preview] scan took {time.monotonic() - started:.0f}s "
                      f"over {len(diffs) + 1} frames\n")
    if chosen is None:
        # The scan disagreed with the probe about the length. Trust neither for
        # the window and fall back to the middle, saying so.
        start = max(plan["lead"], (total_frames - plan["visible"]) // 2)
        plan.update(start_frame=start, chain_start_frame=start - plan["lead"],
                    reason="the motion scan came up short, so the middle of the "
                           "file was taken instead")
        return plan
    start, reason = chosen
    plan.update(start_frame=start, chain_start_frame=start - plan["lead"],
                reason=reason)
    return plan


def _keyframe_at_or_before(src: Path, t: float):
    """Presentation time of the last keyframe at or before `t`, or 0.0.

    A stream copy always starts at a keyframe, so a clip cut at an arbitrary
    time begins a little earlier than asked. Knowing exactly how much earlier
    is what keeps the frame arithmetic below honest - the alternative is to
    re-encode the clip, and that would cost the one property the measurement
    work relies on: identical pixels in, identical depth out.
    """
    if t <= 0:
        return 0.0
    for back in (20.0, 120.0, None):
        argv = [FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
                "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
                "-of", "csv=p=0"]
        lo = 0.0 if back is None else max(0.0, t - back)
        argv += ["-read_intervals", f"{lo:.3f}%{t + 0.5:.3f}", str(src)]
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        except Exception:
            return 0.0
        times = [float(v) for v in out.stdout.split() if v.replace(".", "", 1).isdigit()]
        before = [v for v in times if v <= t + 1e-6]
        if before:
            return max(before)
    return 0.0


async def _extract_preview_clip(job_id, src: Path, dest: Path, plan, logf):
    """Cut the planned window - lead, visible and tail - out of src into dest.

    Returns the number of frames that sit in front of the visible window in
    the extracted clip: the planned lead *plus* whatever the stream copy added
    by starting at the previous keyframe. That whole count is what the trim
    after the conversion cuts away again - the drift alone is not, and getting
    that wrong shifts the visible window by the length of the lead.
    """
    fps = plan["fps"]
    if plan["whole_source"]:
        await _write_log(job_id, logf,
                          f"[preview] source is shorter than {PREVIEW_SECONDS:g}s plus "
                          f"lead and tail, using it whole - note that the "
                          f"convergence estimator therefore starts cold\n")
        start_time, key_time, drift = 0.0, 0.0, 0
    else:
        start_time = plan["chain_start_frame"] / fps
        key_time = _keyframe_at_or_before(src, start_time)
        drift = int(round((start_time - key_time) * fps))
        await _write_log(
            job_id, logf,
            f"[preview] window: {plan['reason']}\n"
            f"[preview] visible frames {plan['start_frame']}-"
            f"{plan['start_frame'] + plan['visible'] - 1} "
            f"({plan['start_frame'] / fps:.1f}s-"
            f"{(plan['start_frame'] + plan['visible']) / fps:.1f}s), "
            f"{plan['lead']} frames of lead in front and {plan['tail']} behind\n"
            f"[preview] cutting from the keyframe at {key_time:.3f}s, which adds "
            f"{drift} frames on top of the lead, so {plan['lead'] + drift} frames "
            f"come off the front again afterwards\n")

    total = plan["total"] + drift
    # -nostats keeps ffmpeg's \r progress out of a log viewer that is already
    # busy throttling iw3's.
    base = [FFMPEG_BIN, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "warning", "-y"]
    # -ss *before* -i is an input seek: it lands on the keyframe at or before
    # the mark, which is exactly what a stream copy needs.
    window = (["-ss", f"{key_time:.3f}"] if key_time else []) + ["-i", str(src)]
    if not plan["whole_source"]:
        # A hair over the frame count, so rounding can never cut the last frame.
        window += ["-t", f"{(total + 1) / fps:.3f}"]
    # Only the first video and (if present) the first audio track: subtitle and
    # attachment streams carried over from a container like MKV have no place
    # in the mp4 the clip is written to.
    maps = ["-map", "0:v:0", "-map", "0:a:0?"]

    # The video is copied, never re-encoded: the clip has to look exactly like
    # the source or it cannot be used to judge the source. The audio is
    # re-encoded unconditionally - mp4 refuses plenty of the audio codecs that
    # arrive in mkv/wmv containers, and a minute of AAC costs no measurable
    # time. That keeps the fallback below for the one case that really needs
    # it: a video codec mp4 cannot hold at all.
    #
    # No -avoid_negative_ts make_zero. The AAC encoder starts 1024 samples
    # early, and make_zero pays for that by pushing the *video* back too -
    # 21 ms with 48 kHz sound. -t then cut the last frame down to a 12 ms stub,
    # iw3's frame-rate filter rounded the stub away, and the pipeline stopped
    # at stage 3 one depth map short (23.09., a clip with MP3 sound). Left to
    # itself, mp4 records the priming in an edit list, and picture and sound
    # both start at 0.
    rc = await _run_logged(job_id, base + window + maps +
                            ["-c:v", "copy", "-c:a", "aac", str(dest)], logf)
    if rc < 0:
        # Killed by a signal - that is a cancellation, not a bad source.
        raise RuntimeError(f"preview extraction terminated by signal {-rc}")
    if rc == 0 and dest.exists() and dest.stat().st_size > 0:
        return plan["lead"] + drift

    # Re-encoding a minute is a moment of CPU and always works. It does cost
    # the identical-pixels property above, so it is said out loud.
    await _write_log(job_id, logf,
                      "[preview] stream copy failed, re-encoding the clip instead - "
                      "the clip is no longer pixel-identical to the source\n")
    rc = await _run_logged(job_id, base + window + maps +
                            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                             "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest)], logf)
    if rc != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"preview clip extraction failed (ffmpeg exit code {rc})")
    return plan["lead"] + drift


# ---------------------------------------------------------------------------
# Will iw3 see every frame of the clip?
#
# The pipeline checks its stages against each other by frame count. Stage 1
# extracts with ffmpeg and -fps_mode passthrough; the VDA stage decodes
# through iw3, which always pushes frames through an fps filter at the
# source's own rate. A frame whose timing does not fit that filter is dropped,
# and the pipeline then stops at stage 3 - correctly, but only after DepthPro
# has spent a quarter of an hour. So the same decode runs here first, on the
# CPU, the way iw3's hook_frame runs it: one pull per frame pushed, then a
# drain at the end. A one-minute clip takes seconds.
_FRAME_CHECK = r'''
import sys
sys.path.insert(0, sys.argv[1])
import av
from nunif.utils.video import FixedFPSFilter, convert_known_fps, get_fps, safe_decode
container = av.open(sys.argv[2])
stream = container.streams.video[0]
if stream.codec.name != "hevc":
    stream.thread_type = "AUTO"
fps_filter = FixedFPSFilter(stream, fps=convert_known_fps(get_fps(stream)))
decoded = passed = 0
for packet in container.demux([stream]):
    for frame in safe_decode(packet):
        decoded += 1
        if fps_filter.update(frame) is not None:
            passed += 1
while fps_filter.update(None) is not None:
    passed += 1
container.close()
print(f"[preview] frame check: {decoded} decoded, {passed} through iw3's "
      f"frame-rate filter", flush=True)
sys.exit(0 if decoded == passed else 3)
'''


async def _check_clip_frames(job_id, clip: Path, logf):
    """Stop a pipeline preview before it starts if iw3 would lose frames."""
    rc = await _run_logged(job_id, [sys.executable, "-c", _FRAME_CHECK,
                                     str(NUNIF_DIR), str(clip)], logf,
                            shown=f"frame check on {clip.name}")
    if rc < 0:
        raise RuntimeError(f"frame check terminated by signal {-rc}")
    if rc == 3:
        raise RuntimeError("iw3 would see fewer frames of the preview clip than ffmpeg "
                           "does, so the pipeline would stop at stage 3 - stopped "
                           "before it started")
    if rc != 0:
        raise RuntimeError(f"frame check failed (exit code {rc})")


# ---------------------------------------------------------------------------
# The encoder this app uses for its own two re-encodes
#
# `ffmpeg -encoders` lists what was compiled in, which is not the question.
# Whether hevc_qsv works here depends on a render node being passed into the
# container and on the driver behind it, so the check is two seconds of test
# pattern actually encoded. The answer is cached for the life of the process
# and stated in the log, and the fallback is a software encode rather than a
# failed job.
# ---------------------------------------------------------------------------
_qsv_state = None  # None = not probed, True/False = the answer


def _qsv_works():
    global _qsv_state
    if _qsv_state is not None:
        return _qsv_state
    if not Path(QSV_FFMPEG).exists() or not Path(QSV_DEVICE).exists():
        _qsv_state = False
    else:
        argv = [QSV_FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-init_hw_device", f"qsv=hw:{QSV_DEVICE}", "-filter_hw_device", "hw",
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
                "-vf", "format=nv12,hwupload=extra_hw_frames=64",
                "-c:v", "hevc_qsv", "-global_quality", QSV_GLOBAL_QUALITY,
                "-f", "null", "-"]
        try:
            _qsv_state = subprocess.run(argv, capture_output=True, timeout=120).returncode == 0
        except Exception:
            _qsv_state = False
    print(f"[iw3-webui] hardware encode (hevc_qsv on {QSV_DEVICE}): "
          f"{'available' if _qsv_state else 'not available, using libx265'}", flush=True)
    return _qsv_state


def _encode_args():
    """(prefix args before -i, video codec args) for this app's re-encodes."""
    if _qsv_works():
        return ([QSV_FFMPEG, "-hide_banner", "-nostdin", "-nostats",
                 "-loglevel", "warning", "-y",
                 "-init_hw_device", f"qsv=hw:{QSV_DEVICE}", "-filter_hw_device", "hw"],
                ["-c:v", "hevc_qsv", "-global_quality", QSV_GLOBAL_QUALITY],
                "format=nv12,hwupload=extra_hw_frames=64")
    return ([FFMPEG_BIN, "-hide_banner", "-nostdin", "-nostats",
             "-loglevel", "warning", "-y"],
            ["-c:v", "libx265", "-crf", "20", "-preset", "medium"],
            "format=yuv420p")


def _deliver(path: Path):
    """Hand a finished file to whoever browses the share."""
    try:
        os.chown(path, OUTPUT_UID, OUTPUT_GID)
    except OSError:
        pass  # not root, or a filesystem without ownership - not worth failing on
    return path


def _label_font():
    for candidate in LABEL_FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def _label_chain(text, eye_width):
    """An opaque bar across the top of one eye with `text` centred in it.

    Drawn per eye rather than once across the pair, at the same coordinates in
    each, because a label that sits at a different place in the two eyes has a
    disparity of its own and swims in front of the picture. Opaque for the same
    reason: a translucent bar lets the two eyes' differing pixels through and
    flickers.
    """
    font = _label_font()
    if font is None:
        raise RuntimeError(
            "no TrueType font found for the comparison labels (looked in "
            + ", ".join(LABEL_FONT_CANDIDATES) + "). An A/B/A comparison "
            "without labels cannot be read, so this is refused rather than "
            "shipped unlabelled.")
    safe = text.replace("\\", "").replace("'", "").replace(":", " -")
    return (f"drawbox=x=0:y=0:w={eye_width}:h={LABEL_BAR_HEIGHT}:color=black:t=fill,"
            f"drawtext=fontfile={font}:text='{safe}':fontsize=40:fontcolor=white:"
            f"x=(w-text_w)/2:y={(LABEL_BAR_HEIGHT - 48) // 2}")


def _stream_info(path: Path):
    """(width, height, fps, has_audio, frames) of a file this app produced."""
    argv = [FFPROBE_BIN, "-v", "error", "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,nb_frames",
            "-of", "json", str(path)]
    try:
        data = json.loads(subprocess.run(argv, capture_output=True, text=True,
                                          timeout=120).stdout)
    except Exception:
        return None
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        return None
    num, _, den = (video.get("r_frame_rate") or "0/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": fps,
        "frames": int(video.get("nb_frames") or 0),
        "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
    }


async def _trim_preview(job_id, src: Path, dest: Path, head, visible, fps, logf):
    """Cut the lead and tail frames off a converted preview.

    The lead went through every stage so the convergence estimator would be
    warm by the time the visible window started; it has done its work and has
    no business in the file anybody watches.
    """
    pre, vcodec, upload = _encode_args()
    info = _stream_info(src) or {"has_audio": False}
    parts = [f"[0:v]trim=start_frame={head}:end_frame={head + visible},"
             f"setpts=PTS-STARTPTS,{upload}[v]"]
    maps, acodec = ["-map", "[v]"], []
    if info["has_audio"]:
        parts.append(f"[0:a]atrim=start={head / fps:.6f}:duration={visible / fps:.6f},"
                      f"asetpts=PTS-STARTPTS[a]")
        maps += ["-map", "[a]"]
        acodec = ["-c:a", "aac"]
    argv = pre + ["-i", str(src), "-filter_complex", ";".join(parts)] + \
        maps + vcodec + acodec + [str(dest)]
    rc = await _run_logged(job_id, argv, logf)
    if rc != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"trimming the preview lead failed (ffmpeg exit code {rc})")
    return _deliver(dest)


# ---------------------------------------------------------------------------
# Comparing two previews in one file
#
# 🔴 The trap, and it is the whole reason this is not four lines of ffmpeg:
# the previews are already side-by-side stereo. 3840x1080 is not a picture, it
# is left eye | right eye. Putting two of them next to each other produces
#
#     A-left  A-right  B-left  B-right
#
# and a headset splits that down the middle, so it shows A-right to the left
# eye and B-left to the right eye. Two different recipes, one per eye, at a
# disparity that means nothing. It does not look like a comparison; it looks
# like a fault.
#
# So both arrangements below are built *per eye*: each source is cut into its
# two halves first, the halves are labelled and combined, and only then are
# the two finished eyes put back together as one side-by-side frame.
#
#   A/B/A  the same window three times, A then B then A, geometry untouched.
#          This is how every recipe question of the past week was actually
#          decided - the second A is what makes a small difference visible,
#          because you see the change twice and in both directions.
#
#   stacked  A above B within each eye: twice the eye's height - 1920x2160
#          per eye (3840x2160 in all) from a 1080p source, 2560x2880 per eye
#          (5120x2880) from a 1440p one. The page states the size from the
#          actual previews rather than from this example.
#          Vertical rather than horizontal on purpose. Side by side within an
#          eye would give each recipe half a picture, in a 32:9 strip,
#          compared out of the corner of the eye. Stacked, both keep the full
#          picture and both sit centred.
#          ⚠️ Each eye is then twice as tall as usual (16:18). Players
#          that split a full-SBS frame and letterbox each half handle that;
#          players that assume 16:9 per eye will stretch it. A/B/A is the
#          arrangement with nothing to assume, which is why it is the default.
# ---------------------------------------------------------------------------
COMPARE_LAYOUTS = {
    "aba": "A / B / A, one after another (default)",
    "stacked": "A above B, within each eye",
}


def _compare_dir(job_id) -> Path:
    return OUTPUT_ROOT / "_compare" / job_id


async def _run_compare(job_id, row, logf):
    """Build one comparison file out of two finished previews."""
    params = json.loads(row["params_json"])
    layout = params.get("layout", "aba")
    if layout not in COMPARE_LAYOUTS:
        raise RuntimeError(f"unknown comparison layout: {layout}")

    with _db() as conn:
        a = conn.execute("SELECT * FROM jobs WHERE id=?", (params["a"],)).fetchone()
        b = conn.execute("SELECT * FROM jobs WHERE id=?", (params["b"],)).fetchone()
    for side, r in (("A", a), ("B", b)):
        if r is None or not r["output_path"] or not Path(r["output_path"]).is_file():
            raise RuntimeError(f"side {side} has no finished preview file any more")

    src_a, src_b = Path(a["output_path"]), Path(b["output_path"])
    ia, ib = _stream_info(src_a), _stream_info(src_b)
    if not ia or not ib:
        raise RuntimeError("could not probe one of the two previews")
    # Refused rather than fixed: two clips of different length or geometry are
    # two different questions, and silently scaling one to match the other
    # would make the comparison lie about exactly the thing it is for.
    if (ia["width"], ia["height"]) != (ib["width"], ib["height"]):
        raise RuntimeError(
            f"the two previews have different frame sizes ({ia['width']}x{ia['height']} "
            f"and {ib['width']}x{ib['height']}); a comparison has to be of the same window")
    if ia["width"] % 2 or (ia["width"] // 2) % 2:
        raise RuntimeError(f"frame width {ia['width']} does not split into two even eyes")
    if abs(ia["fps"] - ib["fps"]) > 0.01:
        raise RuntimeError(f"the two previews run at different frame rates "
                            f"({ia['fps']:.3f} and {ib['fps']:.3f})")

    label_a = params.get("label_a") or "A"
    label_b = params.get("label_b") or "B"
    eye_w, eye_h = ia["width"] // 2, ia["height"]
    bar_a, bar_b = _label_chain(label_a, eye_w), _label_chain(label_b, eye_w)

    parts = []
    for idx, bar in ((0, bar_a), (1, bar_b)):
        name = "A" if idx == 0 else "B"
        parts.append(f"[{idx}:v]split=2[{name}l0][{name}r0]")
        parts.append(f"[{name}l0]crop={eye_w}:{eye_h}:0:0,{bar}[{name}L]")
        parts.append(f"[{name}r0]crop={eye_w}:{eye_h}:{eye_w}:0,{bar}[{name}R]")

    both_audio = ia["has_audio"] and ib["has_audio"]
    maps, acodec = [], []
    if layout == "stacked":
        parts += ["[AL][BL]vstack=inputs=2[L]",
                  "[AR][BR]vstack=inputs=2[R]",
                  "[L][R]hstack=inputs=2[vraw]"]
        out_size = (ia["width"], ia["height"] * 2)
        if ia["has_audio"]:
            maps = ["-map", "0:a:0"]
            acodec = ["-c:a", "aac"]
    else:
        parts += ["[AL][AR]hstack=inputs=2[Afull]",
                  "[BL][BR]hstack=inputs=2[Bfull]",
                  "[Afull]split=2[A1][A2]"]
        if both_audio:
            parts.append("[0:a]asplit=2[aa1][aa2]")
            parts.append("[A1][aa1][Bfull][1:a][A2][aa2]concat=n=3:v=1:a=1[vraw][a]")
            maps = ["-map", "[a]"]
            acodec = ["-c:a", "aac"]
        else:
            parts.append("[A1][Bfull][A2]concat=n=3:v=1:a=0[vraw]")
        out_size = (ia["width"], ia["height"])

    pre, vcodec, upload = _encode_args()
    parts.append(f"[vraw]{upload}[v]")

    out_dir = _compare_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"COMPARE_{_slug(label_a)}_vs_{_slug(label_b)}_{layout}_LRF.mp4"

    await _write_log(job_id, logf,
                      f"[compare] {COMPARE_LAYOUTS[layout]}\n"
                      f"[compare] A = {label_a}\n"
                      f"[compare] B = {label_b}\n"
                      f"[compare] each eye {eye_w}x{eye_h}, output "
                      f"{out_size[0]}x{out_size[1]}\n")
    argv = pre + ["-i", str(src_a), "-i", str(src_b),
                  "-filter_complex", ";".join(parts), "-map", "[v]"] + \
        maps + vcodec + acodec + [str(dest)]

    started = time.monotonic()
    rc = await _run_logged(job_id, argv, logf)
    if rc != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"building the comparison failed (ffmpeg exit code {rc})")
    took = time.monotonic() - started

    produced = _stream_info(dest) or {}
    if (produced.get("width"), produced.get("height")) != out_size:
        raise RuntimeError(f"comparison came out {produced.get('width')}x"
                            f"{produced.get('height')}, expected "
                            f"{out_size[0]}x{out_size[1]}")
    _deliver(dest)
    await _write_log(job_id, logf,
                      f"[compare] wrote {dest} in {took:.0f}s\n")
    with _db() as conn:
        conn.execute("UPDATE jobs SET output_path=? WHERE id=?", (str(dest), job_id))
        conn.commit()


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:40] or "x"


# ---------------------------------------------------------------------------
# A comparison made from one file
#
# The first compare feature took two previews that already existed. Asked for
# with a file selected, it compared whatever two previews there were - of a
# different film - because it never looked at the selection. What was wanted
# is the other way round: pick a file, pick two levels, get the comparison.
#
# So one request queues three things: preview A, preview B and a comparison
# that waits for both. The window is worked out ONCE, by the comparison, and
# handed to both previews as an explicit start: two independent scans of the
# same file would almost certainly agree, but "almost certainly" is exactly
# what a comparison of two recipes cannot rest on. A finished preview of the
# same file, level, switches, format and window is reused instead of rerun.
#
# The comparison stays queued while it waits (it is not on any GPU), shows
# what it is waiting for, and is canceled - with the reason - the moment one
# of its previews fails or is canceled.
# ---------------------------------------------------------------------------
# The comparison cuts each frame into a left and a right half, so it only
# makes sense for formats that are side by side.
COMPARE_STEREO_FORMATS = ("full_sbs", "half_sbs", "cross_eyed")
COMPARE_BUILD_SEC = 90.0   # measured: 76 s for a stacked 1440p pair


def _spec_label(spec, side):
    level = QUALITY_BY_ID.get(spec["quality"], {})
    switches = (["+flow"] if spec.get("flow") else []) + (["+4K"] if spec.get("upscale") else [])
    return f"{side} - {level.get('label') or spec['quality']}" + \
        (" " + " ".join(switches) if switches else "")


def _clamp_start(info, plan, seconds):
    """First visible frame for a start time, moved so lead and tail fit."""
    fps = info["fps"]
    total_frames = int(round(info["duration_sec"] * fps))
    wanted = int(round(float(seconds) * fps))
    return max(plan["lead"], min(wanted, total_frames - plan["visible"] - plan["tail"]))


_VISIBLE_FROM_LOG = re.compile(r"^\[preview\] visible frames (\d+)-", re.M)


def _preview_start_from_log(job_id):
    """The first visible frame a finished auto-window preview actually used."""
    try:
        text = (LOG_DIR / f"{job_id}.log").read_text(errors="replace")
    except OSError:
        return None
    m = _VISIBLE_FROM_LOG.search(text)
    return int(m.group(1)) if m else None


def _find_reusable_preview(conn, input_path, stereo_format, spec, window, info, plan):
    """A preview that already is (or is about to be) the one this side needs."""
    level = QUALITY_BY_ID[spec["quality"]]
    rows = conn.execute(
        "SELECT * FROM jobs WHERE mode='preview' AND input_path=? AND quality=? "
        "AND opt_flow=? AND opt_upscale=? AND stereo_format=? "
        "AND status IN ('done','queued','running') "
        "ORDER BY status='done' DESC, created_at DESC",
        (input_path, spec["quality"], int(spec["flow"]), int(spec["upscale"]),
         stereo_format)).fetchall()
    for r in rows:
        # A level's parameters are stored with the job; a job from before a
        # correction to the level is a different recipe under the same name.
        if level.get("params") and json.loads(r["params_json"]) != level["params"]:
            continue
        if r["status"] == "done" and not (r["output_path"] and Path(r["output_path"]).is_file()):
            continue
        if window["whole"]:
            return r
        if r["preview_start_sec"] is not None:
            if _clamp_start(info, plan, r["preview_start_sec"]) == window["start_frame"]:
                return r
        elif r["status"] == "done" and _preview_start_from_log(r["id"]) == window["start_frame"]:
            return r
    return None


async def _plan_file_compare(job_id, row, params, logf):
    """Choose the window once, then find or queue both previews."""
    src = _safe_input_path(row["input_path"])
    info = _probe_video(src)
    if not info or not info.get("duration_sec") or not info.get("fps"):
        raise RuntimeError("could not probe the source for duration and frame rate")
    fps = info["fps"]
    plan = _preview_frames(info)
    request = params.get("window_request")
    if plan["whole_source"]:
        window = {"whole": True, "start_frame": 0, "start_sec": None,
                  "reason": "source too short for a window, both previews take it whole"}
    elif request is not None:
        start = _clamp_start(info, plan, request)
        reason = f"chosen by hand, from {start / fps:.1f}s"
        if start != int(round(float(request) * fps)):
            reason += " (moved so the lead and tail fit)"
        window = {"whole": False, "start_frame": start, "start_sec": start / fps,
                  "reason": reason}
    else:
        _job_progress[job_id] = {"phase": "scan", "percent": None, "frames": None,
                                 "total_frames": None, "elapsed_sec": None,
                                 "eta_sec": None, "rate_fps": None}
        await _write_log(job_id, logf, "[compare] choosing one window for both previews\n")
        started = time.monotonic()
        try:
            diffs = await _scan_motion(job_id, src, logf)
        finally:
            _job_progress.pop(job_id, None)
        if job_id in _cancel_requested:
            raise RuntimeError("canceled during the window scan")
        await _write_log(job_id, logf, f"[compare] scan took {time.monotonic() - started:.0f}s "
                                       f"over {len(diffs) + 1} frames\n")
        chosen = _choose_window(diffs, plan["visible"], plan["lead"])
        if chosen is None:
            total_frames = int(round(info["duration_sec"] * fps))
            chosen = (max(plan["lead"], (total_frames - plan["visible"]) // 2),
                      "the motion scan came up short, so the middle of the file was taken")
        window = {"whole": False, "start_frame": chosen[0], "start_sec": chosen[0] / fps,
                  "reason": chosen[1]}

    ids, reused = {}, {}
    with _db() as conn:
        current = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if current is None or current["status"] != "queued":
            return False     # canceled or removed while the scan ran
        for side in ("a", "b"):
            spec = params["spec_" + side]
            hit = (_find_reusable_preview(conn, row["input_path"], row["stereo_format"],
                                          spec, window, info, plan)
                   if params.get("reuse", True) else None)
            if hit is not None:
                ids[side], reused[side] = hit["id"], hit["status"]
                continue
            level = QUALITY_BY_ID[spec["quality"]]
            ids[side] = _insert_job(
                conn, "preview", row["input_path"], False, row["stereo_format"],
                dict(level["params"]) if level.get("params") else {},
                spec["quality"], spec["flow"], spec["upscale"],
                None if window["whole"] else window["start_sec"], parent_id=job_id)
            reused[side] = None
        params.update(a=ids["a"], b=ids["b"], window=window, reused=reused)
        conn.execute("UPDATE jobs SET params_json=? WHERE id=?", (json.dumps(params), job_id))
        conn.commit()

    visible = plan["visible"]
    where = ("the whole source" if window["whole"] else
             f"frames {window['start_frame']}-{window['start_frame'] + visible - 1} "
             f"({window['start_frame'] / fps:.1f}s-{(window['start_frame'] + visible) / fps:.1f}s)")
    lines = [f"[compare] window: {window['reason']}", f"[compare] both previews cover {where}"]
    for side in ("a", "b"):
        how = (f"reusing {reused[side]} preview {ids[side]}" if reused[side]
               else f"queued preview {ids[side]}")
        lines.append(f"[compare] {params['label_' + side]}: {how}")
    await _write_log(job_id, logf, "\n".join(lines) + "\n")
    return True


def _cancel_compare(job_id, reason):
    with _db() as conn:
        conn.execute("UPDATE jobs SET status='canceled', finished_at=?, error=? "
                      "WHERE id=? AND status='queued'", (_now(), reason, job_id))
        conn.commit()
    with open(LOG_DIR / f"{job_id}.log", "a") as logf:
        logf.write(f"[compare] canceled: {reason}\n")


async def _advance_file_compare(row):
    """'build' once both previews are done, else 'wait' (or 'gone')."""
    job_id = row["id"]
    params = json.loads(row["params_json"])
    if not params.get("a"):
        with open(LOG_DIR / f"{job_id}.log", "a") as logf:
            try:
                planned = await _plan_file_compare(job_id, row, params, logf)
            except Exception as e:
                if job_id in _cancel_requested:
                    _cancel_requested.discard(job_id)
                    return "gone"   # the cancel endpoint already filed it
                await _write_log(job_id, logf, f"\n[job failed, {e}]\n")
                with _db() as conn:
                    conn.execute("UPDATE jobs SET status='failed', finished_at=?, error=? "
                                  "WHERE id=? AND status='queued'", (_now(), str(e), job_id))
                    conn.commit()
                await _publish_log(job_id, "__EOF__")
                return "gone"
        return "wait" if planned else "gone"
    with _db() as conn:
        sides = {s: conn.execute("SELECT id, status, error FROM jobs WHERE id=?",
                                 (params[s],)).fetchone() for s in ("a", "b")}
    for s, r in sides.items():
        label = params["label_" + s].split(" - ", 1)[-1]
        name = f"preview {s.upper()} ({label})"
        if r is None:
            _cancel_compare(job_id, f"{name} was removed")
            await _publish_log(job_id, "__EOF__")
            return "gone"
        if r["status"] in ("failed", "canceled"):
            why = f"{name} {r['status']}" + (f": {r['error']}" if r["error"] else "")
            _cancel_compare(job_id, why)
            await _publish_log(job_id, "__EOF__")
            return "gone"
    if all(r["status"] == "done" for r in sides.values()):
        return "build"
    return "wait"


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
    preview_plan = None
    preview_head = 0
    with open(log_path, "a") as logf:
        if row["mode"] == "compare":
            try:
                await _run_compare(job_id, row, logf)
            except Exception as e:
                if job_id in _cancel_requested:
                    _cancel_requested.discard(job_id)
                    shutil.rmtree(_compare_dir(job_id), ignore_errors=True)
                    await _finish_canceled(job_id, logf)
                    return
                await _write_log(job_id, logf, f"\n[job failed, {e}]\n")
                with _db() as conn:
                    conn.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
                                  (_now(), str(e), job_id))
                    conn.commit()
                await _publish_log(job_id, "__EOF__")
                return
            with _db() as conn:
                conn.execute("UPDATE jobs SET status='done', finished_at=? WHERE id=?",
                              (_now(), job_id))
                conn.commit()
            await _publish_log(job_id, "\n[job done, exit code 0]\n")
            await _publish_log(job_id, "__EOF__")
            return

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
                preview_plan = await _preview_plan(job_id, src, row, logf)
                preview_head = await _extract_preview_clip(
                    job_id, src, clip_path, preview_plan, logf)
                if is_chain:
                    await _check_clip_frames(job_id, clip_path, logf)
            except Exception as e:
                if job_id in _cancel_requested:
                    # Cancelled during the scan or the cut: nothing of it is
                    # worth keeping.
                    _cancel_requested.discard(job_id)
                    shutil.rmtree(out_dir, ignore_errors=True)
                    await _finish_canceled(job_id, logf)
                    return
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

        if job_id in _cancel_requested:
            # Cancelled in the gap between cutting the clip and starting the
            # conversion, when there was no process to signal.
            _cancel_requested.discard(job_id)
            if row["mode"] == "preview":
                shutil.rmtree(_preview_dir(job_id), ignore_errors=True)
            await _finish_canceled(job_id, logf)
            return
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
                        frames_line = None if stage else _CHAIN_FRAMES.match(text.rstrip("\n"))
                        if stage:
                            # The frame count is printed once, after stage 1,
                            # and every later stage's model needs it.
                            previous = _job_progress.get(job_id) or {}
                            if previous.get("chain_frames"):
                                stage["chain_frames"] = previous["chain_frames"]
                            _job_progress[job_id] = stage
                        elif frames_line and _job_progress.get(job_id):
                            _job_progress[job_id]["chain_frames"] = int(frames_line.group(1))
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

    # A job that exits 0 is done even if Cancel was pressed a moment too late;
    # anything else after a Cancel is a cancellation, not a failure.
    canceled = returncode != 0 and job_id in _cancel_requested
    _cancel_requested.discard(job_id)
    if canceled:
        # Unlike a failure, a cancellation has nothing to inspect: its scratch
        # (frames, depth maps - 11 GB after half an hour of a full film) goes.
        with open(log_path, "a") as logf:
            if is_chain:
                await _write_log(job_id, logf, f"[pipeline] canceled, removing scratch "
                                               f"{_chain_work_dir(job_id)}\n")
                await asyncio.to_thread(shutil.rmtree, _chain_work_dir(job_id), True)
            if row["mode"] == "preview":
                await asyncio.to_thread(shutil.rmtree, _preview_dir(job_id), True)
            await _finish_canceled(job_id, logf)
        return
    status = "done" if returncode == 0 else "failed"
    if is_chain:
        work = _chain_work_dir(job_id)
        if status == "done":
            # Tens of thousands of extracted frames and as many depth maps -
            # keeping them would fill the output volume within a few jobs.
            # Off the event loop: on the array this takes over half a minute.
            await asyncio.to_thread(shutil.rmtree, work, True)
        elif work.exists():
            await _publish_log(job_id, f"[pipeline] scratch kept for inspection: {work}\n")
    output_path = None
    if status == "done" and row["mode"] == "preview" and preview_plan:
        # Everything produced so far still carries the lead and tail frames.
        # Cutting them here rather than asking the pipeline script to do it
        # keeps that contract at seven points and puts the whole window
        # arithmetic in one place.
        produced = sorted(p for p in _preview_dir(job_id).glob("*.mp4")
                          if p != clip_path)
        if not produced:
            status = "failed"
            await _publish_log(job_id, "[preview] the conversion produced no file\n")
        elif preview_head or preview_plan["tail"]:
            # Delivery files end in _LRF.mp4. The pipeline's own output already
            # does; a single iw3 pass names its file _LRF_Full_SBS.mp4 and does
            # not. So the suffix is added only when it is missing - and the
            # trim goes to a temporary file first either way, because the name
            # that comes out can equal the name that went in, and ffmpeg
            # reading and writing one file destroys it.
            stem = produced[0].stem
            if not stem.upper().endswith("_LRF"):
                stem += "_LRF"
            final = _preview_dir(job_id) / f"{stem}.mp4"
            tmp = _preview_dir(job_id) / f".{stem}.trimming.mp4"
            try:
                with open(log_path, "a") as logf:
                    await _trim_preview(job_id, produced[0], tmp, preview_head,
                                        preview_plan["visible"], preview_plan["fps"], logf)
                produced[0].unlink(missing_ok=True)
                tmp.replace(final)
                output_path = _deliver(final)
            except Exception as e:
                status = "failed"
                tmp.unlink(missing_ok=True)
                await _publish_log(job_id, f"[preview] {e}\n")
        else:
            output_path = _deliver(produced[0])

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
    # The conversion's own exit code, and separately whatever happened after
    # it: a job whose conversion succeeded and whose trim failed is failed, and
    # saying "exit code 0" about it would be the wrong half of the story.
    error = None
    if returncode != 0:
        error = f"exit code {returncode}"
    elif status != "done":
        error = "the conversion finished but the preview could not be completed"
    with _db() as conn:
        conn.execute("UPDATE jobs SET status=?, finished_at=?, error=?, output_path=? "
                      "WHERE id=?",
                      (status, _now(), error,
                       str(output_path) if output_path else None, job_id))
        conn.commit()
    await _publish_log(job_id, f"\n[job {status}{'' if error is None else ', ' + error}]\n")
    await _publish_log(job_id, "__EOF__")


async def _worker_loop():
    while True:
        with _db() as conn:
            # One sort key, set by create_job and by the reorder endpoints.
            # created_at only breaks ties between rows that somehow share a
            # position; it no longer decides anything on its own.
            row = conn.execute(
                "SELECT id FROM jobs WHERE status='queued' AND mode<>'compare' "
                "ORDER BY position, created_at LIMIT 1"
            ).fetchone()
        if row is None:
            await asyncio.sleep(1.0)
            continue
        # One job at a time - the GPU cannot be shared across concurrent conversions.
        try:
            await _run_job(row["id"])
        except Exception as e:
            canceled = row["id"] in _cancel_requested
            _cancel_requested.discard(row["id"])
            with _db() as conn:
                conn.execute("UPDATE jobs SET status=?, finished_at=?, error=? WHERE id=?",
                              ("canceled" if canceled else "failed", _now(),
                               None if canceled else str(e), row["id"]))
                conn.commit()


async def _finish_canceled(job_id, logf):
    """File a job as canceled - no error, because nothing went wrong."""
    await _write_log(job_id, logf, "\n[job canceled]\n")
    with _db() as conn:
        conn.execute("UPDATE jobs SET status='canceled', finished_at=?, error=NULL, "
                      "output_path=NULL WHERE id=?", (_now(), job_id))
        conn.commit()
    await _publish_log(job_id, "__EOF__")


async def _compare_loop():
    """A second worker, for comparison files only.

    Deliberately not the same queue. A comparison is two or three minutes of
    fixed-function video encoder - a different unit of the same chip from the
    one a conversion occupies, so it genuinely does run alongside one. Behind
    the conversion queue it would instead wait out four hours, and a tool for
    looking at two previews is worth nothing the day after you wanted to look.
    Still one at a time, so two comparisons cannot collide.

    A comparison made from a file sits in the same list while it waits for its
    previews; it is passed over until both are done, so a waiting comparison
    never holds up one that is ready.
    """
    while True:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status='queued' AND mode='compare' "
                "ORDER BY created_at").fetchall()
        ready = None
        for row in rows:
            if json.loads(row["params_json"]).get("source") != "file":
                ready = row
                break
            try:
                state = await _advance_file_compare(row)
            except Exception as e:
                print(f"[iw3-webui] compare {row['id']}: {e}", flush=True)
                continue
            if state == "build":
                ready = row
                break
        if ready is None:
            await asyncio.sleep(2.0)
            continue
        try:
            await _run_job(ready["id"])
        except Exception as e:
            with _db() as conn:
                conn.execute("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
                              (_now(), str(e), ready["id"]))
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
    asyncio.create_task(_compare_loop())
    # Built at start so the first search does not wait for it.
    _ensure_search_index()


class JobCreate(BaseModel):
    mode: str  # "convert" | "preview"
    input_path: str  # relative to /input
    recursive: bool = False
    stereo_format: str = "full_sbs"
    quality: str = "custom"
    flow: bool = False
    upscale: bool = False
    # Null means "use the rule". A number is a second in the source, and it is
    # the first *visible* second - the lead is put in front of it, not taken
    # out of it, so two previews asked for at the same second cover the same
    # window whatever their level.
    preview_start_sec: float | None = None
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
    out = {k: level.get(k) for k in ("id", "label", "chain", "uses", "good_for",
                                     "pros", "cons", "notes")}
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
        "options": [{k: op.get(k) for k in ("id", "label", "chain_only",
                                           "uses", "pros", "cons", "notes")}
                    for op in QUALITY_OPTIONS],
        "default": DEFAULT_QUALITY if available else "fast",
        "chain_available": available,
        "chain_script": str(CHAIN_SCRIPT),
        "measured_on_frames": MEASURED_ON_FRAMES,
        "preview": {
            "seconds": PREVIEW_SECONDS,
            "lead_frames": PREVIEW_LEAD_FRAMES,
            "tail_frames": PREVIEW_TAIL_FRAMES,
            "window_rule": WINDOW_RULE,
        },
        "compare_layouts": [{"id": k, "label": v} for k, v in COMPARE_LAYOUTS.items()],
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
    window = None
    auto_scan = True
    if info and info.get("duration_sec") and info.get("fps"):
        source_frames = info["duration_sec"] * info["fps"]
        if mode == "preview":
            window = _preview_frames(info)
            frames = window["total"]
            auto_scan = not window["whole_source"]
        else:
            frames = source_frames
        source = "selected file"
    else:
        # The reference the levels were measured on, so a first-time visitor
        # still sees the shape of the ladder before picking a file.
        info = {"duration_sec": MEASURED_ON_FRAMES / 29.97, "fps": 29.97,
                "width": 1920, "height": 1080}
        source_frames = MEASURED_ON_FRAMES
        if mode == "preview":
            window = _preview_frames(info)
            frames = window["total"]
        else:
            frames = source_frames
        source = "reference measurement (12 min, 1080p)"
    px = _px_of(info)
    extra = _preview_extra_seconds(info, px, auto_scan) if mode == "preview" else 0.0

    levels = {}
    for lv in QUALITY_LEVELS:
        seconds, extrapolated = _level_seconds(lv["id"], frames, px,
                                               flow=flow, upscale=upscale)
        levels[lv["id"]] = {
            "seconds": round(seconds + extra) if seconds else None,
            "extrapolated": extrapolated,
        }
    # What each switch adds, priced on Standard: the stage itself, and for the
    # upscale also the later stages working on four times the pixels.
    options = {}
    base, _ = _level_seconds(DEFAULT_QUALITY, frames, px)
    for op in QUALITY_OPTIONS:
        with_op, _ = _level_seconds(DEFAULT_QUALITY, frames, px,
                                    flow=op["id"] == "flow", upscale=op["id"] == "upscale")
        options[op["id"]] = {"seconds": round(with_op - base), "extrapolated": True}
    return {
        "frames": round(frames),
        "source_frames": round(source_frames),
        # The share of the film a preview actually costs, computed the same way
        # for whatever is selected: 2,028 of 21,606 frames on the reference,
        # which is the 9.4% the window work quoted.
        "preview_share": round(100.0 * frames / source_frames, 1) if window else None,
        "preview_window": window,
        "resolution_bucket": _bucket(info),
        "width": info.get("width"),
        "height": info.get("height"),
        "pixels_vs_1080p": round(px, 3),
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
        "preview_seconds": PREVIEW_SECONDS,
        "preview_lead_frames": PREVIEW_LEAD_FRAMES,
        "preview_tail_frames": PREVIEW_TAIL_FRAMES,
        "qsv_ffmpeg": QSV_FFMPEG if Path(QSV_FFMPEG).exists() else None,
        "qsv_device": QSV_DEVICE if Path(QSV_DEVICE).exists() else None,
        "hardware_encode": _qsv_works(),
        "label_font": _label_font(),
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


# ---------------------------------------------------------------------------
# Searching the picker
#
# /input is a FUSE union over the array, a few hundred thousand directory
# entries deep, and a plain walk of it runs well past ten minutes - most of it
# spent listing frame dumps (the pipeline's scratch, the research runs), each
# 15-30k images. So the search runs against an in-memory index that a
# background thread builds, never on the event loop and never per keystroke:
#
#   * directories the pipeline owns (_chain) and hidden/system ones are pruned;
#   * a directory whose first SEARCH_BAIL_AFTER entries are all images and
#     nothing else is abandoned mid-listing - it is a frame dump, and reading
#     the other 29,700 names would only confirm it;
#   * a directory holding `*.ok` stage markers is a pipeline work area (the
#     pipeline's own convention): its videos are indexed, its subdirectories -
#     frame and depth-map stores - are not entered. On a cold cache these are
#     what turned a walk of minutes into one of tens of minutes;
#   * only folders and video files are kept, because nothing else can be
#     selected for conversion anyway.
#
# The index is rebuilt when it is older than SEARCH_TTL_SEC and a search
# arrives, or on request. While a rebuild runs, searches keep answering from
# the previous index; during the very first build they answer from what has
# been read so far and say so.
# ---------------------------------------------------------------------------
SEARCH_VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".m4v", ".flv",
                    ".ts", ".m2ts", ".mts", ".mpg", ".mpeg", ".3gp", ".ogv", ".vob",
                    ".divx", ".asf", ".rmvb", ".f4v"}
SEARCH_FRAME_EXT = {".jpg", ".jpeg", ".png", ".webp", ".exr", ".tif", ".tiff", ".npy",
                    ".npz", ".bmp", ".pgm", ".ppm"}
SEARCH_SKIP_DIRS = {"_chain", "$RECYCLE.BIN", "@eaDir", "node_modules", "__pycache__"}
SEARCH_BAIL_AFTER = 300
SEARCH_TTL_SEC = float(os.environ.get("SEARCH_TTL_SEC", "1800"))
SEARCH_LIMIT = 200

_search = {"entries": None, "building": None, "built_at": None, "build_sec": None,
           "dirs": 0, "bailed": 0, "work_areas": 0, "started": None, "error": None}
_search_lock = threading.Lock()


def _build_search_index():
    started = time.monotonic()
    building: list = []
    with _search_lock:
        _search.update(building=building, started=time.time(), error=None)
    dirs = bailed = work_areas = 0
    stack = [INPUT_ROOT]
    try:
        while stack:
            d = stack.pop()
            n = frames = 0
            videos_or_dirs = work_area = False
            sub, local = [], []
            try:
                with os.scandir(d) as it:
                    for e in it:
                        n += 1
                        try:
                            is_dir = e.is_dir(follow_symlinks=False)
                        except OSError:
                            continue
                        if is_dir:
                            videos_or_dirs = True
                            if e.name in SEARCH_SKIP_DIRS or e.name.startswith("."):
                                continue
                            sub.append(Path(e.path))
                            local.append((e.name.lower(), e.name, e.path, "dir"))
                        else:
                            ext = os.path.splitext(e.name)[1].lower()
                            if ext in SEARCH_VIDEO_EXT:
                                videos_or_dirs = True
                                local.append((e.name.lower(), e.name, e.path, "file"))
                            elif ext in SEARCH_FRAME_EXT:
                                frames += 1
                            elif ext == ".ok":
                                work_area = True
                        if (n == SEARCH_BAIL_AFTER and not videos_or_dirs
                                and frames >= 0.95 * n):
                            bailed += 1
                            break
            except OSError:
                continue
            dirs += 1
            root = str(INPUT_ROOT) + "/"
            building.extend((low, name, p[len(root):], kind) for low, name, p, kind in local)
            if work_area:
                work_areas += 1
            else:
                stack.extend(sub)
    except Exception as e:  # never let the thread die silently
        with _search_lock:
            _search.update(building=None, error=str(e))
        print(f"[iw3-webui] search index failed: {e}", flush=True)
        return
    building.sort(key=lambda t: t[2].lower())
    took = time.monotonic() - started
    with _search_lock:
        _search.update(entries=building, building=None, built_at=time.time(),
                       build_sec=round(took, 1), dirs=dirs, bailed=bailed,
                       work_areas=work_areas)
    print(f"[iw3-webui] search index: {len(building)} entries from {dirs} folders "
          f"in {took:.1f}s ({bailed} image-only folders skipped, {work_areas} "
          f"pipeline work areas not entered)", flush=True)


def _ensure_search_index(force=False):
    """Start a (re)build in the background if one is due. Never blocks."""
    with _search_lock:
        if _search["building"] is not None:
            return False
        fresh = (_search["built_at"] is not None
                 and time.time() - _search["built_at"] < SEARCH_TTL_SEC)
        if fresh and not force:
            return False
        _search["building"] = []   # claimed; the thread replaces it
    threading.Thread(target=_build_search_index, name="search-index", daemon=True).start()
    return True


def _search_status():
    with _search_lock:
        s = dict(_search)
    return {
        "entries": len(s["entries"]) if s["entries"] is not None else None,
        "built_at": datetime.fromtimestamp(s["built_at"], timezone.utc).isoformat()
                    if s["built_at"] else None,
        "build_sec": s["build_sec"],
        "folders_read": s["dirs"],
        "image_folders_skipped": s["bailed"],
        "work_areas_not_entered": s["work_areas"],
        "building": s["building"] is not None,
        "building_entries": len(s["building"]) if s["building"] is not None else None,
        "error": s["error"],
        "ttl_sec": SEARCH_TTL_SEC,
    }


@app.get("/api/search")
def search(q: str = "", limit: int = SEARCH_LIMIT):
    """File and folder names under /input; every word has to occur in the name."""
    _ensure_search_index()
    words = [w for w in q.lower().split() if w]
    with _search_lock:
        entries = _search["entries"]
        partial = entries is None
        if partial:
            entries = list(_search["building"] or [])
    limit = max(1, min(int(limit), 1000))
    results, total = [], 0
    if words:
        for low, name, rel, kind in entries:
            if all(w in low for w in words):
                total += 1
                if len(results) < limit:
                    folder = rel[:-len(name)].rstrip("/")
                    results.append({"name": name, "path": rel, "folder": folder, "type": kind})
    return {"query": q, "results": results, "total": total,
            "capped": total > len(results), "partial": partial,
            "index": _search_status()}


@app.post("/api/search/refresh")
def search_refresh():
    started = _ensure_search_index(force=True)
    return {"started": started, "index": _search_status()}


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
    # When each active conversion should be finished, counted down the queue
    # in the order it runs - what a waiting comparison needs to know.
    finish_at, clock, clock_ok = {}, 0.0, True
    for r in rows:
        job = dict(r)
        job["progress"] = None
        job["eta_sec"] = None
        job["eta_estimated"] = False
        job["queue_index"] = None

        if job["status"] == "running":
            progress = _job_progress.get(job["id"])
            if progress and progress.get("phase") == "chain":
                progress = _chain_live_progress(job["id"], progress)
            job["progress"] = progress
            eta = _running_eta(r, progress)
            job["eta_sec"] = eta[0]
            job["eta_estimated"] = eta[1]
        elif job["status"] == "queued" and job["mode"] != "compare":
            queue_index += 1
            job["queue_index"] = queue_index
            job["eta_sec"] = _estimate_seconds(r)
            job["eta_estimated"] = job["eta_sec"] is not None
        if job["status"] in ("running", "queued") and job["mode"] != "compare":
            if job["eta_sec"] is None:
                clock_ok = False
            else:
                clock += job["eta_sec"]
            finish_at[job["id"]] = clock if clock_ok else None
        jobs.append(job)

    for job in jobs:
        if job["mode"] == "compare" and job["status"] == "queued":
            _describe_waiting_compare(job, {j["id"]: j for j in jobs}, finish_at)
    return jobs


def _describe_waiting_compare(job, by_id, finish_at):
    """What a queued comparison is waiting for, and roughly until when."""
    params = json.loads(job["params_json"])
    if params.get("source") != "file":
        return
    if not params.get("a"):
        job["progress"] = _job_progress.get(job["id"]) or {"phase": "plan"}
        return
    waiting, done, ends = [], 0, []
    for side in ("a", "b"):
        p = by_id.get(params[side])
        if p is not None and p["status"] == "done":
            done += 1
            continue
        # A reused preview may have dropped out of the 200-row window; ask.
        if p is None:
            with _db() as conn:
                row = conn.execute("SELECT status FROM jobs WHERE id=?",
                                   (params[side],)).fetchone()
            if row is not None and row["status"] == "done":
                done += 1
                continue
        waiting.append(params["label_" + side])
        ends.append(finish_at.get(params[side]))
    job["progress"] = {"phase": "wait", "done": done, "of": 2, "waiting_for": waiting}
    if ends and all(e is not None for e in ends):
        job["eta_sec"] = max(ends) + COMPARE_BUILD_SEC
        job["eta_estimated"] = True


@app.get("/api/queue-eta")
def queue_eta():
    """Total time left: the running job's own ETA plus estimates for the rest.

    Sorted by position (same as _worker_loop), so the numbers line up with the
    order things will actually run in.
    """
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status IN ('running','queued') "
            "AND mode<>'compare' ORDER BY status <> 'running', position, created_at"
        ).fetchall()

    total = 0.0
    exact = True  # false once any part of the sum is a model-based estimate
    counted = 0
    for r in rows:
        if r["status"] == "running":
            progress = _job_progress.get(r["id"])
            if progress and progress.get("phase") == "chain":
                progress = _chain_live_progress(r["id"], progress)
            seconds, estimated = _running_eta(r, progress)
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
    if job.preview_start_sec is not None:
        if job.mode != "preview":
            raise HTTPException(400, "preview_start_sec only means anything for a preview")
        if job.preview_start_sec < 0:
            raise HTTPException(400, "preview_start_sec must not be negative")
    # A named level *is* its parameter set, stored so the row keeps saying what
    # ran even if the table above is later corrected.
    params = dict(level["params"]) if level.get("params") else dict(job.params)

    with _db() as conn:
        job_id = _insert_job(conn, job.mode, job.input_path, job.recursive,
                             job.stereo_format, params, job.quality, job.flow,
                             job.upscale, job.preview_start_sec)
        conn.commit()
    return {"id": job_id}


def _insert_job(conn, mode, input_path, recursive, stereo_format, params, quality,
                flow, upscale, preview_start_sec, parent_id=None):
    """Queue one conversion or preview; previews go in front. Caller commits."""
    job_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO jobs (id, mode, input_path, recursive, stereo_format, params_json, "
        "status, created_at, position, quality, opt_flow, opt_upscale, "
        "preview_start_sec, parent_id) "
        "VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)",
        (job_id, mode, input_path, int(recursive), stereo_format,
         json.dumps(params), _now(), _next_position(conn),
         quality, int(flow), int(upscale),
         preview_start_sec if mode == "preview" else None, parent_id),
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
    if mode == "preview":
        modes = {r["id"]: r["mode"] for r in
                  conn.execute("SELECT id, mode FROM jobs WHERE status='queued'")}
        ids = [i for i in _queued_ids(conn) if i != job_id]
        at = 0
        while at < len(ids) and modes.get(ids[at]) == "preview":
            at += 1
        ids.insert(at, job_id)
        _apply_queue_order(conn, ids)
    return job_id


def _compare_label(row, fallback):
    """What a preview is called on the bar burned into a comparison."""
    level = QUALITY_BY_ID.get(row["quality"], {})
    name = level.get("label") or row["quality"] or "custom"
    switches = []
    if row["opt_flow"]:
        switches.append("+flow")
    if row["opt_upscale"]:
        switches.append("+4K")
    return f"{fallback} - {name}{(' ' + ' '.join(switches)) if switches else ''}"


@app.get("/api/comparable")
def comparable():
    """Finished previews that still have a file, newest first.

    Only previews: a comparison of two full conversions would be an hour of
    encoding and a file nobody can scrub through, and the whole point of the
    lead frames is that a preview is the same recipe as the full run.
    """
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status='done' AND mode='preview' "
            "AND output_path IS NOT NULL ORDER BY finished_at DESC LIMIT 50").fetchall()
    out = []
    for r in rows:
        if not Path(r["output_path"]).is_file():
            continue
        level = QUALITY_BY_ID.get(r["quality"], {})
        size = _file_size_cached(Path(r["output_path"]))
        out.append({
            # The frame size the comparison will actually be built from, so the
            # page can say what "A above B" comes out as.
            "width": size[0] if size else None,
            "height": size[1] if size else None,
            "id": r["id"],
            "input_path": r["input_path"],
            "quality": r["quality"],
            "quality_label": level.get("label") or r["quality"],
            "flow": bool(r["opt_flow"]),
            "upscale": bool(r["opt_upscale"]),
            "preview_start_sec": r["preview_start_sec"],
            "finished_at": r["finished_at"],
            "file": Path(r["output_path"]).name,
        })
    return out


_size_cache: dict[tuple, tuple] = {}


def _file_size_cached(path: Path):
    """(width, height) of a produced file, probed once per (path, mtime)."""
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return None
    if key not in _size_cache:
        info = _stream_info(path)
        _size_cache[key] = (info["width"], info["height"]) if info else None
    return _size_cache[key]


class CompareCreate(BaseModel):
    a: str
    b: str
    layout: str = "aba"


@app.post("/api/compare")
def create_compare(body: CompareCreate):
    if body.layout not in COMPARE_LAYOUTS:
        raise HTTPException(400, f"layout must be one of: {', '.join(COMPARE_LAYOUTS)}")
    if body.a == body.b:
        raise HTTPException(400, "a comparison needs two different previews")
    with _db() as conn:
        rows = {r["id"]: r for r in conn.execute(
            "SELECT * FROM jobs WHERE id IN (?, ?)", (body.a, body.b))}
    for side, job_id in (("A", body.a), ("B", body.b)):
        r = rows.get(job_id)
        if r is None:
            raise HTTPException(404, f"side {side}: no such job")
        if r["status"] != "done" or r["mode"] != "preview" or not r["output_path"]:
            raise HTTPException(400, f"side {side} is not a finished preview")
        if not Path(r["output_path"]).is_file():
            raise HTTPException(400, f"side {side}: its file is gone")
    # Said out loud rather than refused: comparing two windows of two different
    # films is a legitimate thing to want, it just is not a comparison of
    # recipes, and the label bar will make that obvious anyway.
    a_row, b_row = rows[body.a], rows[body.b]
    params = {
        "a": body.a, "b": body.b, "layout": body.layout,
        "label_a": _compare_label(a_row, "A"),
        "label_b": _compare_label(b_row, "B"),
        "same_source": a_row["input_path"] == b_row["input_path"],
    }
    job_id = str(uuid.uuid4())
    with _db() as conn:
        conn.execute(
            "INSERT INTO jobs (id, mode, input_path, recursive, stereo_format, "
            "params_json, status, created_at, position, quality) "
            "VALUES (?, 'compare', ?, 0, ?, ?, 'queued', ?, 0, 'compare')",
            (job_id, a_row["input_path"], a_row["stereo_format"],
             json.dumps(params), _now()))
        conn.commit()
    return {"id": job_id, "same_source": params["same_source"]}


class CompareFileCreate(BaseModel):
    input_path: str
    level_a: str = "fast"
    level_b: str = "standard"
    flow_a: bool = False
    upscale_a: bool = False
    flow_b: bool = False
    upscale_b: bool = False
    layout: str = "aba"
    stereo_format: str = "full_sbs"
    # Null = the rule picks the window (once, for both). A number is the first
    # visible second, as for a single preview.
    preview_start_sec: float | None = None
    reuse: bool = True


@app.post("/api/compare/file")
def create_file_compare(body: CompareFileCreate):
    """Preview A, preview B and their comparison, from one selected file."""
    if not body.input_path.strip():
        raise HTTPException(400, "select a file to compare")
    target = _safe_input_path(body.input_path)
    if not target.is_file():
        raise HTTPException(400, f"not a file: {body.input_path}")
    if body.layout not in COMPARE_LAYOUTS:
        raise HTTPException(400, f"layout must be one of: {', '.join(COMPARE_LAYOUTS)}")
    if body.stereo_format not in COMPARE_STEREO_FORMATS:
        raise HTTPException(400, "a comparison splits every frame into its left and right "
                                 "eye, so it needs a side-by-side format ("
                                 + ", ".join(COMPARE_STEREO_FORMATS) + ")")
    if body.preview_start_sec is not None and body.preview_start_sec < 0:
        raise HTTPException(400, "preview_start_sec must not be negative")
    specs = {}
    for side, lv_id, flow, up in (("a", body.level_a, body.flow_a, body.upscale_a),
                                  ("b", body.level_b, body.flow_b, body.upscale_b)):
        level = QUALITY_BY_ID.get(lv_id)
        if level is None:
            raise HTTPException(400, f"side {side.upper()}: unknown quality level {lv_id}")
        if not level.get("costs"):
            raise HTTPException(400, f"side {side.upper()}: {level['label']} has no fixed "
                                     f"recipe to compare; pick one of the measured levels")
        if level["chain"] and not _chain_available():
            raise HTTPException(400, f"side {side.upper()}: {level['label']} needs the "
                                     f"multi-stage pipeline, which is not installed")
        if (flow or up) and not level["chain"]:
            raise HTTPException(400, f"side {side.upper()}: the flow and 4K switches are "
                                     f"pipeline stages; {level['label']} is a single pass")
        specs[side] = {"quality": lv_id, "flow": bool(flow), "upscale": bool(up)}
    if specs["a"] == specs["b"]:
        raise HTTPException(400, "A and B are the same recipe - there would be nothing to see")
    if not _probe_video(target):
        raise HTTPException(400, "could not read that file as a video")

    params = {
        "source": "file", "layout": body.layout, "a": None, "b": None,
        "spec_a": specs["a"], "spec_b": specs["b"],
        "label_a": _spec_label(specs["a"], "A"), "label_b": _spec_label(specs["b"], "B"),
        "window_request": body.preview_start_sec, "reuse": body.reuse,
        "same_source": True,
    }
    job_id = str(uuid.uuid4())
    with _db() as conn:
        conn.execute(
            "INSERT INTO jobs (id, mode, input_path, recursive, stereo_format, "
            "params_json, status, created_at, position, quality) "
            "VALUES (?, 'compare', ?, 0, ?, ?, 'queued', ?, 0, 'compare')",
            (job_id, body.input_path, body.stereo_format, json.dumps(params), _now()))
        conn.commit()
    return {"id": job_id, "label_a": params["label_a"], "label_b": params["label_b"]}


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
        row = conn.execute("SELECT status, mode FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "not found")
        if row["status"] == "queued":
            conn.execute("UPDATE jobs SET status='canceled', finished_at=?, error=NULL "
                          "WHERE id=?", (_now(), job_id))
            dropped = []
            if row["mode"] == "compare":
                # A comparison that is still waiting takes the previews it
                # queued itself along with it - but only those still waiting.
                # One already on the GPU, and any finished one it was going to
                # reuse, stay: they are previews in their own right.
                dropped = [r["id"] for r in conn.execute(
                    "SELECT id FROM jobs WHERE parent_id=? AND status='queued'", (job_id,))]
                for pid in dropped:
                    conn.execute("UPDATE jobs SET status='canceled', finished_at=? WHERE id=?",
                                  (_now(), pid))
            conn.commit()
            proc = _running_procs.get(job_id)
            if proc is not None:
                # The comparison's window scan.
                _cancel_requested.add(job_id)
                proc.send_signal(signal.SIGTERM)
            return {"ok": True, "canceled_previews": dropped}
        if row["status"] != "running":
            raise HTTPException(409, "job is not queued or running")
    # Remembered before the signal goes out, so the job's own ending reads it
    # as a cancellation however quickly the process dies.
    _cancel_requested.add(job_id)
    proc = _running_procs.get(job_id)
    if proc is not None:
        proc.send_signal(signal.SIGTERM)
        return {"ok": True, "note": "SIGTERM sent to running job"}
    # Between two processes (a preview between cutting and converting): the
    # job checks the record before it starts the next one.
    return {"ok": True, "note": "cancel recorded; the job stops before its next step"}


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
