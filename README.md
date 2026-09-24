# iw3-webui

A job queue and browser UI for [iw3](https://github.com/nagadomi/nunif), nunif's
2D → stereo-3D video converter.

iw3 upstream ships a CLI and a wxPython desktop GUI. Neither has a queue, so a
batch of films means either babysitting one conversion at a time or writing a
shell loop and losing all visibility into it. Conversions are long — a 90-minute
4K film is on the order of 18 hours on the hardware this was written for — which
makes "what is it doing and when will it be done" the question that actually
matters.

This gives iw3 a persistent queue with live progress, per-job logs and honest
ETAs, in a container you can put on whatever machine holds the GPU.

## What's in here

| Directory | What it is |
|---|---|
| `container/` | The queue, the web UI and the Dockerfile. Stands alone; needs nothing else. |
| `cove-extension/` | Optional. An **Add to iw3 Queue** button for [Cove](https://github.com/coveapp/cove)'s video detail page, which posts to this queue over HTTP. |

The extension needs the container. The container does not need the extension.

## Features

- **Persistent queue** — SQLite under your config volume. Survives restarts; a
  job that was running when the container died is re-queued rather than lost.
- **Reorderable queue** — drag a queued row, or hit ⤒/⤓ to send it to the front
  or the back. See [Queue order](#queue-order).
- **Real progress, not a spinner** — iw3 drives tqdm, which already computes
  percentage, frame counts, rate and remaining time. The backend parses those
  rather than inventing its own numbers. The scene-detection pre-pass draws its
  own bar and is deliberately shown as a *separate*, greyed-out phase so it can
  never be mistaken for conversion progress.
- **ETAs for jobs that haven't started**, from throughput measured on *your*
  machine — see [Estimates](#estimates).
- **Live log streaming** over SSE, throttled so a multi-hour job doesn't take
  the browser tab down with it. The log opens under the job's own row.
- **A short history** — every queued and running job, plus the ten most
  recently finished. Older rows stay in the database (the estimates are
  calibrated from them) and are simply not listed; `GET /api/jobs?history=N`
  returns more.
- **A waiting line** for videos you want to convert some other day — parked
  from the page or over HTTP, never started by themselves. See
  [Waiting line](#waiting-line).
- **Search** over every file and folder name, limited to the folder that is
  open in the picker (the whole share at the top).
- **One-minute preview**, on a window chosen by rule — see [Preview](#preview).
- **Two previews in one file**, interleaved per eye — see [Comparing two previews](#comparing-two-previews-in-one-file).
- **Settings read from iw3 itself** — the form's fields, defaults and choices
  are introspected from iw3's own `create_parser()` at startup, so they cannot
  drift out of sync with the nunif version in the image.
- **Quality levels with honest times** — three measured recipes instead of
  twelve free-form settings, each showing what it costs *for the video you
  selected* and what you give up for the price. See
  [Quality levels](#quality-levels).

## Requirements

- Docker
- A GPU passed into the container, or patience (the CPU works; it is very slow)
- Somewhere to read source video from, and somewhere to write results to

## Quick start

Images are prebuilt per backend, so there is nothing to compile. Pick the tag
that matches your GPU — `cuda`, `xpu` or `cpu`:

```sh
docker run -d --name iw3 \
  --restart unless-stopped \
  --gpus all \
  -p 8790:8790 \
  -e PUID=1000 -e PGID=1000 \
  -v /path/to/config:/config \
  -v /path/to/videos:/input:ro \
  -v /path/to/output:/output \
  ghcr.io/yast2/iw3-webui:cuda
```

Then open `http://<host>:8790`.

The device flag differs, and it is the one thing worth getting right:

| Backend | tag | flag |
|---|---|---|
| NVIDIA | `:cuda` | `--gpus all` |
| Intel Arc | `:xpu` | `--device /dev/dri:/dev/dri:rwm` |
| no GPU | `:cpu` | *(none)* |

There is no device to configure beyond that. `IW3_GPU` defaults to `auto`: the
container looks for an accelerator at startup and uses it, or falls back to the
CPU. Set `0`, `1` or `-1` if you would rather decide yourself.

Or with compose — `docker compose --profile cuda up -d`, after editing the
three paths in [`docker-compose.yml`](docker-compose.yml).

### If it seems slow, it is probably on the CPU

Forgetting the device flag does not produce an error. It produces a container
that works and is dozens of times slower, which looks exactly like a big job.
So the container refuses to be quiet about it: a line in the startup log, a
banner across the top of the web UI, and:

```sh
curl -s localhost:8790/api/health
```

`"device": "cpu"` with a `warning` means the GPU never made it in.

### Volumes

| Mount | Purpose |
|---|---|
| `/config` | `NUNIF_HOME`: model checkpoints, the job database, per-job logs |
| `/input` | Your source videos. Mount read-only; the container never writes here. |
| `/output` | Converted files; previews and comparisons in `Previews/<video>/`; work in progress in `_chain/<job-id>/` |

### Model checkpoints

Several depth checkpoints are CC-BY-NC-4.0 licensed and are **not**
auto-downloaded. On first start the container writes a `README.md` into
`/config` listing the exact filenames and their Hugging Face sources. Models not
on that list (`ZoeD_*`, DepthPro, Depth-Anything v1) download themselves on
first use.

If you pick a model whose checkpoint is missing, iw3 fails with a
`FileNotFoundError` naming the exact path — that is iw3's own behaviour, not a
check added here.

## Queue order

Queued rows are draggable, and each carries ⤒ (run next) and ⤓ (run last).
Dragging is the better gesture over a screenful; the buttons are the ones you
want when the queue is a hundred deep and the row you care about is off-screen.

The order lives in a `position` column and is the single key the worker, the
ETA total and the table all sort by — so the order on screen is the order the
GPU will work through. Existing databases are migrated on first start: the
queue is seeded from the order it already had, so an upgrade mid-queue does not
shuffle anything.

Two things deliberately cannot be dragged:

- **The running job.** It is already on the GPU, and nothing in the queue
  interrupts work in progress. It always sorts first, and moving it is a 409.
- **Finished jobs.** They are not in the queue to begin with.

Previews still jump to the front when created, for the reason in
[Preview](#preview) below — but that is now an *insertion* rule rather than a
sort rule. Once queued, a preview reorders like any other row. The queue has
exactly one order, and it is the one you can see.

Reordering from the outside:

```sh
# run this job next
curl -X POST localhost:8790/api/jobs/<id>/move \
     -H 'Content-Type: application/json' -d '{"to":"top"}'
# to, bottom, up and down are all accepted

# or set the whole order at once
curl -X POST localhost:8790/api/queue/reorder \
     -H 'Content-Type: application/json' -d '{"order":["<id>","<id>", ...]}'
```

`/api/queue/reorder` is a merge, not an assignment: ids it no longer has queued
are ignored, and queued ids you left out keep their relative order behind the
ones you sent. A browser tab that is a few seconds stale therefore cannot lose
or resurrect a job by dragging one row.

## Preview

The **Preview one minute** button converts a one-minute window of the source
with exactly the settings the real job would use.

Three decisions worth explaining:

- **A clip, not stills.** This button used to pass iw3's `--keyframe` and
  produce a handful of images. Stills cannot answer the question a 3D preview
  exists to answer — whether depth stays stable while the picture moves.
  Flicker, and depth bleeding across a hard cut, only show up in motion.
- **The most telling minute, not the middle.** The middle of a file is as
  likely as anywhere else to be a minute in which nothing happens, and every
  recipe looks fine on a static shot. The default window is instead chosen by
  rule: the minute with the most frame-to-frame motion that also *contains a
  cut*, positioned so the cut falls near its middle. A cut is the hardest
  thing a depth model has to survive, and a preview that never crosses one has
  not been asked the question. If the file has no cut at all, the
  highest-motion minute is taken and the job log says so. Pick **Start at a
  time I choose** to override it, which also skips the scan.

  The scan is a 160×90 greyscale decode of the whole file — about 30 s per 12
  minutes of video, against 5 to 25 minutes for the preview it is choosing.

- **Lead frames, and they are not optional.** The pipeline runs 200 frames
  *before* the window and 30 after it, and both are cut away again before you
  see the file. The convergence estimator carries an exponential moving
  average across frames, so a window started cold is not the same recipe as
  the full run — it is a slightly different one, which is the one thing a
  preview must not be. So a 60 s preview of a 29.97 fps source costs
  200 + 1798 + 30 = 2,028 frames of 21,606, or 9.4%, and that is the figure
  the level picker quotes.

The clip handed to the conversion is produced by copying the video stream — no
re-encode, so what you judge is the real source. Because a stream copy always
starts at a keyframe, the clip begins a little before the window asked for;
the exact difference is measured and taken off the front along with the lead.
Audio is transcoded to AAC because mp4 will not accept every audio codec that
arrives in an mkv or wmv. If the video codec itself cannot go into mp4
(`wmv3`, for instance), the clip is re-encoded and the log says so. The clip is
deleted once the preview finishes, and kept if it fails, so you can look at
what iw3 choked on.

A preview is worked on in its scratch folder, `/output/_chain/<job-id>/`, and
only the finished file is moved to where it is browsed:
`/output/Previews/<video>/<video> - <Level><tag>.mp4`, for instance
`Previews/Holiday/Holiday - Standard_G2EMA_LRF.mp4`. The tag at the end is the
one the conversion wrote, untouched, because players read the side-by-side
format from it. A second preview of the same level gets its window start into
the name (`… - Standard from 5m30s_…`). Scratch goes when a job is done or
canceled; a failed job's is kept for 48 hours (`SCRATCH_KEEP_FAILED_HOURS`) so
there is something to look at, then swept. Previews made by earlier versions
(`_previews/<job-id>/`, `_compare/<job-id>/`) are moved into `Previews/` on
start.

`PREVIEW_SECONDS`, `PREVIEW_LEAD_FRAMES` and `PREVIEW_TAIL_FRAMES` change the
three numbers above.

## Comparing two previews in one file

Two finished previews of the same window can be combined into a single file to
watch in a headset. The **Compare** panel appears once there are two to
compare.

🔴 **The part a naive implementation gets wrong:** a preview is already
side-by-side stereo. `3840x1080` is not a picture, it is *left eye | right
eye*. Putting two of them next to each other produces

```
A-left  A-right  B-left  B-right
```

and a headset splits that down the middle, so it shows A's right eye to the
left eye and B's left eye to the right eye — two different recipes, one per
eye, at a disparity that means nothing. Both arrangements below are therefore
built **per eye**: each source is cut into its two halves, the halves are
labelled and combined, and only then are the two finished eyes put back
together as one side-by-side frame.

| Arrangement | Output | What it is for |
|---|---|---|
| **A / B / A** (default) | same geometry as the inputs, three times the length | The same window three times. Seeing the change twice, and in both directions, finds differences that sitting side by side does not. Nothing about the geometry changes, so nothing can misread it. |
| **A above B** | `3840x2160`, each eye `1920x2160` | Both in view at once. Vertical rather than horizontal, so each keeps a full `1920x1080` picture and both sit centred — side by side within an eye would give each one half a picture in a 32:9 strip. ⚠️ Each eye is then 16:18; a player that assumes 16:9 per eye will stretch it. |

The label bar is drawn into each eye separately, at identical coordinates and
fully opaque. A label at a different position in the two eyes has a disparity
of its own and floats in front of the picture; a translucent one lets the two
eyes' differing pixels through and flickers.

Comparisons do not queue behind conversions. They run on their own worker,
because a comparison is two or three minutes of fixed-function video encoder —
a different unit of the chip from the one a conversion occupies — and behind a
four-hour job a tool for looking at two previews is worth nothing.

The output lands next to preview A, as
`/output/Previews/<video>/<video> - Fast vs Standard (stacked)_LRF.mp4` (or
`(A-B-A)`).

## Waiting line

Videos you have picked but do not want converted yet. A parked video never
starts by itself: it is a row in a table of its own that nothing in the queue
reads. Each row shows the video's length and size, its settings — a level
(Standard unless changed) and the flow and 4K switches where the level takes
them — and what those settings cost for that file, priced by the same model the
queue uses, so the figure is the one the queue will show. The settings are
stored with the row as you change them (there is no Save button), so they
survive a reload and read the same on every device. From there, for any
selection:

- **Queue** — one full conversion per video, at its row's settings; queued
  videos leave the waiting line.
- **Preview** — one one-minute preview per video at its row's settings, on the
  window the rule picks, exactly as the Preview button makes it. The row shows
  the preview's state and, once it is done, its file name. A preview of the
  same file at the same settings that is still queued or running is not queued
  a second time.
- **Compare** — two previews of the same window and the comparison of the two.
  What each row compares is written under its level before you click:
  - flow or 4K on: the same level without the switches (A) against the row as
    set (B) — e.g. *Standard vs Standard +flow*;
  - no switches, Standard or Economical: *Fast vs* that level;
  - Fast: *Fast vs Standard*.
- **Remove** — out of the waiting line; nothing on disk is touched.

Preview and Compare leave the video parked. Previews that already exist for
the same file and settings are reused by Compare, as everywhere else — except
a finished pipeline preview whose recipe code lacks `Kante` (`_G2EMA_`,
`_G2EMAFluss_`, …): it was made before the pipeline's edge fix became
compulsory and is a different recipe under the same name. It is never reused,
and where it is listed (the picker for comparing two existing previews, a
row's preview badge) it says `(no edge fix)`. Fast is not affected.

The levels and their recipes belong to this app. A client names a file, at
most a level and the two switches, never parameters:

```sh
# park one or many (title and an outside id are optional, shown in the list)
curl -X POST localhost:8790/api/waiting -H 'Content-Type: application/json' \
     -d '{"items":[{"input_path":"films/Holiday.mp4","title":"Holiday"}],"source":"cove"}'
# every row with its settings, what Compare would make, and its preview/comparison
curl localhost:8790/api/waiting
# change a row's settings (fields left out keep their value; Fast turns the switches off)
curl -X PATCH localhost:8790/api/waiting/<id> -H 'Content-Type: application/json' \
     -d '{"quality":"standard","flow":true}'
# queue at the row's settings (or name quality/flow/upscale per item to override)
curl -X POST localhost:8790/api/waiting/queue -H 'Content-Type: application/json' \
     -d '{"items":[{"id":"<id>"}]}'
curl -X POST localhost:8790/api/waiting/preview -H 'Content-Type: application/json' \
     -d '{"ids":["<id>"]}'
# compare as the row's settings say; level_a/level_b force a plain pair instead
curl -X POST localhost:8790/api/waiting/compare -H 'Content-Type: application/json' \
     -d '{"ids":["<id>"],"layout":"stacked"}'
curl -X POST localhost:8790/api/waiting/remove -H 'Content-Type: application/json' \
     -d '{"ids":["<id>"]}'
```

`preview` and `compare` also take `"items":[{"id":…,"quality":…,"flow":…,"upscale":…}]`:
settings stored before the request acts, which is how the page makes sure a
switch flipped a moment before the button is the one used.

Parking a file that is already parked keeps the one row. Every call answers per
item (`added`, `already waiting`, `queued`, `previewing`, `already previewing`,
`comparing`, `already comparing`, `rejected` with the reason), so one bad path
does not sink a batch.

## Quality levels

The settings form is the full truth and a poor first question. The level picker
asks the question people actually have — good, or fast? — and answers it with
measured recipes rather than plausible defaults:

| Level | What it uses | Gains | Costs | Pipeline |
|---|---|---|---|---|
| **Fast** | One `iw3` process: VDA_B with EMA normalisation, `row_flow` warp, `sod_v1` convergence, x265. | 58 min for a 12-minute film, and the calmest of the three over time (0.0382 px of flicker on still surfaces). | **39% less relief on silhouettes** and about half the fine detail. Takes neither switch. | no |
| **Economical** | Reduced DepthPro_S for fine structure, VDA_B with EMA for the coarse band, band swap, `mlbw_l2` warp. | An hour faster than Standard, and within 8% of its silhouettes. | Softer outlines (−8.0%) and markedly less fine structure (0.8534 against 0.9988). | yes |
| **Standard** | The same chain with DepthPro at full resolution. | Keeps the depth model's fine band intact (0.9988) and the sharpest silhouettes on offer. | 4 h 19 for a 12-minute film, four and a half times Fast. | yes |
| **Custom** | Whatever you put in the form. | Reaches settings the levels fix. | Not a measured recipe; nothing about the result is promised. | no |

Each level carries the full list in the UI, advantages and disadvantages in the
same type — a level whose drawbacks are set smaller than its benefits is being
sold rather than described.

Two switches, both off by default, apply to the multi-stage levels: optical
flow smoothing and denoise-plus-2×-upscale before the warp. Each states what it
costs and what it buys; neither was clearly worth its hours in side-by-side
viewing, which is why they are switches and not levels.

Every figure shown is recomputed for the frame count of the selected source, so
the picker never quotes the length of the file the measurements were taken on.
The underlying numbers, and the caveats on each, come from `/api/quality`:

```sh
curl -s localhost:8790/api/quality
curl -s 'localhost:8790/api/estimate?path=some/film.mkv'
```

### The multi-stage pipeline is not in this repository

The chained levels run five or six programs per job — frame extraction, depth
estimation, depth post-processing, warp, encode — and that pipeline is a
separate, machine-specific script. This app execs it with a level name and
parses its stage markers; it does not reimplement it.

Point `IW3_CHAIN_SCRIPT` at an executable that accepts

```
--level <fast|economical|standard> -i <input file> -o <output dir>
--work <scratch dir> --gpu <id> --stereo-format <name> [--flow] [--upscale]
```

exits non-zero on failure, and announces each stage on its own line as

```
=== 2/6 depth estimation  2026-09-21 15:04:11
```

Without it, the chained levels are shown disabled with the reason, and nothing
else changes. The scratch directory is deleted after a successful job and kept
after a failure.

## Estimates

Queued jobs get an ETA, and the queue header shows the total. Runtime scales
with *frames processed* — duration × `min(source fps, max_fps)` — not with clip
length, so a 50 fps source costs roughly twice a 25 fps source of the same
running time.

The frames-per-second figures come from **your own finished jobs**, grouped by
depth model and resolution and taken as a median. Until a combination has run on
your machine at least once, a seed value measured on an Intel Arc Pro B60 stands
in. See what is being used:

```sh
curl -s localhost:8790/api/throughput
```

Estimates are shown with a leading `~` in grey. A countdown reported by iw3
itself is shown plain. The two are never mixed.

## Configuration

Everything below is an environment variable on the container.

| Variable | Default | Meaning |
|---|---|---|
| `WEBUI_PORT` | `8790` | Port inside the container |
| `PUID` / `PGID` | `99` / `100` | User/group the process drops to; owns the output files |
| `UMASK` | `000` | umask for created files |
| `IW3_GPU` | `auto` | `auto` detects the accelerator; `0`/`1` pick one explicitly, `-1` forces the CPU |
| `PREVIEW_SECONDS` | `60` | Length of the visible part of a preview |
| `PREVIEW_LEAD_FRAMES` | `200` | Frames run before the window and then cut away — see [Preview](#preview) |
| `PREVIEW_TAIL_FRAMES` | `30` | Frames run after the window and then cut away |
| `FFMPEG_BIN` | `ffmpeg` | ffmpeg used for the motion scan and clip extraction |
| `FFPROBE_BIN` | `ffprobe` | ffprobe used to locate keyframes and check produced files |
| `QSV_FFMPEG` | `/usr/lib/jellyfin-ffmpeg/ffmpeg` | Build used for this app's own two re-encodes |
| `QSV_DEVICE` | `/dev/dri/renderD128` | Render node for the hardware encoder |
| `QSV_GLOBAL_QUALITY` | `17` | `-global_quality` for `hevc_qsv` |
| `OUTPUT_UID` / `OUTPUT_GID` | `99` / `100` | Owner set on delivered files |
| `NUNIF_HOME` | `/config` | Checkpoints, queue database, logs |

Only one job runs at a time regardless. iw3 was not built to share a device
between concurrent conversions.

## Other GPUs

There is no vendor-specific code in this project, and none is needed: nunif
resolves the backend itself (`cuda` → `mps` → `xpu`, see `nunif/device.py`), so
one build differs from another only in which base image carries which torch.
The full matrix and the build commands are in
[`container/BUILD.md`](container/BUILD.md).

What is actually verified:

| Backend | builds | converts |
|---|---|---|
| Intel Arc / XPU | ✅ | ✅ **measured** on an Arc Pro B60 |
| NVIDIA / CUDA | ✅ in CI | ❓ never run — no NVIDIA hardware here |
| CPU only | ✅ in CI | ❓ never run |
| AMD / ROCm | ❓ not in CI | ❓ never run |

Every push builds all three CI backends and runs a smoke test that imports
torch and iw3 inside the finished image, so a broken Dockerfile is caught
without hardware. Whether the *conversion* is correct on CUDA or ROCm is a
question this project cannot answer on its own.

I would rather say "unverified" than imply a result I have never seen. If you
run one of these, a report — or a PR correcting this table — is the most useful
thing you could send.

## Security

**There is no authentication.** Anyone who can reach the port can queue jobs,
read logs and browse the directory tree under `/input`. This was built for a
LAN. Put it behind a reverse proxy with auth, or don't expose it.

## Credits

All of the actual conversion is [nagadomi/nunif](https://github.com/nagadomi/nunif).
This repository is a queue and a web front end around `python -m iw3` — it does
not change how iw3 converts anything.

## License

MIT — see [LICENSE](LICENSE). nunif is MIT as well; several depth model
checkpoints are CC-BY-NC-4.0 and are neither redistributed nor auto-downloaded
here.
