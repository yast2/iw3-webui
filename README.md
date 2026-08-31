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
  the browser tab down with it.
- **2-minute preview** — see [Preview](#preview).
- **Settings read from iw3 itself** — the form's fields, defaults and choices
  are introspected from iw3's own `create_parser()` at startup, so they cannot
  drift out of sync with the nunif version in the image.

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
| `/output` | Converted files, and `_previews/<job-id>/` for previews |

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

The **Preview (2 min clip)** button cuts a two-minute clip from the *middle* of
the source and converts that, with exactly the settings the real job would use.

Two decisions worth explaining:

- **A clip, not stills.** This button used to pass iw3's `--keyframe` and
  produce a handful of images. Stills cannot answer the question a 3D preview
  exists to answer — whether depth stays stable while the picture moves.
  Flicker, and depth bleeding across a hard cut, only show up in motion.
- **The middle, not the start.** Openings are titles, logos and fades often
  enough to be unrepresentative of the film behind them.

The clip is produced by copying the video stream — no re-encode, so what you
judge is the real source. Audio is transcoded to AAC because mp4 will not
accept every audio codec that arrives in an mkv or wmv. If the video codec
itself cannot go into mp4 (`wmv3`, for instance), the clip is re-encoded and the
log says so. The clip is deleted once the preview finishes, and kept if it
fails, so you can look at what iw3 choked on.

Set `PREVIEW_CLIP_SECONDS` to change the length.

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
| `PREVIEW_CLIP_SECONDS` | `120` | Length of the preview clip |
| `FFMPEG_BIN` | `ffmpeg` | ffmpeg used for clip extraction |
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
