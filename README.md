# iw3-webui

**A job queue and web interface for [iw3](https://github.com/nagadomi/nunif)**,
the part of nunif that turns ordinary 2D video into stereo 3D for VR headsets.

iw3 comes with a command line and a desktop app, but neither has a queue. That
starts to hurt quickly: a 90-minute 4K film takes around 18 hours on the
machine this was built for, so converting a batch means either sitting next to
it or writing a shell loop and hoping for the best.

iw3-webui fills that gap. You pick videos in your browser, they go into a queue
that survives restarts, and you can always see what's running, how far along
it is and when it'll be done. It runs as a Docker container on whichever
machine has the GPU.

## Contents

- [What's in this repository](#whats-in-this-repository)
- [Features](#features)
- [Getting started](#getting-started)
- [The queue](#the-queue)
- [Quality levels](#quality-levels)
- [Previews](#previews)
- [Comparing two previews](#comparing-two-previews)
- [The waiting line](#the-waiting-line)
- [Time estimates](#time-estimates)
- [Configuration](#configuration)
- [GPU support](#gpu-support)
- [Security](#security)
- [Credits and license](#credits-and-license)

## What's in this repository

| Folder | What it is |
|---|---|
| [`container/`](container/) | The queue, the web interface and the Dockerfile. This is all you need. |
| [`chain/`](chain/README.md) | The multi-stage pipeline behind the Economical and Standard levels, in two versions: one for an Intel Arc server and one for an NVIDIA desktop. |
| [`cove-extension/`](cove-extension/README.md) | Optional. Adds an **Add to iw3 Queue** button to [Cove](https://github.com/coveapp/cove)'s video page. |

The Cove button needs the container, but the container doesn't need the button.

## Features

- **A queue that survives restarts.** Jobs are kept in a small database in
  your config folder. If the container stops in the middle of a job, that job
  goes back into the queue rather than getting lost.
- **Drag to reorder.** Move a job by dragging it, or send it straight to the
  front (⤒) or the back (⤓).
- **Real progress.** The numbers come straight from iw3's own progress bars:
  percentage, frames, speed and time left. Scene detection, which runs first,
  is shown separately in grey so it's never mistaken for the conversion.
- **Time estimates before a job starts**, plus a total for the whole queue.
- **Live logs**, opening right under the job they belong to. Updates are
  throttled, so a job that runs for hours won't freeze your browser tab.
- **Quality levels instead of a wall of settings.** Pick Fast, Economical or
  Standard and see what each would cost for *your* video, and what you give up
  for the speed.
- **One-minute previews** of the most demanding minute of a film, so you can
  judge the result before committing hours of GPU time.
- **Comparisons** of two previews in a single file you can watch in a headset.
- **A waiting line** for videos you want to convert another day.
- **Search** across every file and folder name.
- **Settings that always match your iw3.** The full settings form is read from
  iw3 itself at startup, so it can't fall out of date.

## Getting started

You'll need Docker, a GPU you can hand to the container, and folders for your
source videos and the results. It also runs without a GPU, just very slowly.

Ready-made images exist for NVIDIA, Intel Arc and CPU only, so there's
nothing to compile. For an NVIDIA card:

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

Then open `http://<your-server>:8790`.

For another kind of GPU, change the image tag and the device flag:

| GPU | Image tag | Device flag |
|---|---|---|
| NVIDIA | `:cuda` | `--gpus all` |
| Intel Arc | `:xpu` | `--device /dev/dri:/dev/dri:rwm` |
| None | `:cpu` | *(none)* |

The container finds the GPU on its own (`IW3_GPU=auto`). If you'd rather
choose, set `IW3_GPU` to `0` or `1` for a particular card, or `-1` for the
CPU.

Prefer Compose? Fill in the three paths in
[`docker-compose.yml`](docker-compose.yml) and run
`docker compose --profile cuda up -d` (or `xpu`, or `cpu`).

### Is it really using the GPU?

Forgetting the device flag doesn't cause an error. The container just runs on
the CPU, dozens of times slower, which can easily pass for a big job. So it
tells you in three places: a line in the startup log, a banner across the top
of the page, and the health check:

```sh
curl -s localhost:8790/api/health
```

If that shows `"device": "cpu"` with a `warning`, the GPU didn't make it into
the container.

### Folders

| Mount | What goes there |
|---|---|
| `/config` | Model files, the job database and the job logs |
| `/input` | Your source videos. Mount it read-only; nothing is ever written here. |
| `/output` | Finished conversions. Previews and comparisons go into `Previews/<video>/`, work in progress into `_chain/<job-id>/`. |

### Model files

A few depth models are released under a non-commercial licence
(CC-BY-NC-4.0), so they aren't downloaded automatically. On first start the
container writes a `README.md` into `/config` with the exact file names and
where to get them on Hugging Face. Everything else (`ZoeD_*`, DepthPro,
Depth-Anything v1) downloads itself the first time it's used.

If a model file is missing, iw3 stops with a `FileNotFoundError` that names
the path it was looking for.

## The queue

The order you see is the order the GPU works through: the worker, the total
time and the table all go by the same list.

- **Drag** a queued job to move it. When the queue is long and the job you
  want is off-screen, use **⤒** to run it next or **⤓** to run it last.
- **The running job stays where it is.** It's already on the GPU and always
  sits at the top; trying to move it returns a 409.
- **New previews go to the front**, because a preview is only useful while
  you're still deciding. Once queued, they move like any other job.
- **Upgrading keeps your order.** An existing database is converted on first
  start, in the order it already had.

The page lists everything queued or running, plus the ten most recently
finished jobs. Older jobs stay in the database, where they feed the time
estimates; `GET /api/jobs?history=N` lists more of them.

You can reorder from a script too:

```sh
# run this job next ("bottom", "up" and "down" work as well)
curl -X POST localhost:8790/api/jobs/<id>/move \
     -H 'Content-Type: application/json' -d '{"to":"top"}'

# set the whole order at once
curl -X POST localhost:8790/api/queue/reorder \
     -H 'Content-Type: application/json' -d '{"order":["<id>","<id>", ...]}'
```

`/api/queue/reorder` merges rather than replaces. IDs that are no longer
queued are ignored, and queued jobs you leave out keep their order behind the
ones you sent. So a browser tab that's a few seconds out of date can't lose a
job or bring one back.

## Quality levels

Instead of a dozen settings, the page offers three tested recipes. Each one
shows how long it would take for the video you've picked, and what you gain
and lose compared with the others.

| Level | How it works | Good | Not so good |
|---|---|---|---|
| **Fast** | A single iw3 run: VDA_B depth with EMA smoothing, `row_flow` warp, `sod_v1` convergence, x265. | 58 minutes for a 12-minute film, and the steadiest picture of the three (0.0382 px of flicker on still surfaces). | 39 % less depth around outlines and about half the fine detail. No extra switches. |
| **Economical** | The [multi-stage pipeline](#the-multi-stage-pipeline): a reduced DepthPro_S for fine detail, VDA_B with EMA for the overall shape, the two combined, then an `mlbw_l2` warp. | An hour quicker than Standard, and within 8 % of its outlines. | Softer outlines (−8.0 %) and noticeably less fine detail (0.8534 against 0.9988). |
| **Standard** | The same pipeline with DepthPro at full resolution. | Keeps all of DepthPro's fine detail (0.9988) and gives the sharpest outlines on offer. | 4 h 19 min for a 12-minute film, four and a half times as long as Fast. |
| **Custom** | Whatever you set in the full settings form. | Reaches settings the levels don't. | Not a tested recipe, so there are no promises about the result. |

The page shows the full list of pros and cons for every level, both in the
same size of type.

Economical and Standard have two optional switches, both off by default:
**flow**, which smooths the depth over time along the movement in the
picture, and **4K**, which denoises the frames and doubles their resolution
before the warp. Each shows what it costs and what it gives you. In
side-by-side viewing neither was clearly worth the extra hours, which is why
they're switches rather than levels of their own.

All figures are worked out for the frame count of the video you've selected,
not the film they were measured on. The raw numbers and their caveats:

```sh
curl -s localhost:8790/api/quality
curl -s 'localhost:8790/api/estimate?path=some/film.mkv'
```

### The multi-stage pipeline

Economical and Standard don't run iw3 just once. They run a chain of five to
seven programs per job: frame extraction, depth estimation, depth clean-up,
warp and encode. That chain is a separate script and depends on the machine
it runs on. iw3-webui starts it with the level's name and follows its
progress; it doesn't do the work itself. The scripts used here, one for an
Intel Arc server and one for an NVIDIA desktop, are in
[`chain/`](chain/README.md).

To plug in your own, point `IW3_CHAIN_SCRIPT` at a program that:

- accepts these options:

  ```
  --level <fast|economical|standard> -i <input file> -o <output dir>
  --work <scratch dir> --gpu <id> --stereo-format <name> [--flow] [--upscale]
  ```

- exits with a non-zero status when it fails, and
- prints a line like this at the start of each stage:

  ```
  === 2/6 depth estimation  2026-09-21 15:04:11
  ```

Without it, Economical and Standard are greyed out with the reason shown, and
everything else works as usual. The scratch folder is deleted after a
successful job and kept for a while after a failed one (see
[Previews](#previews)).

## Previews

**Preview one minute** converts one minute of a video with exactly the
settings the full job would use. It's the cheap way to find out whether a
recipe suits a film before you spend hours on it.

**It picks the minute for you.** Almost any recipe looks fine on a quiet
shot, so a random minute tells you very little. Instead it looks for the
minute with the most movement that also contains a hard cut, and places the
cut near the middle, because cuts are the hardest thing for depth estimation
to get through. If a film has no cuts, it takes the minute with the most
movement and says so in the log. To choose yourself, pick **Start at a time I
choose**, which also skips the search.

Finding that minute means skimming the whole file at low resolution (160×90,
greyscale). That takes about 30 seconds per 12 minutes of video, against 5 to
25 minutes for the preview itself.

**It's a real clip, not a few stills.** The button used to make a handful of
still images (iw3's `--keyframe`). Stills can't show what a 3D preview is for:
whether the depth stays steady while the picture moves. Flicker, and depth
leaking across a cut, only show up in motion.

<details>
<summary>Why a preview converts a little more than one minute</summary>

The pipeline converts 200 frames before the minute and 30 after it, then cuts
them off again. The depth smoothing carries a running average from frame to
frame, so starting cold right at the minute would give a slightly different
result from the full run, and a preview has to match the real thing. For a
60-second preview of a 29.97 fps film that's 200 + 1,798 + 30 = 2,028 frames
out of 21,606, or 9.4 %. That's the figure the level picker quotes.

The clip is cut out by copying the video stream without re-encoding it, so
what you judge is the real source. A stream copy always starts at a keyframe,
so the clip begins slightly early; the exact difference is measured and
trimmed off along with the extra frames. Audio is converted to AAC, because
mp4 doesn't accept every audio format found in mkv or wmv files. If the video
format itself can't go into mp4 (`wmv3`, for example), the clip is re-encoded
and the log says so. The clip is deleted once the preview is done, or kept if
it fails, so you can see what iw3 tripped over.

</details>

**Where previews end up.** A preview is made in its own scratch folder,
`/output/_chain/<job-id>/`, and only the finished file is moved to
`/output/Previews/<video>/<video> - <Level><tag>.mp4`, for example
`Previews/Holiday/Holiday - Standard_G2EMAKante_LRF.mp4`. The tag at the end
is left exactly as iw3 wrote it, because players read the 3D format from it.
A second preview at the same level gets its start time in the name
(`… - Standard from 5m30s_…`).

Scratch folders are removed when a job finishes or is cancelled. A failed
job's folder is kept for 48 hours so you can look inside, then cleared away.
Previews made by older versions (`_previews/<job-id>/`, `_compare/<job-id>/`)
are moved into `Previews/` on startup.

`PREVIEW_SECONDS`, `PREVIEW_LEAD_FRAMES` and `PREVIEW_TAIL_FRAMES` change the
length and the extra frames before and after.

## Comparing two previews

Once you have two finished previews of the same minute, the **Compare** panel
combines them into one file to watch in a headset. There are two layouts:

| Layout | What you get | Good for |
|---|---|---|
| **A / B / A** (default) | The same minute three times in a row: A, then B, then A again, at the same size as the previews. | Spotting differences. Seeing the switch twice, in both directions, shows things you'd miss side by side, and the picture itself isn't changed at all. |
| **A above B** | Both at once, one above the other, at `3840x2160` (each eye `1920x2160`). | Seeing both together. Stacking keeps each one a full `1920x1080` picture in the middle of your view; side by side, each would be half a picture in a narrow strip. ⚠️ Each eye is 16:18, so a player that assumes 16:9 per eye will stretch it. |

<details>
<summary>Why comparisons are built eye by eye</summary>

A preview is already side-by-side 3D: a `3840x1080` frame is the left eye
next to the right eye. Simply putting two previews next to each other would
give

```
A-left  A-right  B-left  B-right
```

and a headset splits that down the middle, showing A's right eye to your left
eye and B's left eye to your right. That's two different recipes, one per
eye, and what you'd see means nothing. So each preview is first split into
its two eyes, the eyes are labelled and combined, and only then are the two
finished eyes put back together.

The labels are drawn into each eye at exactly the same position and fully
opaque. A label in a different place in each eye would float in front of the
picture, and a see-through one would flicker.

</details>

Comparisons don't wait behind conversions; they have a worker of their own.
A comparison is only two or three minutes of video encoding, mostly on a
different part of the chip, and one that arrives after a four-hour job is no
use to anyone.

The file is saved next to preview A, as
`Previews/<video>/<video> - Fast vs Standard (stacked)_LRF.mp4`, or with
`(A-B-A)` for the other layout.

## The waiting line

The waiting line holds videos you've picked but don't want converted yet.
Nothing in it ever starts by itself: it's a separate list the queue never
looks at.

Each row shows the video's length and size, its settings, and what those
settings would cost for that file, worked out the same way as in the queue.
The settings are a level (Standard unless you change it) plus the flow and 4K
switches where the level has them. They're saved the moment you change them,
so there's no Save button, and they're the same after a reload or on another
device.

Select one or more rows, then:

- **Queue** starts one full conversion per video with that row's settings.
  The videos then leave the waiting line.
- **Preview** makes a one-minute preview per video with that row's settings,
  on the minute picked automatically. The row shows how it's getting on and,
  when it's done, the file name. If the same preview is already queued or
  running, it isn't added a second time.
- **Compare** makes two previews of the same minute and a comparison of the
  two. Each row shows what it will compare, under its level, before you
  click:
  - flow or 4K switched on: the same level without them (A) against your
    settings (B), for example *Standard vs Standard +flow*
  - Standard or Economical without switches: *Fast vs* that level
  - Fast: *Fast vs Standard*
- **Remove** takes videos off the list. Nothing on disk is touched.

Preview and Compare leave the video in the waiting line. Compare reuses
previews you already have for the same file and settings, with one
exception: pipeline previews whose recipe code lacks `Kante` (`_G2EMA_`,
`_G2EMAFluss_`, …). Those were made before the pipeline's edge fix became
compulsory, so they're a different recipe under the same name. They're never
reused, and wherever they're listed they're marked `(no edge fix)`. Fast
previews aren't affected.

### From a script

A client only ever names a file and, if it likes, a level and the two
switches. The recipes themselves belong to iw3-webui.

```sh
# add one or more videos (title and an outside id are optional, shown in the list)
curl -X POST localhost:8790/api/waiting -H 'Content-Type: application/json' \
     -d '{"items":[{"input_path":"films/Holiday.mp4","title":"Holiday"}],"source":"cove"}'

# every row with its settings, what Compare would make, and its preview or comparison
curl localhost:8790/api/waiting

# change a row's settings (fields you leave out keep their value; Fast turns the switches off)
curl -X PATCH localhost:8790/api/waiting/<id> -H 'Content-Type: application/json' \
     -d '{"quality":"standard","flow":true}'

# queue with the row's settings (or give quality/flow/upscale per item to override)
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

`preview` and `compare` also accept
`"items":[{"id":…,"quality":…,"flow":…,"upscale":…}]`. Those settings are
saved before anything else happens, which is how the page makes sure a switch
you flipped a moment ago is the one that gets used.

Adding a video that's already waiting keeps the existing row. Every call
answers item by item (`added`, `already waiting`, `queued`, `previewing`,
`already previewing`, `comparing`, `already comparing`, or `rejected` with
the reason), so one bad path doesn't spoil a whole batch.

## Time estimates

Every queued job shows an estimated time, and the top of the queue shows the
total. Estimates start with a grey `~`. Once a job is running and iw3 reports
its own countdown, that's shown in plain text instead. The two are never
mixed.

What counts is the number of frames a job processes, not just how long the
video is: a 50 fps video takes roughly twice as long as a 25 fps one of the
same length. A single iw3 run processes the duration × `min(source fps,
max_fps)`; the pipeline processes every frame of the source.

- **Fast, Economical and Standard** are priced from timings measured on the
  machine this was built on, an Intel Arc Pro B60. Fast uses 218 finished
  jobs, by picture size. The pipeline is priced stage by stage, each stage a
  start-up time plus a time per frame that grows with the picture size.
  Those timings were taken at 1080p and 1440p; beyond that range, and for
  Economical as a whole, the figures are extrapolated and the page says so.
  They aren't adjusted to your hardware, so on a different GPU treat them as
  a rough guide.
- **Custom jobs** learn from your own machine. Their speed is the median of
  your finished jobs with the same depth model and resolution class (4K means
  4 megapixels or more). Until a combination has run once, a figure measured
  on the Arc Pro B60 stands in.

To see the figures in use:

```sh
curl -s localhost:8790/api/throughput
```

## Configuration

All settings are environment variables on the container.

| Variable | Default | What it does |
|---|---|---|
| `WEBUI_PORT` | `8790` | Port inside the container |
| `PUID` / `PGID` | `99` / `100` | User and group the app runs as; they own the files it creates |
| `UMASK` | `000` | Permission mask for new files |
| `IW3_GPU` | `auto` | `auto` finds the GPU; `0` or `1` picks a card; `-1` forces the CPU |
| `PREVIEW_SECONDS` | `60` | Length of a preview |
| `PREVIEW_LEAD_FRAMES` | `200` | Extra frames converted before the preview's minute, then cut off (see [Previews](#previews)) |
| `PREVIEW_TAIL_FRAMES` | `30` | The same, after the minute |
| `IW3_CHAIN_SCRIPT` | `/opt/iw3-chain/run_chain.sh` | The [pipeline](#the-multi-stage-pipeline) script for Economical and Standard. If it isn't there, those two levels are switched off. |
| `IW3_CHAIN_WORK` | `/output/_chain` | Scratch folders, one per job |
| `IW3_PREVIEWS` | `/output/Previews` | Where previews and comparisons are saved |
| `SCRATCH_KEEP_FAILED_HOURS` | `48` | How long a failed job's scratch folder is kept |
| `SEARCH_TTL_SEC` | `1800` | How old the file-name search index may get before it's rebuilt |
| `FFMPEG_BIN` | `ffmpeg` | ffmpeg for skimming videos and cutting preview clips |
| `FFPROBE_BIN` | `ffprobe` | ffprobe for finding keyframes and checking finished files |
| `QSV_FFMPEG` | `/usr/lib/jellyfin-ffmpeg/ffmpeg` | An ffmpeg with Intel QSV, used to trim finished previews and to make comparisons. It isn't in the image; without it those use `libx265`, and the startup log says so. |
| `QSV_DEVICE` | `/dev/dri/renderD128` | Intel render device for that encoder |
| `QSV_GLOBAL_QUALITY` | `17` | Quality for `hevc_qsv` (lower is better) |
| `OUTPUT_UID` / `OUTPUT_GID` | `99` / `100` | Owner given to finished files |
| `NUNIF_HOME` | `/config` | Models, queue database and logs |

Only one conversion runs at a time, because iw3 isn't built to share a GPU
between conversions. Comparisons are the exception: they run alongside, on
their own worker.

## GPU support

The conversion itself has no vendor-specific code. nunif picks the backend on
its own (`cuda`, then `mps`, then `xpu`; see `nunif/device.py`), so the images
differ only in which PyTorch they ship. [`container/BUILD.md`](container/BUILD.md)
has the details and the build commands.

Two parts are Intel-specific. iw3-webui's own short encodes prefer Intel's
`hevc_qsv` and use `libx265` without it, and the server pipeline in
`chain/server` is written for Intel Arc. `chain/desktop` is its NVIDIA
counterpart.

What has actually been tested:

| GPU | Image builds | Conversion |
|---|---|---|
| Intel Arc (XPU) | ✅ in CI | ✅ in daily use on an Arc Pro B60 |
| NVIDIA (CUDA) | ✅ in CI | ✅ in daily use on an RTX 5080, with the pipeline in `chain/desktop` running directly on Windows. The `:cuda` image itself hasn't been run there yet. |
| CPU only | ✅ in CI | ❓ not tried |
| AMD (ROCm) | ❓ not in CI | ❓ not tried |

Every push builds the three CI images and checks that torch and iw3 load
inside them, so a broken Dockerfile is caught even without the hardware.
Whether conversions come out right on ROCm, or inside the `:cuda` image, is
still an open question. If you try one, a report or a pull request updating
this table would be very welcome.

## Security

**There's no login.** Anyone who can reach the port can queue jobs, read logs
and browse the folders under `/input`. It's meant for a home network: don't
expose it to the internet, or put it behind a reverse proxy that asks for a
password.

## Credits and license

All of the actual 3D conversion is done by
[nagadomi/nunif](https://github.com/nagadomi/nunif). Fast and Custom run plain
`python -m iw3` and change nothing about how it converts. Economical and
Standard combine iw3's own models and warp in the
[pipeline](chain/README.md), and run the warp with one upstream nunif fix
(40d46847) that the pinned version lacks; see `iw3_kantenfix.py`.

iw3-webui is MIT-licensed; see [LICENSE](LICENSE). nunif is MIT as well.
Several depth model files are CC-BY-NC-4.0; they're neither included here nor
downloaded automatically.
