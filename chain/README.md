# The multi-stage pipeline

The **Economical** and **Standard** levels don't run iw3 just once. They run
this chain: five to seven programs per job, each writing into a scratch folder
and each able to restart on its own. The web interface starts the chain's
driver script and follows the stage markers it prints (see
[how the two connect](../README.md#the-multi-stage-pipeline)). Everything
after that happens here.

There are two drivers for the same recipe, because it runs on two machines:

| Folder | Driver | Runs on | Video encoder |
|---|---|---|---|
| `server/` | `run_voll.sh` (POSIX shell) | Linux, inside the iw3-webui container, Intel Arc | `hevc_qsv` via jellyfin-ffmpeg |
| `desktop/` | `run_chain.py` (Python) | Windows or Linux, NVIDIA | `hevc_nvenc` on Windows, `hevc_qsv` on Linux; `--encoder` picks another, `libx265` included |

`run_voll.sh` is the original. `run_chain.py` is a port of it, with the same
stages, commands and settings. Only the way it starts them differs: argument
lists instead of a shell, the Python and nunif locations taken from the
environment, and cancelling a stage takes down its whole process tree.

The desktop version has since gained a few things the server hasn't yet:

- DepthPro resumes frame by frame after an interruption,
- videos with a rotation flag are handled, and
- the frame-matching check copes with dropped frames as well as doubled ones.

## The recipe

Internally it's called "G2EMA":

```
source -> frames/ (JPEG, ffmpeg; constant frame rate first if the source lies about it)
       -> [optional, off] waifu2x noise_scale2x                --upscale
       -> DepthPro (Standard) / DepthPro_S (Economical)       fine band: outlines, detail
       -> VDA_B with --ema-normalize                           coarse band: calm over time
       -> band swap, r=30, linear (CPU)                        stufe2_band_schnell.py
       -> [optional, off] optical flow smoothing, RAFT fp16    stufe3_fluss.py, --flow
       -> iw3 export folder (mapper mul_1 + mul_2 = 0.5)       build_export_full.py
       -> warp: mlbw_l2, convergence 0.5, sod_v1               iw3 via iw3_kantenfix.py
          (writes a lossless x264 crf 0 intermediate)
       -> encode: HEVC on the GPU's hardware encoder
```

In short: DepthPro is good at sharp outlines and fine detail, VDA is good at
staying calm from one frame to the next, so the chain takes the fine detail
from one and the overall shape from the other.

Fast is a single iw3 call and needs none of this.

Each finished stage leaves an `.ok` marker in the work folder, so a restart
only redoes what's missing. How long each stage took is written to
`zeiten.tsv` in the same folder.

## The files

| File | What it does |
|---|---|
| `run_voll.sh` / `run_chain.py` | The driver: options, stage order, markers and cleanup. |
| `bildrate_pruefen.py` | Checks whether the source needs re-timing to a constant frame rate before the chain starts. Phones, for example, can report 120 fps for a 30 fps clip. Says nothing when all is well. |
| `umnummerieren.py` | Makes a hard-linked copy of a frame folder using the source's numbering, because VDA starts counting at 1 for every clip. |
| `vda_abgleich.py` | Matches VDA's depth maps to the source frames when iw3's frame-rate filter has doubled a frame (or, on the desktop, dropped one). It works this out by decoding the source exactly as iw3 does and comparing frame checksums, not by guessing, and it would rather stop than shift thousands of maps. |
| `vda_ende_pruefen.py` | The older, narrower check that `vda_abgleich.py` replaces: it drops one extra VDA map only if it's provably at the very end. |
| `stufe2_band_schnell.py` | The band swap: DepthPro's fine detail on top of VDA's overall shape. |
| `stufe3_fluss.py` | Optional smoothing over time along the optical flow (torchvision RAFT). |
| `build_export_full.py` | Builds the iw3 export folder (`iw3_export.yml` plus depth PNGs) that the warp reads. |
| `iw3_kantenfix.py` | Runs `python -m iw3` with upstream nunif's edge-dilation fix (40d46847) swapped in. The pinned nunif (d23721f1) turns edge dilation inside out when it imports an export. This isn't optional: the chain stops rather than warp without it. |
| `iw3_accel.py` | Desktop only. Runs DepthPro through a TensorRT engine or `torch.compile` when `IW3_COMPILE` is set, and in plain (eager) PyTorch otherwise. |
| `requirements-win.txt` | Desktop only. The server's `pip freeze` minus the Intel/XPU packages, with torch 2.11.0+cu128. |

The comments inside these scripts are in German, and they record the
measurement behind each choice. Examples taken from real runs have been
anonymised.

## Setting it up

**Server.** Copy `server/` somewhere under the config volume and point the
web interface at the driver:

```
IW3_CHAIN_SCRIPT=/config/chain/run_voll.sh
IW3_CHAIN_WORK=/output/_chain
```

The helper scripts are expected next to the driver (`BIN`, which defaults to
the driver's own folder), and nunif at `/opt/nunif`, as in the image. The
encode step uses jellyfin-ffmpeg for Intel QSV. The Dockerfile in this
repository doesn't install it, so the image this runs in adds it as one extra
layer:

```dockerfile
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gnupg ca-certificates \
 && install -d /etc/apt/keyrings \
 && curl -fsSL https://repo.jellyfin.org/jellyfin_team.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/jellyfin.gpg \
 && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/jellyfin.gpg] https://repo.jellyfin.org/ubuntu noble main" \
      > /etc/apt/sources.list.d/jellyfin.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends jellyfin-ffmpeg7 \
 && rm -rf /var/lib/apt/lists/*
```

If your ffmpeg lives somewhere else, set `FFMPEG_QSV`, `FFPROBE_QSV` and
`QSV_GERAET` (the render device).

**Desktop.** Put `desktop/` in a folder called `chain`, next to a nunif
checkout at the same commit (`..\nunif`, or set `NUNIF_DIR`). You'll also
need a Python with nunif's requirements and a CUDA build of torch (see
`requirements-win.txt`). Then:

```
python chain\run_chain.py --level standard -i <video> -o <out dir> --work <scratch dir> --gpu 0
```

Nothing in this repository sends jobs to the desktop. Here, a small worker
on the server picks up queued jobs, copies the source over SSH, runs
`run_chain.py` and copies the result back. It's specific to this setup, so
it isn't included.

## Options

Both drivers accept:

```
--level <fast|economical|standard>  -i <file>  -o <dir>  --work <dir>  --gpu <n>
[--stereo-format <name>]  [--flow]  [--upscale]  [-d=<divergence>]
```

`run_chain.py` also takes `[--encoder hevc_nvenc|hevc_qsv|libx265]`.

Both exit with a non-zero status on failure, and announce each stage as
`=== i/n <name>  <timestamp>`, which is what the web interface's progress
display reads.
