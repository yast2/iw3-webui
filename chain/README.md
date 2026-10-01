# The multi-stage pipeline

The web UI's **Economical** and **Standard** levels do not run one `iw3`
process. They run this chain: five to seven programs per job, each writing
into a scratch folder, each restartable on its own. The web UI execs the
chain's driver script and reads its stage markers (see
[the contract](../README.md#the-multi-stage-pipeline)); everything below that
line lives here.

There are two drivers for the same recipe, because it runs on two machines:

| Folder | Driver | Runs on | Encoder |
|---|---|---|---|
| `server/` | `run_voll.sh` (POSIX sh) | Linux, inside the iw3-webui container, Intel Arc | `hevc_qsv` via jellyfin-ffmpeg |
| `desktop/` | `run_chain.py` (Python) | Windows or Linux, NVIDIA | `hevc_nvenc` on Windows, `hevc_qsv` on Linux; `--encoder` overrides (`libx265` too) |

`run_voll.sh` is the reference. `run_chain.py` is a port of it with the same
stages, calls and parameters; only the way they are started differs (argument
lists instead of a shell, interpreter and nunif location from the environment,
the whole process tree of a stage taken down on cancel). The desktop driver has
since picked up a few things the server has not yet: per-frame resume of the
DepthPro stage, sources with a rotation matrix, and a frame-mapping check that
handles dropped as well as duplicated frames.

## The recipe ("G2EMA")

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

Fast is the single-process `iw3` call and needs none of this.

Every stage leaves an `.ok` marker in the work folder, so a restart only redoes
what is missing; wall-clock time per stage goes to `zeiten.tsv` there.

## The files

| File | What it does |
|---|---|
| `run_voll.sh` / `run_chain.py` | The driver: options, stage order, markers, cleanup. |
| `bildrate_pruefen.py` | Says whether the source must be re-timed to a constant frame rate before the chain (phones report e.g. 120 fps for a 30 fps clip). Silent in the normal case. |
| `umnummerieren.py` | Hard-link copy of a frame folder under the source's numbering; VDA restarts its numbering at 1 for a clip. |
| `vda_abgleich.py` | Maps VDA's depth maps to source frames when iw3's fps filter duplicated (or, on the desktop, dropped) a frame - by decoding the source exactly as iw3 does and hashing frames, not by guessing. Refuses rather than shifting thousands of maps. |
| `vda_ende_pruefen.py` | The older, narrower check it replaces: drops one surplus VDA map only if it provably sits at the end. |
| `stufe2_band_schnell.py` | Band swap: DepthPro's fine band on VDA's coarse band. |
| `stufe3_fluss.py` | Optional temporal smoothing along optical flow (torchvision RAFT). |
| `build_export_full.py` | Builds the iw3 export folder (`iw3_export.yml` + depth PNGs) the warp imports. |
| `iw3_kantenfix.py` | Runs `python -m iw3` with upstream nunif's edge-dilation fix (40d46847) swapped in: the pinned nunif (d23721f1) inverts edge dilation when importing an export. Compulsory; the chain stops rather than warp without it. |
| `iw3_accel.py` | Desktop only. DepthPro through a TensorRT engine or `torch.compile` when `IW3_COMPILE` is set; eager otherwise. |
| `requirements-win.txt` | Desktop only. The server's `pip freeze` minus Intel/XPU, with torch 2.11.0+cu128. |

The comments in these scripts are in German and record the measurement behind
each choice. Worked examples from real runs have been anonymised.

## Wiring it up

**Server.** Copy `server/` somewhere under the config volume and point the web
UI at the driver:

```
IW3_CHAIN_SCRIPT=/config/chain/run_voll.sh
IW3_CHAIN_WORK=/output/_chain
```

The helpers are found next to the driver (`BIN`, default: the driver's own
folder). nunif is expected at `/opt/nunif`, as in the image. The encode step
uses jellyfin-ffmpeg for QSV, which the Dockerfile in this repository does not
install; the image this runs in adds it as one extra layer:

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

Elsewhere, set `FFMPEG_QSV`, `FFPROBE_QSV` and `QSV_GERAET` (the render node).

**Desktop.** Put `desktop/` in a folder `chain` next to a nunif checkout at the
same pin (`..\nunif`, or set `NUNIF_DIR`), with a Python that has nunif's
requirements and a CUDA torch (`requirements-win.txt`). Then:

```
python chain\run_chain.py --level standard -i <video> -o <out dir> --work <scratch dir> --gpu 0
```

Nothing in this repository sends jobs to the desktop. Here a small worker on
the server takes queued jobs, copies the source over SSH, runs `run_chain.py`
and copies the result back; it is specific to this setup and not included.

## Options both drivers take

```
--level <fast|economical|standard>  -i <file>  -o <dir>  --work <dir>  --gpu <n>
[--stereo-format <name>]  [--flow]  [--upscale]  [-d=<divergence>]
run_chain.py only: [--encoder hevc_nvenc|hevc_qsv|libx265]
```

Exit status is non-zero on failure. Each stage is announced as
`=== i/n <name>  <timestamp>`, which is what the web UI's progress reads.
