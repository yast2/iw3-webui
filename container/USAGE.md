# Converting a file by hand

Normally you'll use the queue. This page covers the other way: running iw3
directly inside the container with `docker exec`. It's handy for one-off
conversions and for trying options the web interface doesn't offer.

Use the same PUID and PGID you started the container with. The examples below
use 99 and 100.

```
docker exec -u 99:100 iw3 python3 -m iw3 \
  -i /input/<path-to-file> \
  -o /output \
  --depth-model VDA_L \
  --divergence 2.0 \
  --convergence 0.5 \
  --edge-dilation 2 \
  --scene-detect \
  --ema-normalize \
  --video-codec libx265 \
  --gpu 0 \
  -y
```

The result is saved as `/output/<original-filename>_LRF_Full_SBS.mp4`. Quest
players use that ending to recognise the 3D format automatically.

## What each option does

| Option | Value | Why |
|---|---|---|
| `-u 99:100` | (an option of `docker exec`, not iw3) | **Don't leave it out.** `docker exec` skips the container's usual switch to PUID/PGID and runs as root. Without it, the output (and any newly downloaded model files) ends up owned by root, which can quietly break later runs. |
| `-i` | a path under `/input` | The source file. `/input` is your video share, mounted read-only. |
| `-o` | `/output` (a **folder**, never a file name) | iw3 only names the result `<original>_LRF_Full_SBS.mp4` when `-o` is a folder. Give it a file name and you lose the ending Quest players rely on. |
| `--depth-model VDA_L` | Video-Depth-Anything Large | Keeps depth consistent from frame to frame. Single-frame models (such as `Any_V2_L`) guess the depth afresh for every frame, which flickers on video. The queue uses `VDA_B` by default: roughly twice as fast, with only a small loss in quality. If a scene looks flat or oddly curved, try `VDA_Metric_L`, which estimates real-world distances rather than relative ones. |
| `--divergence 2.0` | iw3's default | How strong the 3D effect is (the simulated distance between your eyes). Higher means more depth, and more eye strain. Try 1.0–1.5 for something gentler. |
| `--convergence 0.5` | iw3's default | Where the screen sits in depth. 0.5 brings part of the scene out in front of it; 0 keeps everything behind it. |
| `--edge-dilation 2` | iw3's default | Widens the edges of foreground objects before the warp, to hide the gap that opens up behind them. |
| `--scene-detect` | on | Starts the depth estimate afresh at every hard cut, rather than carrying it across. Without it, VDA can let depth leak from one scene into the next. This matters most for films. |
| `--ema-normalize` | on | Smooths the depth scale over time. VDA's overall scale wobbles slightly from frame to frame; this removes the resulting flicker. nunif's own docs recommend it with any VDA model. |
| `--video-codec libx265` | software HEVC | The encode always runs on the CPU: iw3 encodes through PyAV, not a system ffmpeg, so it has no VAAPI or QSV path. The depth estimation is what runs on the GPU. |
| `--gpu 0` | device number | `-1` uses the CPU, `1` a second card. The queue passes whatever `IW3_GPU` is set to. |
| `-y` | overwrite without asking | Needed whenever nobody is there to answer. |

## A whole folder at once

`-i` also takes a folder. With `--recursive` iw3 works through all of it,
one file at a time (iw3 can't share a GPU between conversions anyway):

```
docker exec -u 99:100 iw3 python3 -m iw3 \
  -i /input/<folder> -o /output --recursive --skip-error \
  --depth-model VDA_L --divergence 2.0 --convergence 0.5 --edge-dilation 2 \
  --scene-detect --ema-normalize --video-codec libx265 --gpu 0 -y
```

`--skip-error` stops one bad file from ending the whole batch.
