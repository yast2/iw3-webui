# Add to iw3 Queue: a Cove extension

This puts one button on [Cove](https://github.com/coveapp/cove)'s video page.
Click it, and that video goes into the [iw3 queue](../container), which can
be running anywhere, and Cove shows you the new job's ID.

Nothing is converted inside Cove. The extension looks up the video's file,
translates its path into the one iw3 sees, and sends it to iw3's `/api/jobs`.
From then on the job lives in iw3's queue and you follow it in iw3's own
interface; it doesn't appear in Cove's job list.

Cove and iw3 don't need to share a Docker network, or even a machine. The
extension simply talks to whatever address `IW3_WEBUI_URL` gives it.

## Building it

First fetch Cove's reference assemblies (see
[refs/README.md](refs/README.md)), then:

```sh
docker run --rm -v "$PWD":/work -w /work/src/Iw3Queue \
  mcr.microsoft.com/dotnet/sdk:10.0 \
  bash -c "dotnet build -c Release -o /work/out"
```

## Installing it

Copy three files into a folder of their own in Cove's extensions folder, then
restart Cove:

```sh
mkdir -p /path/to/cove/config/extensions/com.yast2.iw3-queue
cp out/Iw3Queue.dll out/Iw3Queue.deps.json \
   src/Iw3Queue/extension.json \
   /path/to/cove/config/extensions/com.yast2.iw3-queue/
docker restart Cove
```

There's no `frontend/` folder and no JavaScript. A button whose action has no
`handlerName` is sent by Cove straight to the server, so a button that makes
one server call needs no browser code at all.

When it loads, Cove logs `iw3 Queue <version> initialised, target <url>,
media root <root>, depth model <model>`. Check the media root and the model
in that line: it's the cheapest way to catch a mistake before it costs you
hours of GPU time.

## Settings

These are environment variables on the **Cove** container.

| Variable | Default | What it does |
|---|---|---|
| `IW3_WEBUI_URL` | `http://iw3:8790` | Where the iw3 queue can be reached |
| `IW3_QUEUE_MEDIA_ROOT` | `/media/` | The part of Cove's file path to remove; see below |
| `IW3_QUEUE_STEREO_FORMAT` | `full_sbs` | Any 3D format iw3's interface offers |
| `IW3_QUEUE_PARAMS` | see below | A JSON object with the same keys as iw3's own settings form |

### The media root

Cove and iw3 usually see the same files under different paths.
`IW3_QUEUE_MEDIA_ROOT` is Cove's half of that: it's removed from the front of
Cove's path, and whatever is left is the path inside iw3's `/input`.

With Cove mounting `/mnt/user:/media` and iw3 mounting `/mnt/user/videos:/input`:

```
Cove path    /media/videos/films/example.mkv
remove       /media/videos/
sent to iw3  films/example.mkv
```

This doubles as a check. A video outside that folder can't be reached by iw3
at all, so the click is turned down straight away with a `422` and a message
naming both paths, rather than failing later somewhere less obvious.

### The conversion settings

The defaults are chosen with care:

```json
{
  "depth_model": "VDA_B", "divergence": 2.0, "convergence": 0.5,
  "foreground_scale": 0, "edge_dilation": [2, 1],
  "video_codec": "libx265", "pix_fmt": "yuv420p", "max_fps": 1000,
  "scene_detect": true, "ema_normalize": true,
  "ema_decay": 0.75, "ema_buffer": 30
}
```

**Don't replace them with an empty object to "use iw3's defaults".** The
web form shows values filled in from iw3's own option list, but the queue
only passes on the settings a job actually *contains*. An empty object therefore
doesn't mean "the form's defaults"; it means iw3's bare command-line defaults.
Those use `ZoeD_Any_N`, the oldest single-frame depth model there is, with
scene detection and smoothing switched off. That slip once ran 18 jobs and 14
GPU hours on the wrong model before anyone noticed, because the results
looked perfectly fine, just worse.

If `IW3_QUEUE_PARAMS` is empty or isn't valid JSON, the extension logs a
warning and uses the defaults above rather than sending nothing.

`max_fps: 1000` means "no limit". iw3 uses `min(source fps, max_fps)` and has
no special value for unlimited; its own desktop app goes up to 1000. Leave it
at the default of 30 and every 50 or 60 fps video quietly loses half its
frames. Keep it at 15 or more: below that, iw3 silently switches
`ema_normalize` off.

## What the button doesn't do

- **No file picker.** It uses the file Cove already treats as the main one
  for that video (`MaxPath`).
- **No settings dialog.** It's a quick add. Set the defaults once, as above.
- **No progress in Cove.** The pop-up shows the iw3 job ID; you follow the
  progress in iw3's interface.

## Permissions

Using the button requires Cove's `jobs.run` permission, the same one Cove's
own `/api/metadata/generate` needs to start work.

At startup Cove logs a warning that the extension's endpoint is registered
without an authorization policy. That's expected. The permission is checked
inside the handler itself, the same way Cove's bundled extensions do it. The
warning is about the missing declaration, not about the endpoint being open.
