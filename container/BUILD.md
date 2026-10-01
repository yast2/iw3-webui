# Building for your GPU

You probably don't need this page: ready-made images exist for NVIDIA, Intel
Arc and CPU only (see the [README](../README.md#getting-started)). Build your
own if you want a different torch, a different nunif version, or a GPU there's
no image for.

## The short version

The conversion has no vendor-specific code. nunif picks the backend itself
(`cuda`, then `mps`, then `xpu`; see `nunif/device.py`) for any device number
of 0 or above. So the only real difference between builds is which base image
supplies which torch. That's what the three build arguments below set, and
there's nothing else to change.

## The build arguments

| GPU | `BASE_IMAGE` | `TORCH_INSTALL` | `VENV_PATH` |
|---|---|---|---|
| Intel (XPU) | `intel/pytorch:xpu-2.11.0-ubuntu24.04` | *(empty)* | `/opt/venv` |
| NVIDIA (CUDA) | `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime` | *(empty)* | `/opt/conda` |
| AMD (ROCm) | `rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0` | *(empty)* | `/opt/conda/envs/py_3.12` |
| CPU only | `python:3.12-slim` | `--index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1` | `/usr/local` |

Intel and NVIDIA are the easy ones: their base images already ship a
matching torch, so nothing is installed on top. For NVIDIA it's an exact
match: that image carries `torch 2.7.1+cu126`, which is precisely what nunif's
own `requirements-torch-cu126.txt` asks for.

### Getting `VENV_PATH` right

`VENV_PATH` is the one argument that's easy to get wrong, and it matters. The
build installs `python3-dev`, which brings a second Python along with it, in
`/usr/bin`. If that one ends up first in line, pip installs into one Python
and the container runs the other, and you only find out when something tries
to import torch.

That's exactly what happened in this project's first CI run.
`python:3.12-slim` keeps its Python in `/usr/local/bin`, `VENV_PATH=/usr` put
Debian's in front of it, and the build died with
`ModuleNotFoundError: No module named 'torch'`.

So set `VENV_PATH` to the folder whose `bin/` holds the Python that has torch,
and check rather than assume. For any base image:

```sh
docker run --rm <base image> sh -c 'command -v python3; python3 -c "import torch, sys; print(sys.executable, torch.__version__)"'
```

## Build commands

NVIDIA:

```sh
docker build -t iw3-webui:cuda \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime \
  --build-arg VENV_PATH=/opt/conda \
  container/
```

Intel (the defaults):

```sh
docker build -t iw3-webui:xpu container/
```

CPU only:

```sh
docker build -t iw3-webui:cpu \
  --build-arg BASE_IMAGE=python:3.12-slim \
  --build-arg VENV_PATH=/usr/local \
  --build-arg TORCH_INSTALL="--index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1" \
  container/
```

A note on the CPU build: despite its name, nunif's `requirements-torch.txt`
isn't a CPU file. On Linux it asks for `torch==2.7.1+cu128`, a CUDA build,
with the CPU index merely listed alongside. That's why the command above
gives the index explicitly.

## Running it

The container works out which device to use when it starts, so there's
nothing to configure. It can only see a GPU you actually hand to it, though:

| GPU | Run flag |
|---|---|
| NVIDIA | `--gpus all` |
| Intel | `--device /dev/dri:/dev/dri:rwm` |
| AMD | `--device /dev/kfd --device /dev/dri` |
| CPU | *(nothing)* |

Forgetting this is the most common mistake, and it doesn't fail. It just
runs on the CPU at a fraction of the speed. So the container says so loudly:
in the startup log, in a banner at the top of the page, and as a `warning` in
`GET /api/health`. If things seem slow, check that first.

To choose the device yourself instead, set `IW3_GPU`: `0` for the first GPU,
`1` for the second, `-1` to force the CPU.

## Choosing the nunif version

`NUNIF_REF` sets which nunif commit is built in. It's fixed rather than
following the latest version, so that rebuilding to pick up a change here
doesn't quietly upgrade the converter too, and with it how your output looks.

```sh
docker build --build-arg NUNIF_REF=<sha> container/
```

## What has been tested

| GPU | Image builds | Conversion |
|---|---|---|
| Intel XPU | ✅ in CI | ✅ in daily use on an Arc Pro B60 |
| NVIDIA CUDA | ✅ in CI | ✅ in daily use on an RTX 5080, with the pipeline in `chain/desktop` running directly on Windows. The `:cuda` image itself hasn't been run there yet. |
| CPU | ✅ in CI | ❓ not tried |
| AMD ROCm | ❓ not in CI (the base image is too big for a standard runner) | ❓ not tried |

The ROCm `VENV_PATH` was read from that image's published configuration
(`/opt/conda/envs/py_3.12/bin` comes first on its own `PATH`) rather than
guessed, but nothing beyond that has been checked.

CI builds every image above except ROCm on each push, so a broken Dockerfile
is caught without any hardware. Whether conversions come out right on ROCm,
or inside the `:cuda` image, is a separate question that can't be answered
from here. Reports and pull requests are very welcome.
