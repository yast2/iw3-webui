#!/usr/bin/env python3
"""iw3_accel.py -- DepthPro kompiliert statt eager, schaltbar ueber IW3_COMPILE.

Der "Owl3D-Trick": das Tiefenmodell wird fuer genau diese Grafikkarte uebersetzt
(TensorRT-Engine, wie Owl3Ds model*_trt_5090.enc) statt in eager PyTorch zu laufen.
nunifs Dateien bleiben unberuehrt: dieses Skript startet iw3 selbst (wie
`python -m iw3`) und tauscht vorher DepthProModel.load_model aus.

    python iw3_accel.py <iw3-Argumente ...>        # statt python -m iw3 ...
    python iw3_accel.py --accel-bauen [DepthPro]   # Engine vorab bauen, sonst beim ersten Lauf

IW3_COMPILE (Vorgabe: nicht gesetzt = eager, run_chain.py ruft dann gar nicht hierher)
    eager          unveraendert, nur mit Zeitmessung (Vergleichslaeufe)
    trt            TensorRT-Engine fp16, fester Eingang 1x3x1536x1536
    inductor       torch.compile, Modus default (nur ~6 % schneller, 57 s Kompilierung kalt)
    inductor-max   GESPERRT: max-autotune lieferte am 24.09. konstante Tiefenkarten -> laeuft eager

Engines liegen unter $NUNIF_HOME/engines/ (bzw. IW3_ENGINE_DIR), der Grafikkartenname
steht im Dateinamen (…_rtx5080_sm120_trt11.3.engine). Fehlt die Datei, passen Karte,
TensorRT-Version oder Treiber nicht zur .json daneben, oder stimmt die sha256 nicht, wird neu
gebaut (RTX 5080: ONNX ~40 s + TensorRT ~120 s, einmalig). Schlaegt TensorRT fehl, laeuft der
Schritt eager weiter und sagt das laut auf stderr ("### accel: TensorRT FEHLGESCHLAGEN").

Stand 24.09.2026 (RTX 5080, 600 Bilder 1080p): eager 0,253 s/Bild, trt 0,130 s/Bild. trt
verfehlt das Pearson-Gate (>= 0,999 je Bild) an 4 von 600 fast flachen Bildern (min 0,9971);
SSIM am verlustfreien Master rechts min 0,99914. Deshalb Vorgabe: eager. Belege:
interne Messnotiz
"""
import atexit
import hashlib
import json
import os
import re
import runpy
import shutil
import sys
import time

T_START = time.time()
MODUS = (os.environ.get("IW3_COMPILE") or "eager").strip().lower()
MODI = ("eager", "trt", "inductor", "inductor-max")
if MODUS.startswith("inductor"):
    # Vor dem ersten `import torch` setzen -- torch belegt TORCHINDUCTOR_CACHE_DIR sonst selbst
    # mit %TEMP%\torchinductor_<user> (gemessen 24.09. 01:04). Beide Caches in einen Ordner:
    # wer kalt messen will, schiebt genau diesen beiseite (ein Kompilier-Cache ueberlebt den
    # Prozess, interne Messnotiz).
    _home = os.environ.get("NUNIF_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    _wurzel = os.path.abspath(os.environ.get("IW3_INDUCTOR_DIR") or os.path.join(_home, "inductor"))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _wurzel
    os.environ["TRITON_CACHE_DIR"] = os.path.join(_wurzel, "triton")
# Engine-Variante: fp16 durchgehend, ausser was hier auf fp32 steht (ln32 = LayerNorm,
# sm32 = Softmax der Attention). Steht im Dateinamen der Engine.
VARIANTE = os.environ.get("IW3_TRT_VARIANTE") or "fp16"
# ONNX-Exportweg: torchscript (erprobt) oder dynamo (torch.export, faellt auf torchscript zurueck)
EXPORTWEG = os.environ.get("IW3_ONNX_WEG") or "torchscript"


def meldung(text):
    sys.stderr.write(f"### accel: {text}\n")
    sys.stderr.flush()


# ----------------------------------------------------------------- Pfade
def engine_dir():
    d = os.environ.get("IW3_ENGINE_DIR")
    if not d:
        home = os.environ.get("NUNIF_HOME") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        d = os.path.join(home, "engines")
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def gpu_kennung(torch, device):
    """'NVIDIA GeForce RTX 5080' -> 'rtx5080_sm120' (wie Owl3Ds _trt_5090)."""
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    m = re.search(r"(RTX|GTX|TITAN|A|L|H|B)\s*([0-9]{3,4}\s*(?:Ti|SUPER|Super)?(?:\s*Ada)?)", name)
    kurz = (m.group(1) + m.group(2)) if m else name
    kurz = re.sub(r"[^0-9a-z]+", "", kurz.lower())
    return f"{kurz}_sm{major}{minor}", name


# ------------------------------------------------------------ Zeitmessung
class Messung:
    """Zaehlt die Vorwaertslaeufe des Tiefenmodells: Einrichtung (Start bis erstes Bild)
    und Durchsatz (erstes bis letztes Bild) getrennt. IW3_ACCEL_SYNC=1 misst zusaetzlich
    die reine Modellzeit mit CUDA-Synchronisation (bremst die Ueberlappung, nur Diagnose)."""

    def __init__(self):
        self.n = 0
        self.t_erst = None
        self.t_erst_ende = None
        self.t_letzt = None
        self.t_bereit = None
        self.rein = 0.0
        self.sync = os.environ.get("IW3_ACCEL_SYNC") == "1"
        self.bau_s = 0.0
        self.info = ""
        atexit.register(self.bericht)

    def bericht(self):
        if self.n == 0:
            return
        # je_bild = Durchsatz ab Bild 2: das erste Bild traegt bei inductor die Kompilierung
        schleife = self.t_letzt - self.t_erst_ende
        je = schleife / max(self.n - 1, 1)
        zeile = (f"modus={MODUS} n={self.n} start_bis_bild1={self.t_erst - T_START:.1f}s "
                 f"davon_bau={self.bau_s:.1f}s bild1={self.t_erst_ende - self.t_erst:.1f}s "
                 f"bild2_bis_ende={schleife:.1f}s je_bild={je:.4f}s gesamt={self.t_letzt - T_START:.1f}s")
        if self.sync:
            zeile += f" rein_modell={self.rein / self.n:.4f}s"
        if self.info:
            zeile += f" {self.info}"
        meldung(zeile)
        ziel = os.environ.get("IW3_ACCEL_BERICHT")
        if ziel:
            with open(ziel, "a", encoding="utf-8", newline="\n") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + zeile + "\n")


MESSUNG = Messung()


class Gemessen:
    """Wickelt ein Modell ein, ohne sein Verhalten zu aendern."""

    def __init__(self, torch, model):
        self.torch = torch
        self.model = model

    def __call__(self, x):
        t = time.time()
        if MESSUNG.t_erst is None:
            MESSUNG.t_erst = t
        if MESSUNG.sync:
            self.torch.cuda.synchronize()
            t = time.time()
        out = self.model(x)
        if MESSUNG.sync:
            self.torch.cuda.synchronize()
            MESSUNG.rein += time.time() - t
        MESSUNG.n += x.shape[0]
        MESSUNG.t_letzt = time.time()
        if MESSUNG.t_erst_ende is None:
            MESSUNG.t_erst_ende = MESSUNG.t_letzt
        return out

    def __getattr__(self, name):
        return getattr(self.model, name)

    # BaseDepthModel.load ruft .to(device).eval() -- die Huelle muss dabei bleiben
    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self

    def eval(self):
        self.model = self.model.eval()
        return self


# ------------------------------------------------------------- TensorRT
class TRTDepthPro:
    """Ersetzt DepthPro.forward(x) -> (canonical_inverse_depth, fov_deg) durch eine Engine.
    Batch 1 fest; groessere Batches (--tta) laufen Bild fuer Bild."""

    def __init__(self, torch, trt, engine_bytes, img_size, device):
        self.torch = torch
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Engine liess sich nicht laden (deserialize_cuda_engine -> None)")
        self.ctx = self.engine.create_execution_context()
        self.device = device
        self.img_size = img_size
        self.metric_depth = False
        self.namen = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.eingang = [n for n in self.namen if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.ausgang = [n for n in self.namen if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        assert len(self.eingang) == 1 and len(self.ausgang) == 2, self.namen
        self.typ = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32}
        # Eigener Strom: auf dem Vorgabestrom synchronisiert TensorRT nach jedem enqueueV3
        # selbst ("Using default stream in enqueueV3() may lead to performance issues").
        self.strom = torch.cuda.Stream(device)

    def _ein(self, x):
        torch = self.torch
        x = x.to(device=self.device, dtype=torch.float16).contiguous()
        self.ctx.set_tensor_address(self.eingang[0], x.data_ptr())
        outs = []
        for n in self.ausgang:
            shape = tuple(self.ctx.get_tensor_shape(n))
            t = torch.empty(shape, dtype=self.typ[self.engine.get_tensor_dtype(n)], device=self.device)
            self.ctx.set_tensor_address(n, t.data_ptr())
            outs.append(t)
        haupt = torch.cuda.current_stream(self.device)
        self.strom.wait_stream(haupt)
        if not self.ctx.execute_async_v3(self.strom.cuda_stream):
            raise RuntimeError("execute_async_v3 fehlgeschlagen")
        # Eingang und Ausgaben gehoeren dem Hauptstrom; der Allokator darf sie erst nach der
        # Engine wiederverwenden.
        for t in [x] + outs:
            t.record_stream(self.strom)
        haupt.wait_stream(self.strom)
        return outs

    def __call__(self, x):
        torch = self.torch
        if x.shape[0] == 1:
            a, b = self._ein(x)
        else:
            teile = [self._ein(x[i:i + 1]) for i in range(x.shape[0])]
            a = torch.cat([t[0] for t in teile]); b = torch.cat([t[1] for t in teile])
        return a, b

    # BaseDepthModel.load ruft .to(device).eval()
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def _onnx_exportieren(torch, eager, pfad, device):
    """DepthPro als ONNX, fp16, Eingang 1x3xSxS. Gibt die Kennung des Exportwegs zurueck."""
    S = eager.img_size

    class Huelle(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            d, fov = self.m(x)
            return d, fov

    x = torch.zeros((1, 3, S, S), dtype=torch.float16, device=device)
    h = Huelle(eager).eval()
    # Wie eager PyTorch (rechnet layer_norm auf fp16 intern in fp32): im Graphen ausdruecklich
    # fp32, sonst rechnet TensorRT im strongly-typed Netz die Normierung in fp16.
    geflickt = []
    if "ln32" in VARIANTE:
        F = torch.nn.functional

        def ln32(self, x):
            return F.layer_norm(x.float(), self.normalized_shape,
                                None if self.weight is None else self.weight.float(),
                                None if self.bias is None else self.bias.float(), self.eps).to(x.dtype)
        for mod in eager.modules():
            if isinstance(mod, torch.nn.LayerNorm):
                mod.forward = ln32.__get__(mod)
                geflickt.append(mod)
    if any(k in VARIANTE for k in ("conv32", "lin32", "pe32", "dec32")):
        F = torch.nn.functional

        def conv32(self, x):
            # Gewichte liegen fuer den Export echt in fp32 (self.float() unten): gecastete
            # fp16-Konstanten vor einer fp32-Dekonvolution findet TensorRT 11.3 keine Taktik.
            w = self.weight; b = self.bias
            if isinstance(self, torch.nn.ConvTranspose2d):
                y = F.conv_transpose2d(x.float(), w, b, self.stride, self.padding, self.output_padding,
                                       self.groups, self.dilation)
            else:
                y = self._conv_forward(x.float(), w, b)
            return y.to(x.dtype)

        def lin32(self, x):
            return F.linear(x.float(), self.weight.float(),
                            None if self.bias is None else self.bias.float()).to(x.dtype)
        for name, mod in eager.named_modules():
            # pe32 = nur die Patch-Einbettungen der drei ViT (16x16-Faltung, 768 Summanden),
            # dec32 = Faltungen in Decoder und Kopf (volle Aufloesung)
            if type(mod) is torch.nn.Conv2d and (
                    ("pe32" in VARIANTE and "patch_embed" in name)
                    or ("dec32" in VARIANTE and (name.startswith("decoder") or name.startswith("head")))):
                mod.float()
                mod.forward = conv32.__get__(mod)
                geflickt.append(mod)
                continue
            # nur Conv2d: eine fp32-ConvTranspose2d zwischen zwei Casts baut TensorRT 11.3 nicht
            # ("no tactics", /m/encoder/upsample_lowres/ConvTranspose, 24.09. 00:39)
            if "conv32" in VARIANTE and type(mod) is torch.nn.Conv2d:
                mod.float()
                mod.forward = conv32.__get__(mod)
                geflickt.append(mod)
            if "lin32" in VARIANTE and isinstance(mod, torch.nn.Linear):
                mod.forward = lin32.__get__(mod)
                geflickt.append(mod)
    if "sm32" in VARIANTE:
        sdpa_orig = torch.nn.functional.scaled_dot_product_attention

        def sdpa32(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
            assert attn_mask is None and not is_causal
            s = scale if scale is not None else q.shape[-1] ** -0.5
            a = torch.matmul(q, k.transpose(-2, -1)).float() * s
            return torch.matmul(a.softmax(dim=-1).to(q.dtype), v)
        torch.nn.functional.scaled_dot_product_attention = sdpa32
    wege = (True, False) if EXPORTWEG == "dynamo" else (False,)
    fehler = []
    try:
        return _exportieren(torch, h, x, pfad, wege, fehler)
    finally:
        for mod in geflickt:
            del mod.forward
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                mod.half()  # fp16 -> fp32 -> fp16 ist verlustfrei
        if "sm32" in VARIANTE:
            torch.nn.functional.scaled_dot_product_attention = sdpa_orig


def _exportieren(torch, h, x, pfad, wege, fehler):
    for dynamo in wege:
        try:
            with torch.inference_mode(False), torch.no_grad():
                kw = dict(input_names=["x"], output_names=["inverse_depth", "fov_deg"])
                if dynamo:
                    torch.onnx.export(h, (x,), pfad, dynamo=True, external_data=True,
                                      opset_version=int(os.environ.get("IW3_ONNX_OPSET", "20")),
                                      optimize=True, verify=False, **kw)
                else:
                    torch.onnx.export(h, (x,), pfad, dynamo=False, opset_version=17, **kw)
            return "dynamo" if dynamo else "torchscript"
        except Exception as e:  # noqa
            fehler.append(f"dynamo={dynamo}: {type(e).__name__}: {str(e)[:400]}")
            meldung("ONNX-Export " + fehler[-1])
    raise RuntimeError("ONNX-Export gescheitert: " + " | ".join(fehler))


def _engine_bauen(torch, trt, eager, device, ziel, meta):
    bau = os.path.join(engine_dir(), "_bau")
    shutil.rmtree(bau, ignore_errors=True)
    os.makedirs(bau)
    t0 = time.time()
    onnx_pfad = os.path.join(bau, "depthpro.onnx")
    weg = _onnx_exportieren(torch, eager, onnx_pfad, device)
    t_onnx = time.time() - t0
    torch.cuda.empty_cache()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_pfad):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX-Parser: {errs[:800]}")
    config = builder.create_builder_config()
    stufe = int(os.environ.get("IW3_TRT_OPT", "3"))
    config.builder_optimization_level = stufe
    cache_pfad = os.path.join(engine_dir(), f"timing_{meta['gpu']}_trt{meta['trt']}.cache")
    cache = config.create_timing_cache(open(cache_pfad, "rb").read() if os.path.isfile(cache_pfad) else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)
    t1 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("build_serialized_network -> None (siehe TensorRT-Meldungen oben)")
    t_bau = time.time() - t1
    with open(cache_pfad + ".neu", "wb") as f:
        f.write(memoryview(config.get_timing_cache().serialize()))
    os.replace(cache_pfad + ".neu", cache_pfad)
    daten = bytes(memoryview(plan))
    with open(ziel + ".neu", "wb") as f:
        f.write(daten)
    os.replace(ziel + ".neu", ziel)
    meta.update(onnx_weg=weg, onnx_s=round(t_onnx, 1), bau_s=round(t_bau, 1),
                opt_stufe=stufe, gebaut=time.strftime("%Y-%m-%d %H:%M:%S"),
                engine_sha256=hashlib.sha256(daten).hexdigest(), engine_byte=len(daten))
    with open(ziel + ".json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    shutil.rmtree(bau, ignore_errors=True)
    meldung(f"Engine gebaut: {os.path.basename(ziel)}  ONNX {t_onnx:.1f}s ({weg}) + TensorRT {t_bau:.1f}s")
    return daten


def eingangsgroesse(model_type):
    """DepthPro 1536, DepthPro_S 1024: vier Encoder-Kacheln breit (depth_pro.py: img_size)."""
    from iw3.depth_pro_model import NAME_MAP
    return NAME_MAP[model_type] * 4


def trt_laden(torch, eager_holen, model_type, device, bauen_erzwingen=False):
    """eager_holen() laedt das eager-Modell -- nur noetig, wenn gebaut werden muss. Mit
    vorhandener Engine spart das die ~20 s fuer torch.hub und 1,9 GB Gewichte je Lauf."""
    import tensorrt as trt
    kennung, gpu_name = gpu_kennung(torch, device)
    trt_ver = ".".join(trt.__version__.split(".")[:2])
    S = eingangsgroesse(model_type)
    ziel = os.path.join(engine_dir(), f"{model_type.lower()}_{S}_{VARIANTE}_{kennung}_trt{trt_ver}.engine")
    meta = dict(modell=model_type, eingang=[1, 3, S, S], dtype=VARIANTE, gpu=kennung, gpu_name=gpu_name,
                trt=trt_ver, trt_voll=trt.__version__, torch=torch.__version__,
                treiber=_treiber())
    daten = None
    if os.path.isfile(ziel) and os.path.isfile(ziel + ".json") and not bauen_erzwingen:
        try:
            alt = json.load(open(ziel + ".json", encoding="utf-8"))
            if (alt.get("gpu_name") == gpu_name and alt.get("trt_voll") == trt.__version__
                    and alt.get("treiber") == meta["treiber"]):
                daten = open(ziel, "rb").read()
                if hashlib.sha256(daten).hexdigest() != alt.get("engine_sha256"):
                    meldung("Engine-Pruefsumme passt nicht -> neu bauen")
                    daten = None
            else:
                meldung(f"Engine fuer {alt.get('gpu_name')}/TRT {alt.get('trt_voll')}/Treiber "
                        f"{alt.get('treiber')} -> neu bauen")
        except (OSError, ValueError) as e:
            meldung(f"Engine unlesbar ({e}) -> neu bauen")
            daten = None
    if daten is None:
        t = time.time()
        eager = eager_holen()
        assert eager.img_size == S, (eager.img_size, S)
        daten = _engine_bauen(torch, trt, eager, device, ziel, meta)
        MESSUNG.bau_s += time.time() - t
    MESSUNG.info = f"engine={os.path.basename(ziel)}"
    return TRTDepthPro(torch, trt, daten, S, device)


def _treiber():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa
        return "?"


# ------------------------------------------------------------- Inductor
def inductor(torch, model, device, modus):
    MESSUNG.info = f"cache={os.environ['TORCHINDUCTOR_CACHE_DIR']}"
    mode = "max-autotune" if modus == "inductor-max" else None
    return torch.compile(model, mode=mode)


# --------------------------------------------------------- Einklinken in iw3
def einklinken():
    global MODUS
    if MODUS == "inductor-max":
        # 24.09.2026 01:36: max-autotune (triton-windows 3.7.1, sm_120) -> alle 600 Karten konstant
        meldung("inductor-max ist gesperrt (lieferte konstante Tiefenkarten) -> eager")
        MODUS = "eager"
    if MODUS not in MODI:
        meldung(f"unbekannter Modus IW3_COMPILE={MODUS} ({'|'.join(MODI)}) -> eager")
        MODUS = "eager"
    import torch
    from iw3 import depth_pro_model

    if os.environ.get("IW3_ACCEL_CUDNN") == "1":
        torch.backends.cudnn.benchmark = True
    original = depth_pro_model.DepthProModel.load_model

    def load_model(self, model_type, resolution=None, device=None):
        cuda = device is not None and getattr(device, "type", str(device)) == "cuda"
        if MODUS == "trt" and cuda:
            geladen = {}

            def eager_holen():
                if "m" not in geladen:
                    geladen["m"] = original(self, model_type, resolution=resolution, device=device)
                return geladen["m"]
            try:
                ersatz = trt_laden(torch, eager_holen, model_type, device)
                geladen.clear()
                torch.cuda.empty_cache()
                return Gemessen(torch, ersatz)
            except Exception as e:  # noqa
                import traceback
                traceback.print_exc()
                meldung(f"TensorRT FEHLGESCHLAGEN ({type(e).__name__}: {e}) -> eager")
                return Gemessen(torch, eager_holen())
        model = original(self, model_type, resolution=resolution, device=device)
        if not cuda:
            meldung(f"Geraet {device} ist nicht CUDA -> eager")
            return Gemessen(torch, model)
        if MODUS == "inductor":
            return Gemessen(torch, inductor(torch, model, device, MODUS))
        return Gemessen(torch, model)

    depth_pro_model.DepthProModel.load_model = load_model


def vorab_bauen(model_type="DepthPro"):
    """Engine ohne iw3-Lauf bauen (z. B. nach einem Kartenwechsel)."""
    import torch
    from iw3.depth_pro_model import DepthProModel
    m = DepthProModel(model_type)
    m.load(gpu=0)  # ohne einklinken(): das eager-Modell
    t = time.time()
    trt_laden(torch, lambda: m.model, model_type, m.device, bauen_erzwingen=True)
    meldung(f"vorab gebaut in {time.time() - t:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--accel-bauen":
        vorab_bauen(sys.argv[2] if len(sys.argv) > 2 else "DepthPro")
        sys.exit(0)
    einklinken()
    sys.argv[0] = "iw3"
    runpy.run_module("iw3", run_name="__main__", alter_sys=True)
