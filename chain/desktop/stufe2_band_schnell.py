#!/usr/bin/env python3
"""Stufe 2, Bandtausch — schnelle Fassung. Rechenweg unveraendert.


Gegen `stufe2_band.py` geaendert ist ausschliesslich, **wie** gerechnet wird,
nie **was**:

1. **PNG-Schreiben in Faeden, zlib-Stufe 1.** Die Zerlegung vom 17.09. weist
   446 ms von 540 ms je Bild dem PIL-Speichern zu — 83 %. Dasselbe Rezept hat
   am 16.09. den Fluss von 0,888 auf 0,295 s/Bild gebracht. zlib-Stufe 1
   aendert nur die Containerbytes, nicht ein einziges Pixel; verglichen wird
   deshalb nach Regel 3 ueber die **dekodierten Pixel**, nicht ueber die Datei.
2. **PNG-Lesen in Faeden, vorausschauend.** 55 ms je Bild, ueberlappbar.
3. **Exakte Quantile statt 200.000 Zufallspixel.**
   Das Original zieht `torch.randperm(...)[:200000]` **ohne gesetzten
   Zufallsgenerator** und ist damit von Lauf zu Lauf verschieden (nachgewiesen:
   20 von 20 Bildern unterschiedlich bei identischem Aufruf). Die
   Unterabtastung ist ein Schaetzer fuer die Quantile der Vollverteilung; hier
   wird der Grenzwert dieses Schaetzers genommen, also **derselbe Abgleich ohne
   Stichprobenrauschen**. Nebeneffekt: `randperm` kostete 13,9 ms je Bild.
   Damit ist die Stufe erstmals reproduzierbar.

Aufruf wie gehabt:

    python3 stufe2_band_schnell.py BASIS_DIR DETAIL_DIR AUSGABE_DIR --radius 30
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def geraet():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def kastenfilter(x, r):
    """Mittelwert ueber ein (2r+1)^2-Fenster, separabel und randgespiegelt."""
    k = 2 * r + 1
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.avg_pool2d(x, (1, k), stride=1)
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    x = F.avg_pool2d(x, (k, 1), stride=1)
    return x


def lade_np(pfad):
    a = np.asarray(Image.open(pfad)).astype(np.float32) / 65535.0
    if a.ndim == 3:
        a = a.mean(axis=2)
    return a


def fit(x, y):
    """Ausgleichsgerade y = m*x + c ueber 33 Punkte, geschlossen in float64.

    Das Original nimmt hier `torch.linalg.lstsq`. Auf der B60 ist das ein
    Rueckfall XPU -> CPU, und der ist **nicht reproduzierbar**: bei
    nachweislich bitgleichen Eingaben `dq`/`bq` kamen in acht Durchlaeufen
    zwei verschiedene Loesungen heraus (1,111627936363 gegen 1,111628055573,
    achte Stelle). Das war die einzige verbliebene Quelle des
    Nichtdeterminismus und schlug als 1 Stufe von 65535 bis ins Bild durch.

    Fuer zwei Parameter und 33 Punkte braucht es kein LAPACK: die
    Normalgleichung ist geschlossen loesbar, in float64 gerechnet genauer als
    die QR-Zerlegung in float32 und ohne jeden Thread reproduzierbar.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x.mean()
    ym = y.mean()
    dx = x - xm
    m = float((dx * (y - ym)).sum() / (dx * dx).sum())
    c = float(ym - m * xm)
    rest = float(np.abs(m * x + c - y).mean())
    return (m, c), rest


def abgleich(detail, basis, form):
    """Robuste affine Abbildung DepthPro -> Basisskala ueber 33 Quantilpaare.

    Zwei Aenderungen gegen das Original, beide nur am Rechenweg:
    - Quantile ueber **alle** Pixel statt ueber 200.000 zufaellig gezogene;
    - Geradenfit geschlossen in float64 statt `torch.linalg.lstsq`.
    """
    d = detail.flatten()
    b = basis.flatten()
    qs = torch.linspace(0.01, 0.99, 33, device=d.device)
    dq = torch.quantile(d, qs).double().cpu().numpy()
    bq = torch.quantile(b, qs).double().cpu().numpy()

    lin, r_lin = fit(dq, bq)
    kehr, r_kehr = fit(1.0 / (dq + 1e-4), bq)
    if form == "auto":
        form = "linear" if r_lin <= r_kehr else "kehrwert"
    if form == "linear":
        return form, r_lin, r_kehr, (lambda t: lin[0] * t + lin[1])
    return form, r_lin, r_kehr, (lambda t: kehr[0] / (t + 1e-4) + kehr[1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("basis")
    p.add_argument("detail")
    p.add_argument("ausgabe")
    p.add_argument("--radius", type=int, default=30)
    p.add_argument("--form", choices=("linear", "kehrwert", "auto"), default="linear")
    p.add_argument("--staerke", type=float, default=1.0)
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--lese-faeden", type=int, default=4)
    p.add_argument("--schreib-faeden", type=int, default=6)
    p.add_argument("--zlib", type=int, default=1, help="PNG-Kompressionsstufe (Vorgabe 1)")
    p.add_argument("--vorrat", type=int, default=8, help="wie viele Bilder vorausgelesen werden")
    a = p.parse_args()

    dev = geraet()
    bs = sorted(f for f in os.listdir(a.basis) if f.endswith(".png"))
    ds = sorted(f for f in os.listdir(a.detail) if f.endswith(".png"))
    if len(bs) != len(ds):
        sys.exit("Anzahl ungleich: Basis %d, Detail %d — Abbruch" % (len(bs), len(ds)))
    if not bs:
        sys.exit("leer — Abbruch")
    os.makedirs(a.ausgabe, exist_ok=True)
    n = min(len(bs), a.max) if a.max else len(bs)
    print("Geraet: %s | %d Bilder | Radius %d | Form %s | Staerke %s | "
          "Lesefaeden %d | Schreibfaeden %d | zlib %d"
          % (dev, n, a.radius, a.form, a.staerke, a.lese_faeden,
             a.schreib_faeden, a.zlib), flush=True)

    leser = ThreadPoolExecutor(max_workers=a.lese_faeden)
    schreiber = ThreadPoolExecutor(max_workers=a.schreib_faeden)

    def lies(i):
        return (lade_np(os.path.join(a.basis, bs[i])),
                lade_np(os.path.join(a.detail, ds[i])))

    def schreib(pfad, arr):
        Image.fromarray(arr, mode="I;16").save(pfad, compress_level=a.zlib)

    # Invertierungsprobe ueber das ganze Material, nicht nur ueber Bild 1: ein
    # schwarzes Startbild (Abblende, Titelkarte) gibt beiden Modellen nichts zu
    # sehen, und ihre erfundenen Karten korrelieren beliebig (60-fps-Testclip
    # Purge: Bild 1 -0,596, Median ueber 20 Bilder +0,810, 18 von 20 positiv).
    probe = sorted({int(round(x)) for x in np.linspace(0, n - 1, min(25, n))})
    kk = []
    for j in probe:
        aB, aD = lies(j)
        B = torch.from_numpy(aB)[None, None].to(dev)
        D = torch.from_numpy(aD)[None, None].to(dev)
        if D.shape[-2:] != B.shape[-2:]:
            D = F.interpolate(D, size=B.shape[-2:], mode="bilinear", align_corners=False)
        k = float(torch.corrcoef(torch.stack([B.flatten(), D.flatten()]))[0, 1])
        if k == k:  # NaN bei einer ganz flachen Karte: keine Aussage, auslassen
            kk.append(k)
    if kk:
        k = float(np.median(kk))
        print("Korrelation Basis<->Detail: Median %+.3f ueber %d Bilder (min %+.3f, max %+.3f)"
              % (k, len(kk), min(kk), max(kk)), flush=True)
        if k < 0:
            sys.exit("negativ — eine der beiden Karten ist invertiert. Abbruch.")
    else:
        print("Korrelation Basis<->Detail: keine auswertbare Probe (alle Karten flach)", flush=True)

    # Vorrat an Lesevorgaengen, damit die GPU nie auf die Platte wartet.
    warteschlange = Queue()
    for i in range(min(a.vorrat, n)):
        warteschlange.put((i, leser.submit(lies, i)))
    naechster = min(a.vorrat, n)

    schreibfutures = []
    t0 = time.time()
    abgeschnitten_max = 0.0
    for i in range(n):
        _, fut = warteschlange.get()
        aB, aD = fut.result()
        if naechster < n:
            warteschlange.put((naechster, leser.submit(lies, naechster)))
            naechster += 1

        B = torch.from_numpy(aB)[None, None].to(dev)
        D = torch.from_numpy(aD)[None, None].to(dev)
        if D.shape[-2:] != B.shape[-2:]:
            if D.shape[-1] * D.shape[-2] >= B.shape[-1] * B.shape[-2]:
                B = F.interpolate(B, size=D.shape[-2:], mode="bilinear",
                                  align_corners=False)
            else:
                D = F.interpolate(D, size=B.shape[-2:], mode="bilinear",
                                  align_corners=False)

        form, r_lin, r_kehr, abb = abgleich(D, B, a.form)
        Da = abb(D)
        erg = kastenfilter(B, a.radius) + a.staerke * (Da - kastenfilter(Da, a.radius))

        ab = float(((erg < 0) | (erg > 1)).float().mean()) * 100.0
        abgeschnitten_max = max(abgeschnitten_max, ab)

        arr = (erg[0, 0].clamp(0, 1).cpu().numpy() * 65535.0 + 0.5).astype(np.uint16)
        schreibfutures.append(schreiber.submit(
            schreib, os.path.join(a.ausgabe, bs[i]), arr))
        # Rueckstau begrenzen, sonst laeuft der Speicher voll.
        if len(schreibfutures) >= 3 * a.schreib_faeden:
            schreibfutures.pop(0).result()

        if i == 0:
            print("Abgleichform: %s  (Restfehler linear %.5f, kehrwert %.5f)"
                  % (form, r_lin, r_kehr), flush=True)
            for name, t in (("Basis  ", B), ("Detail ", Da), ("Ergebnis", erg)):
                q = torch.quantile(t.flatten(), torch.tensor(
                    [0.01, 0.5, 0.9, 0.99], device=t.device))
                print("  %s p1 %+.4f  p50 %+.4f  p90 %+.4f  p99 %+.4f"
                      % (name, q[0], q[1], q[2], q[3]), flush=True)
            aend = erg - B
            print("Aenderung gegen Basis: std %.5f, p1..p99 %+.4f .. %+.4f"
                  % (float(aend.std()),
                     float(torch.quantile(aend.flatten(), 0.01)),
                     float(torch.quantile(aend.flatten(), 0.99))), flush=True)
            print("Abgeschnitten (ausserhalb 0..1): %.3f %%" % ab, flush=True)
        if (i + 1) % 200 == 0:
            v = (time.time() - t0) / (i + 1)
            print("  %d/%d  %.3f s/Bild  Rest %.1f min"
                  % (i + 1, n, v, (n - i - 1) * v / 60), flush=True)

    for f in schreibfutures:
        f.result()
    leser.shutdown(wait=True)
    schreiber.shutdown(wait=True)

    ges = time.time() - t0
    print("\nFertig: %d Bilder in %s  (%.1f min, %.4f s/Bild)"
          % (n, a.ausgabe, ges / 60, ges / n))
    print("Abgeschnitten, schlimmstes Bild: %.3f %%" % abgeschnitten_max)


if __name__ == "__main__":
    main()
