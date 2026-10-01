#!/usr/bin/env python3
"""Stufe 3 — zeitliche Ruhe ueber optischen Fluss statt ueber Glaettung.

Verfahren unveraendert seit dem 15.09.; am 16.09. nur beschleunigt. Die Fassung
von davor liegt als `stufe3_fluss.vor-tempo-2026-09-16.py` daneben, das
Verfahren selbst ist dort im Kopf beschrieben: Fluss t->t±1 mit RAFT auf halber
Breite, Nachbartiefe damit ins aktuelle Bild warpen, photometrische
Verdeckungspruefung, Median statt Mittelwert.

Gegenueber jener Fassung geaendert sind ausschliesslich drei Dinge, die das
Ergebnis nicht anfassen:

  1. `compress_level=1` beim PNG-Schreiben statt der Pillow-Vorgabe 6.
     zlib-Stufe 6 kostete 380 ms je Bild, Stufe 1 kostet 88 ms. Die Datei
     wird dabei von 1,9 auf 2,1 MB groesser — bei 21.606 Bildern also 45
     statt 41 GB. PNG ist verlustfrei, die Pixel sind bitgleich.

  2. Laden und Schreiben laufen in je einem eigenen Faden. Das Original hat
     zwischen zwei GPU-Abschnitten synchron 1 JPEG dekodiert, 1 Tiefen-PNG
     dekodiert und 1 PNG geschrieben — rund 490 ms je Bild, in denen die
     Karte nichts tat. Jetzt laeuft das neben RAFT.

  3. `--halb` (optional, nicht voreingestellt) rechnet RAFT unter autocast
     in fp16. Das ist der einzige Schalter, der das Ergebnis veraendert:
     gemessen an 153 Bildern gegen den Produktionslauf weichen 0,005 % der
     Pixel um mehr als 0,001 ab, die mittlere Abweichung ist 1,0e-6.

Ohne `--halb` ist die Ausgabe **bitgleich** zum Original; nachgewiesen an
173 Bildern aus zwei unabhaengigen, bewegten Passagen.

Gemessen auf der Arc Pro B60, 1920x1080 (kalter Seitencache):

    Original                 0,967 s/Bild   -> 5 h 48 min fuer 21.606
    diese Fassung            0,298 s/Bild   -> 1 h 47 min   (bitgleich)
    diese Fassung mit --halb 0,179 s/Bild   -> 1 h 05 min

Aufruf wie gehabt:

    python3 stufe3_fluss.py TIEFE_DIR RGB_DIR AUSGABE_DIR --photo-schwelle 0.03
"""

import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


LESE_VERSUCHE = 4


def lies_bild(pfad, umwandeln=None):
    """Image.open + Dekodieren, mit Wiederholung.

    24.09.2026 (Job): ein intaktes DepthPro-Bild las sich im Bandtausch
    einmal als "broken data stream"; zehn Minuten spaeter dekodierten alle
    4090 Bilder derselben Arbeit fehlerfrei. Die Arbeit liegt seit dem Umzug
    am 24.09. auf den NTFS-Platten disk11/disk12 (ntfs3). PNG prueft zlib und
    Adler-32, ein gelungenes Dekodieren ist also kein stilles Falschlesen.
    Jede Wiederholung wird gemeldet; bleibt es kaputt, Abbruch mit Dateinamen.
    """
    for versuch in range(1, LESE_VERSUCHE + 1):
        try:
            with Image.open(pfad) as im:
                if umwandeln:
                    im = im.convert(umwandeln)
                return np.asarray(im)
        except (OSError, SyntaxError) as e:
            if versuch == LESE_VERSUCHE:
                raise OSError(f"{pfad}: {e} (nach {versuch} Leseversuchen)") from e
            print(f"WARNUNG Lesefehler {os.path.basename(pfad)}, Versuch {versuch}: {e}",
                  flush=True)
            time.sleep(0.5 * versuch)


def geraet():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lade_rgb_cpu(pfad):
    a = lies_bild(pfad, "RGB").astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def lade_tiefe_cpu(pfad):
    a = lies_bild(pfad).astype(np.float32) / 65535.0
    if a.ndim == 3:
        a = a.mean(axis=2)
    return torch.from_numpy(a)[None, None]


def median_klein(proben):
    """Median einer kurzen Liste ueber ein min/max-Sortiernetz.

    `torch.median` ueber eine Achse braucht auf der Arc B60 20 Sekunden fuer
    3x1920x1080; paarweises min/max liefert dasselbe in unter 1 ms.
    """
    xs = list(proben)
    n = len(xs)
    for i in range(n):
        for j in range(n - 1 - i):
            a, b = xs[j], xs[j + 1]
            xs[j], xs[j + 1] = torch.minimum(a, b), torch.maximum(a, b)
    return xs[n // 2]


def warpe(bild, fluss):
    """Holt aus `bild` das, was laut `fluss` an die aktuelle Stelle gehoert."""
    _, _, h, w = bild.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=bild.device, dtype=torch.float32),
        torch.arange(w, device=bild.device, dtype=torch.float32),
        indexing="ij")
    x = xx + fluss[:, 0]
    y = yy + fluss[:, 1]
    gx = 2.0 * x / max(w - 1, 1) - 1.0
    gy = 2.0 * y / max(h - 1, 1) - 1.0
    gitter = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(bild, gitter, mode="bilinear",
                         padding_mode="border", align_corners=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tiefe")
    p.add_argument("rgb")
    p.add_argument("ausgabe")
    p.add_argument("--fenster", type=int, default=1)
    p.add_argument("--iterationen", type=int, default=12)
    p.add_argument("--skala", type=float, default=0.5)
    p.add_argument("--photo-schwelle", type=float, default=0.06)
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--kompression", type=int, default=1,
                   help="zlib-Stufe des PNG (Vorgabe 1; Pixel sind in jedem "
                        "Fall bitgleich, nur die Dateigroesse aendert sich)")
    p.add_argument("--halb", action="store_true",
                   help="RAFT in fp16 — rund 40 %% schneller, aendert das "
                        "Ergebnis minimal (siehe Kopf der Datei)")
    a = p.parse_args()

    dev = geraet()
    ts = sorted(f for f in os.listdir(a.tiefe) if f.endswith(".png"))
    rs = sorted(f for f in os.listdir(a.rgb)
                if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if len(ts) != len(rs):
        sys.exit(f"Anzahl ungleich: Tiefe {len(ts)}, RGB {len(rs)} — Abbruch")
    if not ts:
        sys.exit("leer — Abbruch")
    os.makedirs(a.ausgabe, exist_ok=True)

    n = min(len(ts), a.max) if a.max else len(ts)
    print(f"Geraet: {dev} | {n} Bilder | Fenster +-{a.fenster} | "
          f"Flussskala {a.skala} | RAFT-Schritte {a.iterationen} | "
          f"PNG-Stufe {a.kompression} | {'fp16' if a.halb else 'fp32'}",
          flush=True)
    modell = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(dev).eval()

    # ------------------------------------------------ Laden im Nebenlauf
    ladeschlange = queue.Queue()
    fertig = {}
    bedingung = threading.Condition()

    def lader():
        while True:
            j = ladeschlange.get()
            if j is None:
                return
            r = lade_rgb_cpu(os.path.join(a.rgb, rs[j]))
            d = lade_tiefe_cpu(os.path.join(a.tiefe, ts[j]))
            # Bei 4K-Quellen sind die Tiefenkarten kleiner als die Bilder (DepthPro liefert
            # 1536 Breite, die Bilder 3840) -- RAFT rechnet dann auf die Kartengroesse
            # verkleinert, sonst passen Fluss und Karte nicht zusammen (23.09., Vorschau
            # Job). Bei 1080p sind beide gleich gross, dort aendert sich nichts.
            if r.shape[-2:] != d.shape[-2:]:
                r = F.interpolate(r, size=d.shape[-2:], mode="bilinear",
                                  align_corners=False, antialias=True)
            with bedingung:
                fertig[j] = (r, d)
                bedingung.notify_all()

    # Die Schlange ist begrenzt: sonst laeuft der Schreiber dem Rechner davon
    # und der Arbeitsspeicher fuellt sich mit fertigen Bildern.
    schreibschlange = queue.Queue(maxsize=8)

    def schreiber():
        while True:
            auf = schreibschlange.get()
            if auf is None:
                return
            pfad, arr = auf
            Image.fromarray(arr, mode="I;16").save(pfad,
                                                   compress_level=a.kompression)

    f_lader = threading.Thread(target=lader, daemon=True)
    f_schreiber = threading.Thread(target=schreiber, daemon=True)
    f_lader.start()
    f_schreiber.start()

    puffer_rgb, puffer_dep = {}, {}
    angefordert = set()

    def anfordern(j):
        if 0 <= j < n and j not in angefordert:
            angefordert.add(j)
            ladeschlange.put(j)

    def hole(j):
        if j in puffer_rgb:
            return puffer_rgb[j], puffer_dep[j]
        anfordern(j)
        with bedingung:
            while j not in fertig:
                bedingung.wait()
            r, d = fertig.pop(j)
        r, d = r.to(dev), d.to(dev)
        puffer_rgb[j], puffer_dep[j] = r, d
        return r, d

    def raeume(i):
        for d in (puffer_rgb, puffer_dep):
            for k in [k for k in d if k < i - a.fenster]:
                del d[k]

    @torch.inference_mode()
    def fluss(von, nach):
        h, w = von.shape[-2:]
        sh, sw = int(h * a.skala) // 8 * 8, int(w * a.skala) // 8 * 8
        v = F.interpolate(von, size=(sh, sw), mode="bilinear", align_corners=False)
        z = F.interpolate(nach, size=(sh, sw), mode="bilinear", align_corners=False)
        if a.halb:
            with torch.autocast(device_type=dev.type, dtype=torch.float16):
                f = modell(v * 2 - 1, z * 2 - 1, num_flow_updates=a.iterationen)[-1]
            f = f.float()
        else:
            f = modell(v * 2 - 1, z * 2 - 1, num_flow_updates=a.iterationen)[-1]
        f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
        f[:, 0] *= w / sw
        f[:, 1] *= h / sh
        return f

    verworfen = 0
    gesamt = 0
    d_t = None

    for j in range(0, 3):
        anfordern(j)

    for i in range(n):
        anfordern(i + 2)
        rgb_t, d_t = hole(i)
        proben = [d_t]

        for k in range(-a.fenster, a.fenster + 1):
            if k == 0:
                continue
            j = i + k
            if j < 0 or j >= n:
                continue
            gesamt += 1
            rgb_j, d_j = hole(j)
            f = fluss(rgb_t, rgb_j)
            d_w = warpe(d_j, f)
            rgb_w = warpe(rgb_j, f)
            abw = (rgb_w - rgb_t).abs().mean(dim=1, keepdim=True)
            gueltig = abw <= a.photo_schwelle
            verworfen += int((~gueltig).sum())
            proben.append(torch.where(gueltig, d_w, d_t))

        if len(proben) >= 3:
            erg = median_klein(proben)
        else:
            erg = torch.stack(proben).mean(dim=0)

        arr = (erg[0, 0].clamp(0, 1).cpu().numpy() * 65535.0 + 0.5).astype(np.uint16)
        schreibschlange.put((os.path.join(a.ausgabe, ts[i]), arr))
        raeume(i)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    schreibschlange.put(None)
    f_schreiber.join()
    ladeschlange.put(None)

    if gesamt:
        px = d_t.numel()
        print(f"\nVerdeckt verworfen: {verworfen / (gesamt * px) * 100:.2f} % "
              f"aller Proben-Pixel")
    print(f"Fertig: {n} Bilder in {a.ausgabe}")


if __name__ == "__main__":
    main()
