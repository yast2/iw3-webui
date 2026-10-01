#!/usr/bin/env python3
"""Ein ueberzaehliges VDA-Tiefenbild am Ende entfernen -- nur, wenn es nachweislich am Ende liegt.

    python3 vda_ende_pruefen.py VDA_DIR PRO_DIR N

iw3 liest das Video in Stufe 4 selbst und fuellt das Ende mitunter um ein Bild auf, waehrend
Stufe 1 (-fps_mode passthrough) genau die Bilder der Datei zieht (23.09.2026: 1801 gegen 1800
bei einem selbst geschnittenen 60-s-Clip). Geprueft wird die Bewegung (Differenz
aufeinanderfolgender Karten) von VDA gegen DepthPro in sechs Fenstern ueber den Clip, Versatz 0
gegen +1. Ein Bild zu viel mitten im Clip zeigt ab dort Versatz +1 -> keine Aenderung, Ausgang 1.
Ausgang 0 = letztes VDA-Bild entfernt.
"""
import os
import sys

import numpy as np
from PIL import Image


def main():
    vda_dir, pro_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    vda = sorted(f for f in os.listdir(vda_dir) if f.endswith(".png"))
    pro = sorted(f for f in os.listdir(pro_dir) if f.endswith(".png"))
    if len(pro) != n or len(vda) != n + 1 or n < 120:
        print(f"    VDA-Ende: nicht pruefbar (VDA {len(vda)}, DepthPro {len(pro)}, Quelle {n})", flush=True)
        return 1
    groesse = Image.open(os.path.join(vda_dir, vda[0])).size

    def karte(pfad):
        bild = Image.open(pfad)
        if bild.size != groesse:
            bild = bild.resize(groesse, Image.BILINEAR)
        return np.asarray(bild, dtype=np.float32).ravel()

    def korr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

    # Bewegung ueber SCHRITT Bilder, in den sechs bewegtesten von bis zu zwanzig Fenstern: wo
    # sich nichts bewegt (schwarzer Anfang, Standbild), ist jeder Versatz gleich gut und die
    # Pruefung stumm. Am 60-fps-Testfilm (21.600 Bilder) liegt die Spitze in 7 von 8 Fenstern bei 0,
    # z. B. +0,60 gegen +0,50 bei +1; der Ausreisser ist das schwarze Ende.
    SCHRITT, LAENGE = 4, 30
    rand = max(1, n // 50)
    kandidaten = sorted(set(np.linspace(rand, n - rand - LAENGE - SCHRITT - 2, 20).astype(int)))
    bewegung = []
    for s in kandidaten:
        p0, p1 = karte(os.path.join(pro_dir, pro[s])), karte(os.path.join(pro_dir, pro[s + LAENGE]))
        bewegung.append((float(np.abs(p1 - p0).mean()), s))
    fenster = [s for _, s in sorted(bewegung, reverse=True)[:6]]
    # Versatz 0 muss in jedem Fenster vor BEIDEN Nachbarn liegen: ein eingeschobenes Bild
    # mitten im Clip zeigt ab dort +1, ein fehlendes -1 -- beides heisst Abbruch.
    siege, m0, mn = 0, [], []
    for s in sorted(fenster):
        s = max(s, 1)
        p = [karte(os.path.join(pro_dir, pro[i])) for i in range(s, s + LAENGE + SCHRITT)]
        v = {i: karte(os.path.join(vda_dir, vda[i])) for i in range(s - 1, s + LAENGE + SCHRITT + 1)}

        def k(off):
            return float(np.mean([korr(v[i + off + SCHRITT] - v[i + off], p[i - s + SCHRITT] - p[i - s])
                                  for i in range(s, s + LAENGE)]))
        k0, kn = k(0), max(k(-1), k(1))
        m0.append(k0)
        mn.append(kn)
        siege += k0 > kn
    a, b = float(np.mean(m0)), float(np.mean(mn))
    zeile = (f"Abgleich Bewegung ({SCHRITT} Bilder, {len(fenster)} bewegteste Fenster): Versatz 0 in "
             f"{siege}/{len(fenster)} vorn, Mittel {a:+.3f} gegen {b:+.3f} beim besseren Nachbarn")
    if siege == len(fenster) == 6 and a - b >= 0.03:
        os.remove(os.path.join(vda_dir, vda[-1]))
        print(f"    VDA: 1 Bild zu viel am Ende ({vda[-1]}) entfernt -- {zeile}", flush=True)
        return 0
    print(f"    VDA: 1 Bild zu viel, Lage nicht eindeutig -- {zeile}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
