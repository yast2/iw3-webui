#!/usr/bin/env python3
"""Hardlink-Kopie eines Bildordners unter Quellbild-Nummerierung.

VDA bekommt einen Videoausschnitt und nummeriert seine Ausgabe wieder ab 1.
Damit die ganze Kette eine einzige Nummerierung spricht (und ein Ausschnitt
spaeter ueber die Bildnummer statt ueber den Index gezogen werden kann),
wird die Ausgabe hier auf die Quellnummer umgehaengt.

    umnummerieren.py QUELLE ZIEL ERSTE_QUELLNUMMER
"""
import os, sys

q, z, erste = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(z, exist_ok=True)
alle = sorted(f for f in os.listdir(q) if f.endswith(".png"))
for i, f in enumerate(alle):
    b = os.path.join(z, f"{erste + i:08d}.png")
    if not os.path.exists(b):
        os.link(os.path.join(q, f), b)
da = sorted(os.listdir(z))
print(f"{z}: {len(da)} Dateien, {da[0]} .. {da[-1]}  (Quelle {len(alle)}, "
      f"{alle[0]} .. {alle[-1]})")
