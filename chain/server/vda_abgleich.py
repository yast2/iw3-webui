#!/usr/bin/env python3
"""VDA-Karten den Quellbildern zuordnen, wenn iw3 im Videopfad Bilder gedoppelt hat -- nachgewiesen, nicht geraten.

    python3 vda_abgleich.py SRC EXPORT_YML ROH_DEPTH VDA_DIR PRO_DIR FRAMES_DIR N [--trocken]

Stufe 1 zieht mit -fps_mode passthrough genau die N Bilder der Datei; iw3 liest das Video in der VDA-Stufe selbst
und schickt jedes Bild durch ffmpegs fps-Filter (nunif utils/video.py hook_frame, FixedFPSFilter, Pin d23721f1).
Laeuft die Uhr der Datei gegen die Nennrate (30.09.2026, 60-fps-Film: 12,5 ppm langsam, 1448 Ticks Drift ueber
77.421 Bilder), setzt der Filter dort, wo die Drift eine halbe Bilddauer erreicht, ein Quellbild doppelt -- mitten
im Film, nicht am Ende. Ab da steht jede VDA-Karte um eins versetzt.

1. Zuordnung: die Quelle wird genau wie in iw3 dekodiert und gefiltert (dieselbe Klasse, dieselbe Rate:
   guessed_rate, gedeckelt durch max_fps aus der yml), jedes Bild vor und nach dem Filter per Pruefsumme
   erkannt. Ergebnis: welcher Filterplatz (= VDA-Karte) welches Quellbild traegt.
   Abbruch, wenn die Dekodierung nicht N Quellbilder oder nicht so viele Plaetze wie VDA-Karten liefert, wenn
   ein Quellbild fehlt (ausgelassen statt gedoppelt) oder die Rate nicht zur yml passt.
2. Gegenprobe am Inhalt: Bewegung VDA gegen DepthPro (Differenz ueber 4 Bilder, Korrelation) mit der neuen
   Zuordnung, in den 6 bewegtesten von 20 Fenstern (Bewegung im Bild frames/ gemessen, nicht in DepthPro --
   interne Messnotiz) plus je einem Fenster direkt nach jeder Doppelung. Versatz 0 muss
   in jedem Fenster vor BEIDEN Nachbarn liegen, im Mittel um >= 0,03 (Schwellen wie vda_ende_pruefen.py).
3. Erst dann: VDA_DIR neu aus ROH_DEPTH verlinkt, je Quellbild die ERSTE Karte, die es traegt; die Doppelkarte
   bleibt nur in ROH_DEPTH.
Ausgang 0 = VDA_DIR hat jetzt N Karten in Quellnummerierung (bzw. mit --trocken: waere so), sonst 1.
"""
import hashlib
import os
import sys
from fractions import Fraction

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, "/opt/nunif")
import av  # noqa: E402
from nunif.utils.video import AV_READ_OPTIONS, FixedFPSFilter, convert_known_fps  # noqa: E402


def meld(t):
    print(f"    VDA-Abgleich: {t}", flush=True)


def pruefsumme(fr):
    return hashlib.blake2b(np.frombuffer(fr.planes[0], np.uint8)[::61].tobytes(), digest_size=8).digest()


def zuordnung(src, max_fps, yml_fps):
    """Wie hook_frame: Dekodieren, fps-Filter, Nachziehen am Ende. Liefert (rate, Pruefsummen ein, Plaetze->Quellbild)."""
    c = av.open(src, **AV_READ_OPTIONS)
    st = c.streams.video[0]
    if st.codec.name != "hevc":
        st.thread_type = "AUTO"
    fps = st.guessed_rate
    if float(fps) > max_fps:
        fps = max_fps
    fps = convert_known_fps(fps)
    if abs(float(fps) - float(yml_fps)) > 1e-6:
        raise SystemExit(f"Rate {fps} passt nicht zur yml ({yml_fps})")
    filt = FixedFPSFilter(st, fps=fps)
    ein, aus = [], []
    for pkt in c.demux([st]):
        for fr in pkt.decode():
            ein.append(pruefsumme(fr))
            o = filt.update(fr)
            if o is not None:
                aus.append(pruefsumme(o))
    while True:
        o = filt.update(None)
        if o is None:
            break
        aus.append(pruefsumme(o))
    c.close()
    # Platz k traegt dasselbe Quellbild wie Platz k-1 oder das naechste; alles andere ist kein fps-Filter-Muster.
    # Gleiche Nachbarbilder (Standbild) werden vorgezogen: welche von zwei gleichen Karten entfaellt, ist gleichgueltig,
    # und ein zu frueher Sprung endet spaeter in einer Pruefsumme, die nicht passt -> Abbruch.
    platz, j = [], 0
    for k, h in enumerate(aus):
        if k > 0:
            if j + 1 < len(ein) and ein[j + 1] == h:
                j += 1
            elif ein[j] != h:
                raise SystemExit(f"Platz {k}: Pruefsumme passt weder zu Quellbild {j} noch {j + 1}")
        elif ein[0] != h:
            raise SystemExit("Platz 0 traegt nicht Quellbild 0")
        platz.append(j)
    return fps, ein, platz


def main():
    a = [x for x in sys.argv[1:] if x != "--trocken"]
    trocken = "--trocken" in sys.argv
    src, yml, roh, vda_dir, pro_dir, frames_dir, n = a[0], a[1], a[2], a[3], a[4], a[5], int(a[6])
    with open(yml) as f:
        y = yaml.safe_load(f)
    max_fps = float(y["user_data"]["export_options"]["max_fps"])
    roh_karten = sorted(f for f in os.listdir(roh) if f.endswith(".png"))
    pro = sorted(f for f in os.listdir(pro_dir) if f.endswith(".png"))
    bilder = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if len(pro) != n or len(bilder) != n:
        meld(f"nicht pruefbar (DepthPro {len(pro)}, Bilder {len(bilder)}, Quelle {n})")
        return 1

    try:
        fps, ein, platz = zuordnung(src, max_fps, Fraction(str(y["fps"])))
    except SystemExit as e:
        meld(f"Zuordnung gescheitert -- {e}")
        return 1
    if len(ein) != n or len(platz) != len(roh_karten):
        meld(f"Dekodierung {len(ein)} Quellbilder / {len(platz)} Plaetze, Stufe 1 {n} / VDA {len(roh_karten)} -- passt nicht")
        return 1
    fehlend = sorted(set(range(n)) - set(platz))
    if fehlend:
        meld(f"fps-Filter laesst Quellbilder aus (1-basiert {[i + 1 for i in fehlend[:10]]}) -- dafuer gibt es keine Karte")
        return 1
    doppelt = [k for k in range(1, len(platz)) if platz[k] == platz[k - 1]]
    gleich = sum(1 for i in range(1, n) if ein[i] == ein[i - 1])
    erste = {}
    for k, i in enumerate(platz):
        erste.setdefault(i, k)
    karte_zu_bild = [erste[i] for i in range(n)]  # Quellbild i (0-basiert) -> Platz/roh-Index
    # 1-basiert wie die Dateinamen: Karte (k+1) ist die zweite Kopie von Bild platz[k]+1
    liste = ", ".join(f"Karte {k + 1} = Bild {platz[k] + 1}" for k in doppelt[:8]) + (" ..." if len(doppelt) > 8 else "")
    meld(f"fps {fps}: {n} Quellbilder -> {len(platz)} Plaetze, {len(doppelt)} gedoppelt ({liste}); "
         f"gleiche Nachbarbilder in der Quelle: {gleich}")
    if not doppelt:
        meld("keine Doppelung gefunden, Zaehlung trotzdem daneben -- Abbruch")
        return 1

    # ---- Gegenprobe am Inhalt
    groesse = Image.open(os.path.join(roh, roh_karten[0])).size

    def karte(pfad, grau=False, gr=groesse):
        b = Image.open(pfad)
        if grau:
            b = b.convert("L")
        if b.size != gr:
            b = b.resize(gr, Image.BILINEAR)
        return np.asarray(b, dtype=np.float32).ravel()

    def vda(i):  # Quellbild i (0-basiert) unter der neuen Zuordnung
        return karte(os.path.join(roh, roh_karten[karte_zu_bild[i]]))

    def korr(x, z):
        x = x - x.mean()
        z = z - z.mean()
        return float((x * z).sum() / (np.sqrt((x * x).sum() * (z * z).sum()) + 1e-9))

    SCHRITT, LAENGE = 4, 30
    rand = max(1, n // 50)
    kandidaten = sorted(set(np.linspace(rand, n - rand - LAENGE - SCHRITT - 2, 20).astype(int)))

    def bildbewegung(s):
        return float(np.mean([np.abs(karte(os.path.join(frames_dir, bilder[i + SCHRITT]), True, (240, 135))
                                     - karte(os.path.join(frames_dir, bilder[i]), True, (240, 135))).mean()
                              for i in range(s, s + LAENGE, 6)]))

    fenster = [s for _, s in sorted(((bildbewegung(s), s) for s in kandidaten), reverse=True)[:6]]
    for k in doppelt:  # direkt hinter jeder Doppelung, sofern dort Platz ist
        s = platz[k] + 2
        if s + LAENGE + SCHRITT + 2 < n and bildbewegung(s) >= 0.3:
            fenster.append(s)
    fenster = sorted(set(max(s, 2) for s in fenster))
    siege, m0, mn, zeilen = 0, [], [], []
    for s in fenster:
        p = [karte(os.path.join(pro_dir, pro[i])) for i in range(s, s + LAENGE + SCHRITT)]
        v = {i: vda(i) for i in range(s - 1, s + LAENGE + SCHRITT + 1)}

        def k_(off):
            return float(np.mean([korr(v[i + off + SCHRITT] - v[i + off], p[i - s + SCHRITT] - p[i - s])
                                  for i in range(s, s + LAENGE)]))
        km, k0, kp = k_(-1), k_(0), k_(1)
        m0.append(k0)
        mn.append(max(km, kp))
        siege += k0 > max(km, kp)
        zeilen.append(f"{s + 1}: {km:+.3f}/{k0:+.3f}/{kp:+.3f}")
    a0, b0 = float(np.mean(m0)), float(np.mean(mn))
    zeile = (f"Bewegung VDA gegen DepthPro, neue Zuordnung, Fenster ab Bild (-1/0/+1): {' | '.join(zeilen)} -> "
             f"Versatz 0 in {siege}/{len(fenster)} vorn, Mittel {a0:+.3f} gegen {b0:+.3f}")
    if not (siege == len(fenster) and len(fenster) >= 6):
        meld(f"Gegenprobe NICHT bestanden -- {zeile}")
        return 1
    meld(f"Gegenprobe Fenster bestanden -- {zeile}")

    # Ortsprobe an jeder Doppelung: die beiden Karten desselben Quellbilds unterscheiden sich nur um VDAs zeitliche
    # Glaettung, also deutlich weniger als Karten benachbarter Bilder. Die kleinste Nachbardifferenz im Umkreis von
    # 15 Plaetzen muss genau dort liegen und hoechstens 0,7x den Median betragen. Ist das Bild dort ruhig (mittlere
    # Aenderung < 0,3 Grauwerte), sind alle Karten gleich und die Lage innerhalb der Ruhe ohne Belang.
    for k in doppelt:
        lo, hi = max(1, k - 15), min(len(platz) - 1, k + 15)
        dv = {j: float(np.abs(karte(os.path.join(roh, roh_karten[j]))
                              - karte(os.path.join(roh, roh_karten[j - 1]))).mean()) for j in range(lo, hi + 1)}
        med = float(np.median(list(dv.values())))
        tief = min(dv, key=dv.get)
        i0, i1 = max(0, platz[k] - 15), min(n - 1, platz[k] + 15)
        ruhe = float(np.mean([np.abs(karte(os.path.join(frames_dir, bilder[i]), True, (240, 135))
                                     - karte(os.path.join(frames_dir, bilder[i - 1]), True, (240, 135))).mean()
                              for i in range(i0 + 1, i1 + 1)]))
        text = (f"Karte {k + 1}: Differenz zur Vorgaengerkarte {dv[k]:.1f}, Median im Umkreis {med:.1f} "
                f"({dv[k] / (med + 1e-9):.2f}x), kleinste bei Karte {tief + 1}; Bildaenderung {ruhe:.2f}")
        if tief == k and dv[k] <= 0.7 * med:
            meld(f"Ortsprobe bestanden -- {text}")
        elif ruhe < 0.3:
            meld(f"Ortsprobe ohne Aussage, Bild ruhig -- {text}")
        else:
            meld(f"Ortsprobe NICHT bestanden -- {text}")
            return 1

    if trocken:
        meld(f"[trocken] wuerde {vda_dir} aus {len(roh_karten)} Rohkarten neu verlinken, {len(doppelt)} Doppelkarte(n) weg")
        return 0
    neu, alt = vda_dir.rstrip("/") + ".neu", vda_dir.rstrip("/") + ".alt"
    for d in (neu, alt):
        if os.path.exists(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
            os.rmdir(d)
    os.makedirs(neu)
    for i in range(n):
        os.link(os.path.join(roh, roh_karten[karte_zu_bild[i]]), os.path.join(neu, f"{i + 1:08d}.png"))
    os.rename(vda_dir, alt)
    os.rename(neu, vda_dir)
    for f in os.listdir(alt):
        os.remove(os.path.join(alt, f))
    os.rmdir(alt)
    meld(f"{vda_dir} neu verlinkt: {n} Karten, Rohkarte(n) {[roh_karten[k] for k in doppelt[:8]]} ausgelassen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
