#!/usr/bin/env python3
"""VDA-Karten den Quellbildern zuordnen, wenn iw3 im Videopfad Bilder gedoppelt oder ausgelassen hat --
nachgewiesen, nicht geraten. PC-Fassung (run_chain.py), nach iw3:/config/chain/vda_abgleich.py vom 30.09.2026.

    python vda_abgleich.py SRC EXPORT_YML ROH_DEPTH VDA_DIR PRO_DIR FRAMES_DIR N [--trocken]

Stufe 1 zieht mit -fps_mode passthrough genau die N Bilder der Datei; iw3 liest das Video in der VDA-Stufe selbst
und schickt jedes Bild durch ffmpegs fps-Filter (nunif utils/video.py hook_frame, FixedFPSFilter, Pin d23721f1).
Laeuft die Uhr der Datei gegen die Nennrate, setzt der Filter dort, wo die Drift eine halbe Bilddauer erreicht, ein
Quellbild doppelt -- mitten im Film (30.09., Server, 60-fps-Film: ein Bild, Karte 40080). Pendelt die Drift dort um
die halbe Bilddauer, doppelt der Filter UND laesst aus, abwechselnd (01.10., PC, .mov-Quelle, .mov 30/1:
18.120 Bilder -> 18.121 Plaetze, 5 Doppelungen und 4 Auslassungen zwischen Bild 9004 und 9014).

1. Zuordnung: die Quelle wird genau wie in iw3 dekodiert und gefiltert (dieselbe Klasse, dieselbe Rate:
   guessed_rate, gedeckelt durch max_fps aus der yml; HEVC ohne Thread-Dekodierung wie hook_frame), jedes Bild vor und
   nach dem Filter per Pruefsumme erkannt. Ergebnis: welcher Filterplatz (= VDA-Karte) welches Quellbild traegt.
   Erlaubt ist nur das Filtermuster: naechstes Bild, dasselbe noch einmal, oder genau eines ausgelassen.
   Ein ausgelassenes Quellbild i bekommt die Karte, die an seinem Zeitplatz steht: die zweite Kopie von i-1 direkt
   davor (VDA hat dort Bild i-1 noch einmal gesehen, 1/Rate daneben). Ohne diese Kopie, am Rand, bei mehr als
   MAX_LUECKEN Auslassungen oder wenn Dekodierung und Stufe 1/VDA nicht zusammen passen: Abbruch.
2. Gegenprobe Fenster (Inhalt, unabhaengig von 1.): Bewegung VDA gegen DepthPro (Differenz ueber 4 Bilder,
   Korrelation) mit der neuen Zuordnung, in den 6 bewegtesten von 20 Fenstern (Bewegung in frames/ gemessen) plus
   je einem Fenster direkt hinter jeder Stelle. Versatz 0 muss in JEDEM Fenster vor BEIDEN Nachbarn liegen, >= 6.
3. Ortsprobe (Inhalt, unabhaengig von 1. und 2.): die zwei Karten desselben Quellbilds unterscheiden sich nur um VDAs
   zeitliche Glaettung. Jeder Doppelplatz muss eine Delle sein: Differenz zur Vorgaengerkarte <= 0,7x BEIDER
   direkter Nachbardifferenzen und <= 0,7x Median der uebrigen in +-3 Plaetzen. Zufallsrate dieser Regel an
   .mov-Quelle (aufrechte Karten, 18.106 Nicht-Doppel-Stellen): 1,37 %. Bild dort ruhig (< 0,3 Grauwerte): ohne Aussage.
4. Erst wenn 2. und 3. bestehen: VDA_DIR neu aus ROH_DEPTH verlinkt (vda.neu -> Umbenennen).
Ausgang 0 = VDA_DIR hat jetzt N Karten in Quellnummerierung (mit --trocken: haette), sonst 1 (nichts veraendert).
"""
import hashlib
import json
import os
import re
import shutil
import sys
from fractions import Fraction

import numpy as np
from PIL import Image

NUNIF = os.environ.get("NUNIF_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nunif")
sys.path.insert(0, NUNIF)
import av  # noqa: E402
from nunif.utils.video import AV_READ_OPTIONS, FixedFPSFilter, convert_known_fps  # noqa: E402

MAX_LUECKEN = 10


def meld(t):
    print(f"    VDA-Abgleich: {t}", flush=True)


def pruefsumme(fr):
    return hashlib.blake2b(np.frombuffer(fr.planes[0], np.uint8)[::61].tobytes(), digest_size=8).hexdigest()


def yml_werte(pfad):
    text = open(pfad, encoding="utf-8").read()
    try:
        import yaml
        y = yaml.safe_load(text)
        return Fraction(str(y["fps"])), float(y["user_data"]["export_options"]["max_fps"])
    except ImportError:
        fps = re.search(r"^fps:\s*'?([0-9./]+)", text, re.M).group(1)
        mx = re.search(r"^\s+max_fps:\s*([0-9.]+)", text, re.M).group(1)
        return Fraction(fps), float(mx)


def dekodieren(src, max_fps, yml_fps, ablage):
    """Wie hook_frame: Dekodieren, fps-Filter, Nachziehen am Ende. Pruefsummen ein/aus, in ABLAGE gemerkt
    (Schluessel: Groesse, mtime, Rate), damit ein zweiter Lauf auf derselben Quelle nicht neu dekodiert."""
    st_src = os.stat(src)
    schluessel = [st_src.st_size, int(st_src.st_mtime), str(yml_fps), max_fps]
    try:
        with open(ablage) as f:
            d = json.load(f)
        if d["schluessel"] == schluessel:
            meld(f"Dekodierung aus {ablage} ({len(d['ein'])} / {len(d['aus'])})")
            return Fraction(d["fps"]), d["ein"], d["aus"]
    except (OSError, ValueError, KeyError):
        pass
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
    try:
        with open(ablage + ".neu", "w") as f:
            json.dump({"schluessel": schluessel, "fps": str(fps), "ein": ein, "aus": aus}, f)
        os.replace(ablage + ".neu", ablage)
    except OSError:
        pass
    return fps, ein, aus


def plaetze(ein, aus):
    """Platz k -> Quellbild. Erlaubt: naechstes Bild, dasselbe noch einmal, oder eines ausgelassen. Gleiche
    Nachbarbilder (Standbild) werden vorgezogen; ein falscher Sprung endet spaeter in einer Pruefsumme, die nicht
    passt -> Abbruch."""
    if not aus or aus[0] != ein[0]:
        raise SystemExit("Platz 0 traegt nicht Quellbild 0")
    platz, j = [0], 0
    for k in range(1, len(aus)):
        h = aus[k]
        if j + 1 < len(ein) and ein[j + 1] == h:
            j += 1
        elif ein[j] == h:
            pass
        elif j + 2 < len(ein) and ein[j + 2] == h:
            j += 2
        else:
            raise SystemExit(f"Platz {k}: Pruefsumme passt zu keinem der Quellbilder {j}..{j + 2}")
        platz.append(j)
    return platz


def main():
    a = [x for x in sys.argv[1:] if x != "--trocken"]
    trocken = "--trocken" in sys.argv
    src, yml, roh, vda_dir, pro_dir, frames_dir, n = a[0], a[1], a[2], a[3], a[4], a[5], int(a[6])
    yml_fps, max_fps = yml_werte(yml)
    roh_karten = sorted(f for f in os.listdir(roh) if f.endswith(".png"))
    pro = sorted(f for f in os.listdir(pro_dir) if f.endswith(".png"))
    bilder = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if len(pro) != n or len(bilder) != n:
        meld(f"nicht pruefbar (DepthPro {len(pro)}, Bilder {len(bilder)}, Quelle {n})")
        return 1

    try:
        fps, ein, aus = dekodieren(src, max_fps, yml_fps, os.path.join(os.path.dirname(roh), "vda_abgleich_dekodierung.json"))
        platz = plaetze(ein, aus)
    except SystemExit as e:
        meld(f"Zuordnung gescheitert -- {e}")
        return 1
    if len(ein) != n or len(platz) != len(roh_karten):
        meld(f"Dekodierung {len(ein)} Quellbilder / {len(platz)} Plaetze, Stufe 1 {n} / VDA {len(roh_karten)} -- passt nicht")
        return 1
    doppelt = [k for k in range(1, len(platz)) if platz[k] == platz[k - 1]]
    erste = {}
    for k, i in enumerate(platz):
        erste.setdefault(i, k)
    fehlend = [i for i in range(n) if i not in erste]
    gleich = sum(1 for i in range(1, n) if ein[i] == ein[i - 1])
    karte_zu_bild = [erste.get(i) for i in range(n)]  # Quellbild i (0-basiert) -> Platz/roh-Index
    if len(fehlend) > MAX_LUECKEN:
        meld(f"{len(fehlend)} Quellbilder ausgelassen (1-basiert {[i + 1 for i in fehlend[:10]]} ...) -- mehr als {MAX_LUECKEN}")
        return 1
    for i in fehlend:
        k = erste.get(i - 1, -9) + 1
        if not (0 < i < n - 1) or k >= len(platz) or platz[k] != i - 1 or erste.get(i + 1) != k + 1:
            meld(f"Quellbild {i + 1} ausgelassen, aber an seinem Platz steht keine zweite Kopie von Bild {i} -- Abbruch")
            return 1
        karte_zu_bild[i] = k
    ersatz = {k for k in (karte_zu_bild[i] for i in fehlend)}
    weg = [k for k in doppelt if k not in ersatz]  # Doppelplaetze, die keine Luecke fuellen -> entfallen
    # 1-basiert wie die Dateinamen in vda/ (Karte k+1 = roh-Platz k)
    liste = ", ".join(f"Karte {k + 1} = Bild {platz[k] + 1}" for k in doppelt[:8]) + (" ..." if len(doppelt) > 8 else "")
    luecken = ", ".join(f"Bild {i + 1} <- Karte {karte_zu_bild[i] + 1}" for i in fehlend)
    meld(f"fps {fps}: {n} Quellbilder -> {len(platz)} Plaetze, {len(doppelt)} gedoppelt ({liste}); "
         f"{len(fehlend)} ausgelassen{' (' + luecken + ')' if fehlend else ''}; gleiche Nachbarbilder in der Quelle: {gleich}")
    if not doppelt and not fehlend:
        meld("keine Doppelung gefunden, Zaehlung trotzdem daneben -- Abbruch")
        return 1

    # ---- 2. Gegenprobe Fenster
    groesse = Image.open(os.path.join(roh, roh_karten[0])).size

    def karte(pfad, grau=False, gr=groesse):
        with Image.open(pfad) as b:
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

    def bildbewegung(s):
        return float(np.mean([np.abs(karte(os.path.join(frames_dir, bilder[i + SCHRITT]), True, (240, 135))
                                     - karte(os.path.join(frames_dir, bilder[i]), True, (240, 135))).mean()
                              for i in range(s, s + LAENGE, 6)]))

    SCHRITT, LAENGE = 4, 30
    rand = max(1, n // 50)
    kandidaten = sorted(set(np.linspace(rand, n - rand - LAENGE - SCHRITT - 2, 20).astype(int)))
    fenster = [s for _, s in sorted(((bildbewegung(s), s) for s in kandidaten), reverse=True)[:6]]
    stellen = sorted({platz[k] for k in doppelt} | set(fehlend))
    haeufungen = [[stellen[0]]]  # Stellen mit weniger als 30 Bildern Abstand gehoeren zusammen
    for x in stellen[1:]:
        if x - haeufungen[-1][-1] <= 30:
            haeufungen[-1].append(x)
        else:
            haeufungen.append([x])
    for h in haeufungen:  # ein Fenster direkt hinter jeder Haeufung, sofern dort Platz und Bewegung ist
        s = h[-1] + 2
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

    # ---- 3. Ortsprobe: jeder Doppelplatz ist eine Delle -- hoechstens 0,7x BEIDER direkter Nachbardifferenzen und
    # 0,7x Median der uebrigen in +-3 Plaetzen. Bezug bewusst oertlich: an der .mov-Quelle (01.10.) faellt das Tempo im Bild
    # ueber 30 Plaetze auf ein Viertel (Differenzen ~1000 vor 9003, ~250 hinter 9015). Ein Median ueber +-15 machte
    # aus der Delle bei 9004 (529,6 gegen 912,3/1256,4) 0,94x und brach den Lauf 07:50 ab; eine Rangfolge ueber +-15
    # (erste Fassung) ebenso. Zufallsrate der oertlichen Regel ueber den ganzen Film 1,37 %, alle 5 Doppelungen
    # 0,42-0,58x; eine falsch gesetzte Doppelung 1,29x. interne Messnotiz
    geprueft, alle_doppel = set(), set(doppelt)
    for k in doppelt:
        if k in geprueft:
            continue
        lo, hi = max(1, k - 15), min(len(platz) - 1, k + 15)
        hier = [d for d in doppelt if lo <= d <= hi]
        geprueft.update(hier)
        hi = min(len(platz) - 1, max(hier) + 15)
        dv = {j: float(np.abs(karte(os.path.join(roh, roh_karten[j]))
                              - karte(os.path.join(roh, roh_karten[j - 1]))).mean()) for j in range(lo, hi + 1)}
        rest = [dv[j] for j in dv if j not in alle_doppel]
        med = float(np.median(rest))

        def oertlich(d):  # Median der Nicht-Doppel-Differenzen in +-3 Plaetzen
            return float(np.median([dv[j] for j in range(d - 3, d + 4) if j in dv and j != d and j not in alle_doppel]))
        quote = {d: dv[d] / (min(dv.get(d - 1, np.inf), dv.get(d + 1, np.inf), oertlich(d)) + 1e-9) for d in hier}
        i0, i1 = max(0, platz[k] - 15), min(n - 1, platz[max(hier)] + 15)
        ruhe = float(np.mean([np.abs(karte(os.path.join(frames_dir, bilder[i]), True, (240, 135))
                                     - karte(os.path.join(frames_dir, bilder[i - 1]), True, (240, 135))).mean()
                              for i in range(i0 + 1, i1 + 1)]))
        text = (f"Doppelplaetze {[d + 1 for d in hier]}: Differenz zur Vorgaengerkarte (Quote gegen Nachbarn und Median +-3) "
                f"{', '.join(f'{dv[d]:.1f} ({quote[d]:.2f}x)' for d in hier)}, Median der uebrigen in +-15 {med:.1f}; "
                f"Bildaenderung {ruhe:.2f}")
        if max(quote.values()) <= 0.7:
            meld(f"Ortsprobe bestanden -- {text}")
        elif ruhe < 0.3:
            meld(f"Ortsprobe ohne Aussage, Bild ruhig -- {text}")
        else:
            meld(f"Ortsprobe NICHT bestanden -- {text}")
            return 1

    if trocken:
        meld(f"[trocken] wuerde {vda_dir} aus {len(roh_karten)} Rohkarten neu verlinken: Rohkarte(n) "
             f"{[roh_karten[k] for k in weg[:8]]} entfallen, {len(fehlend)} Luecke(n) mit der Karte an ihrem Platz gefuellt")
        return 0
    neu, alt = vda_dir.rstrip("/\\") + ".neu", vda_dir.rstrip("/\\") + ".alt"
    for d in (neu, alt):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(neu)
    for i in range(n):
        ziel = os.path.join(neu, f"{i + 1:08d}.png")
        try:
            os.link(os.path.join(roh, roh_karten[karte_zu_bild[i]]), ziel)
        except OSError:
            shutil.copy2(os.path.join(roh, roh_karten[karte_zu_bild[i]]), ziel)
    if len([f for f in os.listdir(neu) if f.endswith(".png")]) != n:
        meld(f"{neu}: nicht {n} Karten -- Abbruch, {vda_dir} unveraendert")
        return 1
    if os.path.exists(vda_dir):
        os.rename(vda_dir, alt)
    os.rename(neu, vda_dir)
    shutil.rmtree(alt, ignore_errors=True)
    meld(f"{vda_dir} neu verlinkt: {n} Karten, Rohkarte(n) {[roh_karten[k] for k in weg[:8]]} entfallen, "
         f"{len(fehlend)} Luecke(n) gefuellt ({luecken or '-'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
