#!/usr/bin/env python3
"""Baut ein iw3-Exportverzeichnis fuer den Volllauf.

Unterschiede zu `build_export.py` (das fuer die 60-s-Tests gebaut wurde):

  * `--rgb-dir` uebernimmt bereits extrahierte JPEGs per Hardlink, statt sie ein
    zweites Mal aus dem Video zu ziehen. Bei 21.606 Bildern spart das rund zehn
    Minuten und fuenf Gigabyte.
  * `--mapper` wird durchgereicht. Die Testlaeufe 19-21 liefen mit
    `mul_1+mul_2=0.5`; `build_export.py` hatte `none` fest verdrahtet, und die
    yml wurde danach von Hand nachgezogen. Das ist hier nicht mehr noetig.
  * Der Ton wird mitextrahiert. Die Testclips waren stumm, weil das bei einer
    Minute Pruefmaterial niemanden gestoert hat.

Laeuft im iw3-Container, wo /opt/nunif liegt.
"""

import argparse
import os
import shutil
import subprocess
import sys
from fractions import Fraction

sys.path.insert(0, "/opt/nunif")
from iw3.export_config import ExportConfig, VIDEO_TYPE  # noqa: E402

RGB_ENDUNGEN = (".jpg", ".png")


def fps_von(ffprobe, video):
    roh = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True).stdout.strip()
    # 01.10.2026, .mov-Quelle (.mov, Display-Matrix -90): csv haengt die Seitendaten an
    # dieselbe Zeile, "30/1,Display Matrix,...,-90" -- int("1,") warf ValueError, Export exit 1,
    # fuenfmal. Nur das erste Feld der ersten Zeile ist die Bildrate.
    n, d = roh.splitlines()[0].split(",")[0].strip().split("/")
    return Fraction(int(n), int(d))


def verknuepfe(quelle, ziel):
    if os.path.exists(ziel):
        return
    try:
        os.link(quelle, ziel)
    except OSError:
        shutil.copy2(quelle, ziel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("quellvideo")
    p.add_argument("tiefenordner")
    p.add_argument("exportdir")
    p.add_argument("--rgb-dir", default=None,
                   help="fertige JPEG-Sequenz; ohne Angabe wird aus dem Video extrahiert")
    p.add_argument("--mapper", default="none")
    p.add_argument("--skip-mapper", action="store_true",
                   help="nur sinnvoll bei --mapper none")
    p.add_argument("--kein-ton", action="store_true")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--qscale", type=int, default=2)
    # Polsterung war bis 16.09. auf 3 vorbelegt und hat die Bildspur um genau
    # diese drei Bilder gegen den Ton verschoben: iw3 warpt jedes Bild einzeln,
    # muxt audio.m4a ab 0 und macht aus 21.606 Paaren 21.612 Bilder. Bei den
    # stummen 60-s-Testclips fiel das nicht auf. Einen Zweck hat die Polsterung
    # in dieser Konfiguration nicht -- ein zeitliches Fenster hat hier nichts:
    # --ema-normalize ist aus, und process_images() setzt das Konvergenzmodell
    # ohnehin mit enable_ema=False zurueck. Vorgabe deshalb 0.
    p.add_argument("--pad", type=int, default=0)
    a = p.parse_args()

    fps = fps_von(a.ffprobe, a.quellvideo)
    rgb = os.path.join(a.exportdir, "rgb")
    dep = os.path.join(a.exportdir, "depth")
    os.makedirs(rgb, exist_ok=True)
    os.makedirs(dep, exist_ok=True)
    print(f"fps {fps}  ->  {a.exportdir}", flush=True)

    # 26.09.2026: --upscale liefert rgb2x/ als PNG (waifu2x --format png); hier wurden nur
    # JPEGs gesucht, "uebernehme 0 JPEGs ... leer -- Abbruch" -- kein Hochskalier-Auftrag kam je
    # durch den Export. iw3 liest rgb/ ueber ImageLoader.listdir, jede Bildendung geht. Die
    # Endung der Quelle bleibt erhalten; der JPEG-Weg ist unveraendert.
    if a.rgb_dir:
        quell_rgb = sorted(f for f in os.listdir(a.rgb_dir) if f.endswith(RGB_ENDUNGEN))
        if len({os.path.splitext(f)[1] for f in quell_rgb}) > 1:
            sys.exit(f"{a.rgb_dir}: JPEG und PNG gemischt -- Abbruch")
        print(f"uebernehme {len(quell_rgb)} Bilder aus {a.rgb_dir}", flush=True)
        for i, f in enumerate(quell_rgb):
            verknuepfe(os.path.join(a.rgb_dir, f),
                       os.path.join(rgb, f"{i + a.pad:08d}{os.path.splitext(f)[1]}"))
    else:
        subprocess.run(
            [a.ffmpeg, "-v", "error", "-i", a.quellvideo,
             # Ohne passthrough gleicht ffmpeg die Bildrate an und verdoppelt
             # dabei still einzelne Bilder.
             "-fps_mode", "passthrough",
             "-start_number", str(a.pad),
             "-qscale:v", str(a.qscale), os.path.join(rgb, "%08d.jpg")],
            check=True)

    if not a.kein_ton:
        ton = os.path.join(a.exportdir, "audio.m4a")
        if not os.path.exists(ton):
            r = subprocess.run(
                [a.ffmpeg, "-v", "error", "-y", "-i", a.quellvideo,
                 "-vn", "-c:a", "copy", ton],
                capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(ton):
                # AAC-Kopie scheitert bei manchen Containern; dann neu kodieren.
                r = subprocess.run(
                    [a.ffmpeg, "-v", "error", "-y", "-i", a.quellvideo,
                     "-vn", "-c:a", "aac", "-b:a", "192k", ton],
                    capture_output=True, text=True)
            print(f"Ton: {'ok' if os.path.exists(ton) else 'FEHLT -- ' + r.stderr[:200]}",
                  flush=True)

    rgbs = sorted(f for f in os.listdir(rgb) if f.endswith(RGB_ENDUNGEN))
    rgb_endung = os.path.splitext(rgbs[0])[1][1:] if rgbs else "jpg"
    quellen = sorted(f for f in os.listdir(a.tiefenordner) if f.endswith(".png"))
    n_rgb, n_dep = len(rgbs), len(quellen)
    print(f"RGB {n_rgb} Bilder, Tiefe {n_dep} Bilder", flush=True)
    if n_rgb == 0 or n_dep == 0:
        sys.exit("leer -- Abbruch")

    toleranz = max(2, int(0.002 * max(n_rgb, n_dep)))
    if abs(n_rgb - n_dep) > toleranz:
        sys.exit(f"Abweichung {abs(n_rgb - n_dep)} > Toleranz {toleranz} -- Abbruch")
    if n_rgb != n_dep:
        print(f"  Abweichung {abs(n_rgb - n_dep)} innerhalb Toleranz {toleranz}: "
              f"letztes Tiefenbild wird wiederholt", flush=True)

    for i in range(n_rgb):
        verknuepfe(os.path.join(a.tiefenordner, quellen[min(i, n_dep - 1)]),
                   os.path.join(dep, f"{i + a.pad:08d}.png"))

    erste, letzte = a.pad, a.pad + n_rgb - 1
    for i in range(a.pad):
        for ordner, endung, von, nach in (
                (rgb, rgb_endung, erste, i), (dep, "png", erste, i),
                (rgb, rgb_endung, letzte, letzte + 1 + i), (dep, "png", letzte, letzte + 1 + i)):
            q = os.path.join(ordner, f"{von:08d}.{endung}")
            if os.path.exists(q):
                verknuepfe(q, os.path.join(ordner, f"{nach:08d}.{endung}"))

    t_rgb = len([f for f in os.listdir(rgb) if f.endswith(RGB_ENDUNGEN)])
    t_dep = len([f for f in os.listdir(dep) if f.endswith(".png")])
    if t_rgb != t_dep:
        sys.exit(f"nach Polsterung ungleich: {t_rgb} RGB vs {t_dep} Tiefe")

    cfg = ExportConfig(
        type=VIDEO_TYPE,
        basename=os.path.basename(a.quellvideo),
        fps=fps,
        mapper=a.mapper,
        skip_mapper=bool(a.skip_mapper),
        skip_edge_dilation=False,
    )
    yml = os.path.join(a.exportdir, "iw3_export.yml")
    cfg.save(yml)
    print(f"fertig: {n_rgb} echte + {t_rgb - n_rgb} Polster = {t_rgb} Paare", flush=True)
    print(open(yml).read(), flush=True)


if __name__ == "__main__":
    main()
