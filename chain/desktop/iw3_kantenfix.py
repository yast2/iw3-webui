#!/usr/bin/env python3
"""`python3 -m iw3`, aber mit nunif 40d46847 im Import-Pfad.

24.09.2026. Der Pin d23721f1 in /opt/nunif kehrt beim Import
einer --export-YAML die Kantenweitung um:

    iw3/utils.py 1720  depths = -dilate_edge(-depths, args.edge_dilation)
    iw3/utils.py 1908  depth = -dilate_edge(-depth.unsqueeze(0), ...).squeeze(0)

Upstream 40d468475558 (17.06.2026, nur dev, "Fix edge dilation when importing
YAML exported using --export") rechnet dilate_edge(depths, ...) ohne die zwei
Vorzeichen. Genau diese zwei Zeilen werden hier getauscht, sonst nichts; jede
muss genau einmal vorkommen, sonst Abbruch statt eines stillen Laufs ohne Fix.
/opt/nunif bleibt unberuehrt -- der Tausch lebt nur in diesem Prozess.

Aufruf wie `python3 -m iw3 ...`, aus /opt/nunif (run_voll.sh macht `cd` dorthin).

27.09.2026: --kein-faststart (setzt nur run_chain.py, fuer den verlustfreien Zwischenstand).
iw3/utils.py 1075/1795 schreibt jede mp4 mit movflags +faststart; beim Zwischenstand von
Job (x264 crf 0, 104 GB) schob das den ganzen mdat ein zweites Mal durch die Platte,
~38 min bei 40-65 MB/s, obwohl die Kodierung ihn danach ohnehin ganz liest. Die Lieferdatei
behaelt ihr +faststart (ffmpeg-Aufruf der Kodierung). Ohne die Option aendert sich nichts.
"""
import importlib.util
import os
import runpy
import sys

TAUSCH = (
    ("depths = -dilate_edge(-depths, args.edge_dilation)",
     "depths = dilate_edge(depths, args.edge_dilation)"),
    ("depth = -dilate_edge(-depth.unsqueeze(0), args.edge_dilation).squeeze(0)",
     "depth = dilate_edge(depth.unsqueeze(0), args.edge_dilation).squeeze(0)"),
)
FASTSTART = 'container_options={"movflags": "+faststart"} if args.video_format == "mp4" else {},'
OHNE_FASTSTART = 'container_options={},'


def quelle_getauscht(pfad, ohne_faststart=False):
    with open(pfad, encoding="utf-8") as f:
        quelle = f.read()
    for alt, neu in TAUSCH:
        n = quelle.count(alt)
        if n != 1:
            sys.exit(f"### ABBRUCH Kantenfix: {n}x statt 1x gefunden: {alt}")
        quelle = quelle.replace(alt, neu)
    if ohne_faststart:
        # Kostet nur Zeit, nie Bilder: passt die Stelle nicht, bleibt faststart an, kein Abbruch.
        n = quelle.count(FASTSTART)
        if n == 2:
            quelle = quelle.replace(FASTSTART, OHNE_FASTSTART)
            print("[kantenfix] faststart aus (Zwischenstand, 2 Stellen)", flush=True)
        else:
            print(f"[kantenfix] WARNUNG: faststart-Stelle {n}x statt 2x - faststart bleibt an", flush=True)
    return quelle


def lade_gepatcht(ohne_faststart=False):
    """iw3.utils aus der getauschten Quelle laden, bevor iw3.cli es importiert."""
    # wie -m: das Arbeitsverzeichnis (/opt/nunif) statt des Skriptordners vorn
    if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
        sys.path[0] = os.getcwd()
    else:
        sys.path.insert(0, os.getcwd())
    import iw3  # Paket; sein __init__ laedt utils nicht
    if "iw3.utils" in sys.modules:
        sys.exit("### ABBRUCH Kantenfix: iw3.utils war schon geladen, der Tausch kaeme zu spaet")
    spec = importlib.util.find_spec("iw3.utils")
    modul = importlib.util.module_from_spec(spec)
    sys.modules["iw3.utils"] = modul
    exec(compile(quelle_getauscht(spec.origin, ohne_faststart), spec.origin, "exec"), modul.__dict__)
    iw3.utils = modul
    return modul


if __name__ == "__main__":
    ohne_faststart = "--kein-faststart" in sys.argv[1:]
    if ohne_faststart:
        sys.argv.remove("--kein-faststart")  # iw3 selbst kennt die Option nicht
    lade_gepatcht(ohne_faststart)
    print("[kantenfix] iw3.utils mit nunif 40d46847 geladen (2 Zeilen getauscht)", flush=True)
    runpy.run_module("iw3", run_name="__main__", alter_sys=True)
