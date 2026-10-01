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


def quelle_getauscht(pfad):
    with open(pfad, encoding="utf-8") as f:
        quelle = f.read()
    for alt, neu in TAUSCH:
        n = quelle.count(alt)
        if n != 1:
            sys.exit(f"### ABBRUCH Kantenfix: {n}x statt 1x gefunden: {alt}")
        quelle = quelle.replace(alt, neu)
    return quelle


def lade_gepatcht():
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
    exec(compile(quelle_getauscht(spec.origin), spec.origin, "exec"), modul.__dict__)
    iw3.utils = modul
    return modul


if __name__ == "__main__":
    lade_gepatcht()
    print("[kantenfix] iw3.utils mit nunif 40d46847 geladen (2 Zeilen getauscht)", flush=True)
    runpy.run_module("iw3", run_name="__main__", alter_sys=True)
