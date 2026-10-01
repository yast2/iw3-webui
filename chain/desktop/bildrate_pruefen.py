#!/usr/bin/env python3
"""bildrate_pruefen.py QUELLE -- sagt, ob die Quelle vor der Kette auf feste Bildrate muss.

24.09.2026, eine Handyaufnahme: Android-Aufnahme, gemeldet r_frame_rate 120/1, tatsaechlich 6326 Bilder
in 208,66 s (Abstand im Median 33,3 ms, 5,5 bis 70,8 ms). ffmpeg -fps_mode passthrough zog die 6326
echten Bilder, iw3 las fuer VDA die gemeldeten 120 fps und fuellte auf 25038 auf, und
build_export_full.py haette 120 fps ins yml geschrieben. Die Kette braucht eine Quelle, deren
gemeldete Bildrate stimmt.

Ausgabe auf stdout:
  nichts            gemeldete Rate x Dauer trifft die Bildzahl auf 1,5 Bilder genau (Normalfall)
  eine Normrate     sonst die, die dem echten Schnitt am naechsten liegt, z. B. "30" oder "30000/1001"
Exit 0 in beiden Faellen, 1 wenn ffprobe die Quelle nicht lesen kann.
"""
import json
import math
import subprocess
import sys
from fractions import Fraction

NORM = ["12", "15", "20", "24000/1001", "24", "25", "30000/1001", "30", "48", "50",
        "60000/1001", "60", "90", "100", "120000/1001", "120"]


def bruch(s):
    try:
        f = Fraction(s)
        return f if f > 0 else None
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def main():
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
                        "-of", "json", sys.argv[1]], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"### ffprobe: {r.stderr.strip()[:200]}", file=sys.stderr)
        return 1
    d = json.loads(r.stdout)
    s = d["streams"][0]
    dauer = float(s.get("duration") or d.get("format", {}).get("duration") or 0)
    gemeldet, schnitt = bruch(s.get("r_frame_rate")), bruch(s.get("avg_frame_rate"))
    n = int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None
    if n is None and schnitt:
        n = round(float(schnitt) * dauer)
    # Ohne Bildzahl oder Dauer laesst sich nichts pruefen -- dann wie bisher.
    if not (gemeldet and n and dauer > 0):
        return 0
    if abs(float(gemeldet) * dauer - n) <= 1.5:
        return 0
    echt = n / dauer
    print(min(NORM, key=lambda x: abs(math.log(float(Fraction(x)) / echt))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
