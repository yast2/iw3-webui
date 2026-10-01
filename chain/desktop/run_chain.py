#!/usr/bin/env python3
"""run_chain.py -- run_voll.sh als plattformneutrales Python (Linux-Server und Windows).

Portierung von run_voll.sh @ sha256 7919fd21d9a166c10019a79ff2b9f48224346d772eb27f5cae0605ebf1e88d54
(Produktionskette "G2EMA", eingefroren am 21.09.2026), nachgezogen bis sha256 12430e09
(24.09.2026: feste Bildrate, Bildtoleranz, relativer Tonversatz; TORCH_HOME bleibt
Server-Sache, dort laeuft uid 99 ohne Heimatordner). Stufen, Aufrufe und Parameter
sind dieselben; geaendert ist nur, wie sie gestartet werden:

  * keine Shell: jeder Aufruf ist eine Argumentliste, Pfade ueber os.path
  * "python3" ist der Interpreter, der dieses Skript ausfuehrt (sys.executable),
    oder die Umgebungsvariable PYTHON
  * nunif liegt unter NUNIF_DIR (Vorgabe /opt/nunif, auf Windows ..\\nunif neben chain\\)
  * ffmpeg/ffprobe fuer Stufe 1 und den Export kommen aus PATH; FFMPEG_DIR wird fuer
    die Kindprozesse vorangestellt
  * die Lieferkodierung ist schaltbar: --encoder hevc_qsv (Server, Vorgabe auf Linux),
    hevc_nvenc (Windows, Vorgabe dort), libx265 (nur fuer Pruefzwecke auf der CPU)
  * Abbruch (TERM/INT/HUP, auf Windows zusaetzlich BREAK) nimmt den ganzen Prozessbaum
    der laufenden Stufe mit
  * jede Befehlszeile geht nach $BASE/log/befehle.txt

Aufruf wie run_voll.sh (Vertrag aus ADR-022 wortgleich):

    run_chain.py --level <fast|economical|standard> -i <datei> -o <verzeichnis>
                 --work <verzeichnis> --gpu <n> [--flow] [--upscale]
                 [--stereo-format <fmt>] [-d=<divergenz>] [--encoder <enc>]

Umgebungsvariablen wie run_voll.sh: SRC BASE BIN D STUFE FLUSS HOCH MASTER_LOESCHEN
LESER GPU STEREO FFMPEG_QSV FFPROBE_QSV QSV_GERAET QSV_QUALITAET VGL NAME;
neu: NUNIF_DIR PYTHON FFMPEG_DIR ENCODER FFMPEG_ENC FFPROBE_ENC NVENC_CQ NVENC_PRESET,
IW3_COMPILE (trt|inductor|inductor-max: DepthPro ueber iw3_accel.py; leer = eager, Vorgabe).

Kantenfix (PFLICHT seit 24.09.2026 16:35, immer an und nicht abschaltbar; wie run_voll.sh
auf dem Server): der
nunif-Pin d23721f1 kehrt beim Import einer --export-YAML die Kantenweitung um
(iw3/utils.py 1720/1908). Der Warp laeuft deshalb immer ueber iw3_kantenfix.py neben
diesem Skript, das genau diese zwei Zeilen wie upstream 40d46847 im Prozess tauscht;
nunif selbst bleibt unberuehrt. Die Variante heisst <...>Kante (Lieferdatei
..._G2EMAKante_LRF.mp4). Kein Abschalten: IW3_KANTENFIX/KANTENFIX werden nicht gelesen,
--kein-kantenfix bricht ab, --kantenfix wird fuer alte Aufrufe still angenommen, und fehlt
iw3_kantenfix.py, bricht die Kette ab, statt ohne Fix zu warpen. Fast importiert nichts,
dort gibt es den Fehler nicht und der Fix entfaellt.

Wiederanlauf der Feinband-Stufe (25.09.2026): Der PC wurde am 25.09. um 01:07:45 mitten in
DepthPro heruntergefahren; beim Neustart um 08:49 rechnete die Stufe alle 17977 Bilder von
vorn, obwohl 15680 fertig in pro\\depth lagen (~70 min GPU). Liegen beim Start der Stufe schon
Tiefenbilder dort, wird jede bis IEND gelesen (PIL verify(), CRC jedes Blocks); was sich nicht
lesen laesst oder zu keinem Bild gehoert, wird geloescht, und iw3 laeuft mit --resume, rechnet
also nur Bilder ohne Tiefenkarte. Das ist sicher, weil der Bildordner-Export jedes Bild fuer
sich normiert (depth_model.disable_ema(), minmax je Bild; iw3/utils.py export_images). Danach
muss die Zahl der Tiefenbilder der Bildzahl entsprechen, sonst bricht die Kette ab.
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import time

WINDOWS = os.name == "nt"
SKRIPT = os.path.abspath(__file__)
if WINDOWS and hasattr(sys.stdout, "reconfigure"):
    # Pfade mit Umlauten; die Oberflaeche liest die Ausgabe als UTF-8.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------- Einstellungen
E = os.environ.get

SRC = E("SRC") or "/input/beispiel.mp4"
BASE = E("BASE") or "/output/_voll/_freeze/lauf"
VGL_VORGABE = "/output/bench/owl3d-vergleich"  # nur beim Aufruf von Hand (ohne -o)
BIN = E("BIN") or os.path.dirname(SKRIPT)
D = E("D") or "1.376"
STUFE = E("STUFE") or "standard"
FLUSS = E("FLUSS") or "0"
HOCH = E("HOCH") or "0"
# Leer heisst "nicht gesetzt" -- wie ${MASTER_LOESCHEN-} im Original.
MASTER_LOESCHEN = E("MASTER_LOESCHEN", "")
LESER = E("LESER") or "3"
GPU = E("GPU") or "0"
STEREO = E("STEREO") or "full_sbs"

FFMPEG_QSV = E("FFMPEG_QSV") or "/usr/lib/jellyfin-ffmpeg/ffmpeg"
FFPROBE_QSV = E("FFPROBE_QSV") or "/usr/lib/jellyfin-ffmpeg/ffprobe"
QSV_GERAET = E("QSV_GERAET") or "/dev/dri/renderD128"
QSV_QUALITAET = E("QSV_QUALITAET") or "17"

# NVENC: ❓ noch NICHT gegen die QSV-Zeile (SSIM 0,991196 bei 18,72 Mbit/s) geeicht --
# Kandidat aus dem Plan, Abschnitt 2d, Weg A. Bis zur Eichung ist cq ein Platzhalter.
NVENC_CQ = E("NVENC_CQ") or "18"
NVENC_PRESET = E("NVENC_PRESET") or "p5"
FFMPEG_ENC = E("FFMPEG_ENC") or "ffmpeg"
FFPROBE_ENC = E("FFPROBE_ENC") or "ffprobe"
ENCODER = E("ENCODER") or ("hevc_nvenc" if WINDOWS else "hevc_qsv")

# Kompiliertes Tiefenmodell (iw3_accel.py): trt | inductor | inductor-max. Leer/eager = der
# unveraenderte Weg ueber `python -m iw3`. Betrifft nur die DepthPro-Stufe.
KOMPILIERT = (E("IW3_COMPILE") or "").strip().lower()
if KOMPILIERT in ("eager", "0", "aus", "off"):
    KOMPILIERT = ""

NUNIF_DIR = E("NUNIF_DIR") or (
    os.path.join(os.path.dirname(os.path.dirname(SKRIPT)), "nunif") if WINDOWS else "/opt/nunif")
PYTHON = E("PYTHON") or sys.executable
FFMPEG_DIR = E("FFMPEG_DIR") or ""

OUT = ""
VGL = E("VGL") or ""
NAME = E("NAME") or ""
KANTENFIX = "1"   # Pflicht seit 24.09.2026, s. Kopf; nur Fast setzt ihn unten auf "0"

ENCODER_WAHL = ("hevc_qsv", "hevc_nvenc", "libx265")


def ausgabe(text=""):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def jetzt():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fehlt(arg):
    ausgabe(f"Argument {arg} braucht einen Wert")
    sys.exit(2)


# ------------------------------------------------------ Argumente wie im sh
def argumente(argv):
    global STUFE, SRC, OUT, BASE, GPU, STEREO, FLUSS, HOCH, MASTER_LOESCHEN, D, ENCODER
    i = 0
    while i < len(argv):
        a = argv[i]
        rest = len(argv) - i

        def wert():
            if rest < 2:
                fehlt(a)
            return argv[i + 1]

        if a == "--level":
            STUFE = wert(); i += 2
        elif a.startswith("--level="):
            STUFE = a[len("--level="):]; i += 1
        elif a in ("-i", "--input"):
            SRC = wert(); i += 2
        elif a.startswith("--src="):
            SRC = a[len("--src="):]; i += 1
        elif a in ("-o", "--output"):
            OUT = wert(); i += 2
        elif a == "--work":
            BASE = wert(); i += 2
        elif a.startswith("--base="):
            BASE = a[len("--base="):]; i += 1
        elif a == "--gpu":
            GPU = wert(); i += 2
        elif a == "--stereo-format":
            STEREO = wert(); i += 2
        elif a in ("--flow", "--fluss"):
            FLUSS = "1"; i += 1
        elif a == "--kein-fluss":
            FLUSS = "0"; i += 1
        elif a in ("--upscale", "--hochskalieren"):
            HOCH = "1"; i += 1
        elif a == "--kantenfix":
            i += 1   # Pflicht, s. Kopf -- bleibt nur fuer alte Aufrufe (pc_worker, bat)
        elif a == "--kein-kantenfix":
            ausgabe("### ABBRUCH: der Kantenfix ist Pflicht (seit 24.09.2026) -- abschalten geht nicht")
            sys.exit(2)
        elif a == "--master-weg":
            MASTER_LOESCHEN = "1"; i += 1
        elif a.startswith("-d="):
            D = a[len("-d="):]; i += 1
        elif a == "--encoder":
            ENCODER = wert(); i += 2
        elif a.startswith("--encoder="):
            ENCODER = a[len("--encoder="):]; i += 1
        elif a in ("-h", "--help"):
            ausgabe(__doc__.strip("\n"))
            sys.exit(0)
        else:
            ausgabe(f"unbekanntes Argument: {a}")
            sys.exit(2)


argumente(sys.argv[1:])

# ------------------------------------------------------------ Rezeptstufe
if STUFE == "standard":
    MODELL_FEIN, STAMM_VAR = "DepthPro", "G2EMA"
elif STUFE == "economical":
    MODELL_FEIN, STAMM_VAR = "DepthPro_S", "G5EMA"
elif STUFE == "fast":
    MODELL_FEIN, STAMM_VAR = "-", "SCHNELL"
else:
    ausgabe(f"unbekannte Stufe: {STUFE} (fast|economical|standard)")
    sys.exit(2)
if STUFE == "fast" and (FLUSS == "1" or HOCH == "1"):
    ausgabe("### ABBRUCH: --flow und --upscale sind Kettenstufen; 'fast' ist keine Kette")
    sys.exit(2)
KANTENFIX_PY = os.path.join(os.path.dirname(SKRIPT), "iw3_kantenfix.py")
# Fast importiert nichts, der Fix greift dort nicht: aus, damit die Variante (und der
# Dateiname) nicht etwas behauptet, das nie gerechnet wurde. Wie run_voll.sh.
if STUFE == "fast":
    KANTENFIX = "0"
if KANTENFIX == "1" and not os.path.isfile(KANTENFIX_PY):
    ausgabe(f"### ABBRUCH: {KANTENFIX_PY} fehlt")
    sys.exit(1)
if ENCODER not in ENCODER_WAHL:
    ausgabe(f"unbekannter Encoder: {ENCODER} ({'|'.join(ENCODER_WAHL)})")
    sys.exit(2)

# ------------------------------------------------- Stereoformat -> iw3-Schalter
SF_TABELLE = {
    "full_sbs": [],
    "half_sbs": ["--half-sbs"],
    "tb": ["--tb"],
    "half_tb": ["--half-tb"],
    "vr180": ["--vr180"],
    "cross_eyed": ["--cross-eyed"],
    "rgbd": ["--rgbd"],
    "half_rgbd": ["--half-rgbd"],
    "anaglyph": ["--anaglyph", "dubois"],
}
if STEREO not in SF_TABELLE:
    ausgabe(f"unbekanntes Stereoformat: {STEREO}")
    sys.exit(2)
SF = SF_TABELLE[STEREO]

VAR = STAMM_VAR
if FLUSS == "1":
    VAR += "Fluss"
if HOCH == "1":
    VAR += "4K"
if KANTENFIX == "1":
    VAR += "Kante"

# Wie `basename "$SRC" | sed 's/\.[^.]*$//'`: nur die letzte Endung ab, auch bei
# Namen ohne Stamm (".x" -> ""), anders als Path.stem.
STAMM = re.sub(r"\.[^.]*$", "", os.path.basename(SRC.rstrip("/\\") or SRC))
NAME = NAME or f"{STAMM}_{VAR}_LRF.mp4"

# Absolute Pfade, bevor irgendetwas startet. Das sh wechselt nach /opt/nunif und
# loest relative Pfade danach dort auf -- mkdir aber vorher im Aufrufverzeichnis.
# Hier gilt fuer alles das Aufrufverzeichnis. (Die Oberflaeche gibt ohnehin
# absolute Pfade.)
SRC = os.path.abspath(SRC)
BASE = os.path.abspath(BASE)
if not OUT:
    OUT = BASE
    VGL = VGL or VGL_VORGABE
    MASTER_LOESCHEN = MASTER_LOESCHEN or "0"
else:
    OUT = os.path.abspath(OUT)
    MASTER_LOESCHEN = MASTER_LOESCHEN or "1"

MASTER = os.path.join(BASE, f"master_{VAR}.mp4")
Z = os.path.join(BASE, "zeiten.tsv")
BEFEHLE = os.path.join(BASE, "log", "befehle.txt")

if not os.path.isfile(SRC):
    ausgabe(f"### ABBRUCH: Quelle nicht gefunden: {SRC}")
    sys.exit(1)
try:
    for d in (BASE, os.path.join(BASE, "log"), OUT):
        os.makedirs(d, exist_ok=True)
except OSError as e:
    print(f"mkdir: {e}", file=sys.stderr)
    sys.exit(1)
if VGL:
    try:
        os.makedirs(VGL, exist_ok=True)
    except OSError as e:
        print(f"mkdir: {e}", file=sys.stderr)
if not os.path.isdir(NUNIF_DIR):
    print(f"run_chain.py: cd: {NUNIF_DIR}: nicht gefunden (NUNIF_DIR)", file=sys.stderr)
    sys.exit(1)


def hardlink_pruefen():
    """umnummerieren.py verknuepft ohne Rueckfall; build_export_full.py faellt still
    auf Kopieren zurueck. Beides braucht ein Dateisystem mit Hardlinks -- auf Windows
    lokales NTFS, keine SMB-Freigabe. Vorher pruefen statt nach DepthPro scheitern."""
    a = os.path.join(BASE, ".hardlink-probe")
    b = a + "-2"
    try:
        with open(a, "w") as f:
            f.write("x")
        os.link(a, b)
        return True
    except OSError:
        return False
    finally:
        for p in (b, a):
            try:
                os.remove(p)
            except OSError:
                pass


if not hardlink_pruefen():
    ausgabe(f"### ABBRUCH: {BASE} kann keine Hardlinks -- Arbeitsverzeichnis muss lokal sein (NTFS/ext4/btrfs)")
    sys.exit(1)

if not os.path.isfile(Z):
    with open(Z, "w", encoding="utf-8", newline="\n") as f:
        f.write("stufe\tvariante\tbeginn\twall_s\ts_pro_bild\tbemerkung\n")

# ------------------------------------------------ Umgebung der Kindprozesse
KIND_UMGEBUNG = dict(os.environ)
KIND_UMGEBUNG["PYTHONPATH"] = os.pathsep.join(
    p for p in (NUNIF_DIR, os.environ.get("PYTHONPATH", "")) if p)
if FFMPEG_DIR:
    KIND_UMGEBUNG["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
if WINDOWS:
    KIND_UMGEBUNG.setdefault("PYTHONUTF8", "1")

# ---------------------------------------------------- Punkt 5: Stufenmarken
if STUFE == "fast":
    STUFEN = 1
else:
    STUFEN = 7
    if FLUSS == "1":
        STUFEN += 1
    if HOCH == "1":
        STUFEN += 1
I = 0


def naechste():
    global I
    I += 1


def schritt(text):
    ausgabe(f"=== {I}/{STUFEN} {text}  {jetzt()}")


def s_pro_bild(wall, n):
    try:
        n = float(n)
    except ValueError:
        n = 0.0
    return "%.4f" % (float(wall) / n) if n > 0 else "-"


def zeit(stufe, wall, n, bemerkung):
    with open(Z, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{stufe}\t{VAR}\t{jetzt()}\t{wall}\t{s_pro_bild(wall, n)}\t{bemerkung}\n")
    ausgabe(f"    {stufe}: {wall}s")


def uhr():
    return int(time.time())


def zaehle(ordner):
    """Wie `ls ORDNER | wc -l`: alle Eintraege ausser Punktdateien."""
    try:
        return len([e for e in os.listdir(ordner) if not e.startswith(".")])
    except OSError:
        print(f"ls: cannot access '{ordner}': No such file or directory", file=sys.stderr)
        return 0


def schreibe(pfad, text):
    with open(pfad, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def lies(pfad):
    with open(pfad, encoding="utf-8") as f:
        return f.read().rstrip("\n")


def beruehre(pfad):
    with open(pfad, "a"):
        pass
    os.utime(pfad, None)


def verschiebe(quelle, ziel):
    try:
        os.replace(quelle, ziel)
    except OSError:
        shutil.copy2(quelle, ziel)
        os.remove(quelle)


def rechte_99_100(pfad):
    # chown 99:100 wie im sh; auf Windows gibt es das nicht, ohne Rechte scheitert es still.
    try:
        os.chown(pfad, 99, 100)
    except (OSError, AttributeError):
        pass


def erstes(ordner, passt):
    """Wie `find ORDNER … | head -1`, aber in fester (sortierter) Reihenfolge."""
    for wurzel, verz, dateien in os.walk(ordner):
        verz.sort()
        for name in sorted(verz) + sorted(dateien):
            p = os.path.join(wurzel, name)
            if passt(p, name):
                return p
    return ""


def tiefenbilder_pruefen(ordner, bilder):
    """Wiederanlauf der Feinband-Stufe (s. Kopf): vorhandene Tiefenbilder pruefen, bevor iw3
    --resume sie stehen laesst. Liefert (behalten, geloescht), oder None ohne PIL -- dann
    rechnet die Stufe wie bisher alles neu."""
    try:
        from PIL import Image
    except ImportError:
        return None
    namen = {os.path.splitext(e)[0] for e in os.listdir(bilder)}
    behalten = geloescht = 0
    for e in os.scandir(ordner):
        if e.name.startswith(".") or not e.is_file():
            continue
        try:
            if os.path.splitext(e.name)[0] not in namen:
                raise ValueError("kein Bild dazu")
            with Image.open(e.path) as bild:
                bild.verify()
            behalten += 1
        except Exception:
            os.remove(e.path)
            geloescht += 1
    return behalten, geloescht


# ------------------------------------------- Punkt 7: Abbruch nimmt die Stufe mit
KIND = None


def _baum_beenden(proc, hart):
    try:
        import psutil
        try:
            eltern = psutil.Process(proc.pid)
            alle = eltern.children(recursive=True) + [eltern]
        except psutil.NoSuchProcess:
            return
        for p in alle:
            try:
                p.kill() if hart else p.terminate()
            except psutil.NoSuchProcess:
                pass
        return
    except ImportError:
        pass
    if WINDOWS:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.kill(proc.pid, signal.SIGKILL if hart else signal.SIGTERM)
        except OSError:
            pass


def abbruch(signum=None, frame=None):
    ausgabe("")
    ausgabe(f"### ABBRUCH: Signal empfangen {jetzt()}")
    proc = KIND
    if proc is not None:
        _baum_beenden(proc, hart=False)
        n = 0
        while n < 10 and proc.poll() is None:
            time.sleep(1)
            n += 1
        _baum_beenden(proc, hart=True)
    os._exit(143)


for _name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGBREAK"):
    if hasattr(signal, _name):
        try:
            signal.signal(getattr(signal, _name), abbruch)
        except (OSError, ValueError):
            pass


def befehl_protokollieren(cmd):
    with open(BEFEHLE, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{jetzt()}\t" + "\t".join(cmd) + "\n")


def programm(name):
    """Programmname wie im sh ueber PATH aufloesen -- aber ueber das PATH der
    Kindprozesse. Windows' CreateProcess saehe sonst nur das eigene PATH."""
    if os.path.dirname(name):
        return name
    return shutil.which(name, path=KIND_UMGEBUNG.get("PATH")) or name


def lauf(log, cmd):
    """Startet eine Stufe, merkt sich den Prozess fuer den Abbruch und wartet.
    log "-" = Ausgabe durchreichen, sonst stdout+stderr in die Datei."""
    global KIND
    befehl_protokollieren(cmd)
    start = [programm(cmd[0])] + cmd[1:]
    extra = {}
    if WINDOWS:
        # Eigene Gruppe: Strg+C/Strg+Pause treffen nur dieses Skript, das dann den
        # Baum der Stufe gezielt beendet (Punkt 7).
        extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    if log == "-":
        KIND = subprocess.Popen(start, cwd=NUNIF_DIR, env=KIND_UMGEBUNG, **extra)
    else:
        with open(log, "wb") as f:
            KIND = subprocess.Popen(start, cwd=NUNIF_DIR, env=KIND_UMGEBUNG,
                                    stdout=f, stderr=subprocess.STDOUT, **extra)
    # Mit Zeitschranke warten: auf Windows unterbricht ein Signal ein
    # unbefristetes wait() nicht.
    while True:
        try:
            rc = KIND.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            continue
    KIND = None
    if rc != 0 and log != "-":
        # 01.10.: exit 1 aus dem Export stand nur in log/exp_*.txt auf dem PC -- das Job-Log
        # auf dem Server endete stumm bei "=== 5/7". Die letzten Zeilen gehoeren nach stdout.
        ausgabe(f"### Stufe endete mit {rc}: {log} -- letzte Zeilen:")
        try:
            with open(log, "rb") as f:
                f.seek(max(0, os.path.getsize(log) - 8192))
                rest = f.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
            for z in [z for z in rest if z.strip()][-15:]:
                ausgabe("    " + z[:300])
        except OSError as e:
            ausgabe(f"    (Log nicht lesbar: {e})")
    return rc


def direkt(cmd):
    """Aufruf im Vordergrund ohne lauf() -- wie umnummerieren.py im sh."""
    befehl_protokollieren(cmd)
    return subprocess.call([programm(cmd[0])] + cmd[1:], cwd=NUNIF_DIR, env=KIND_UMGEBUNG)


def einfangen(cmd):
    befehl_protokollieren(cmd)
    r = subprocess.run([programm(cmd[0])] + cmd[1:], cwd=NUNIF_DIR, env=KIND_UMGEBUNG,
                       stdout=subprocess.PIPE, text=True)
    return r.stdout.strip()


def awk_zahl(s):
    """awk wandelt einen String ueber seinen fuehrenden Zahlanteil, sonst 0."""
    m = re.match(r"\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", s or "")
    return float(m.group(0)) if m else 0.0


def ganzzahl(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


P = PYTHON
ausgabe(f"### run_chain.py  Stufe={STUFE}  Variante={VAR}  {jetzt()}")
ausgabe(f"    Quelle      {SRC}")
ausgabe(f"    Arbeit      {BASE}")
ausgabe(f"    Ausgabe     {OUT}")
ausgabe(f"    Divergenz   {D}")
ausgabe(f"    GPU         {GPU}")
ausgabe(f"    Stereo      {STEREO}")
ausgabe(f"    Fluss       {'AN' if FLUSS == '1' else 'aus'}")
ausgabe(f"    4K          {'AN' if HOCH == '1' else 'aus'}")
ausgabe(f"    Kantenfix   {'AN (nunif 40d46847, nur Warp)' if KANTENFIX == '1' else 'entfaellt (fast importiert nichts)'}")
ausgabe(f"    Lieferdatei {NAME}")
ausgabe(f"    Encoder     {ENCODER}")
if KOMPILIERT:
    ausgabe(f"    Kompiliert  {KOMPILIERT} ({MODELL_FEIN}, iw3_accel.py)")

# ------------------------------------------ 0  Feste Bildrate, nur wenn noetig
# 24.09.: Handyaufnahme (Android, gemeldet 120 fps, echt 30,3) -- ffmpeg zog 6326 echte
# Bilder, iw3 fuellte fuer VDA auf die gemeldeten 120 auf (25038), und der Export
# haette 120 fps ins yml geschrieben. Dieselbe Falle, nur kleiner, steckt in 16
# 4K60-Studiodateien (2-8 Bilder mehr, als 60 fps x Dauer fassen). Solche Quellen einmal
# auf die naechste Normrate bringen (Bilder doppeln/auslassen, Ton kopiert); danach
# sehen alle Stufen dieselbe Quelle. bildrate_pruefen.py schweigt im Normalfall.
if not os.path.isfile(os.path.join(BASE, "cfr.ok")):
    cmd = [P, os.path.join(BIN, "bildrate_pruefen.py"), SRC]
    befehl_protokollieren(cmd)
    r = subprocess.run([programm(cmd[0])] + cmd[1:], cwd=NUNIF_DIR, env=KIND_UMGEBUNG,
                       stdout=subprocess.PIPE, text=True)
    if r.returncode != 0:
        sys.exit(1)
    ziel = r.stdout.rstrip("\n")
    if ziel:
        if os.path.isfile(os.path.join(BASE, "frames.ok")):
            ausgabe(f"### ABBRUCH: {BASE} stammt von vor der Bildratenpruefung -- Arbeitsordner leeren")
            sys.exit(1)
        ausgabe(f"    Bildrate: gemeldet passt nicht zur Bildzahl -> fest {ziel} fps (x264 crf 10)")
        t0 = uhr()

        def cfr(ton):
            return lauf("-", ["ffmpeg", "-v", "error", "-y", "-i", SRC, "-map", "0:v:0",
                              "-map", "0:a:0?", "-fps_mode", "cfr", "-r", ziel,
                              "-c:v", "libx264", "-crf", "10", "-preset", "veryfast",
                              "-pix_fmt", "yuv420p"] + ton +
                        ["-movflags", "+faststart", os.path.join(BASE, "quelle_cfr.mp4")])
        if cfr(["-c:a", "copy"]) != 0 and cfr(["-c:a", "aac", "-b:a", "256k"]) != 0:
            sys.exit(1)
        zeit("bildrate", uhr() - t0, 0, f"fest {ziel} fps")
    schreibe(os.path.join(BASE, "cfr.ok"), f"{ziel}\n")
# Auf den Inhalt pruefen, nicht auf die Dateigroesse: "leer" ist hier ein Zeilenumbruch.
if lies(os.path.join(BASE, "cfr.ok")):
    SRC = os.path.join(BASE, "quelle_cfr.mp4")
    ausgabe(f"    Quelle fest {lies(os.path.join(BASE, 'cfr.ok'))} fps: {SRC}")

# ===========================================================================
# Stufe "fast" -- ein einzelner iw3-Durchlauf, keine Kette.
# ===========================================================================
if STUFE == "fast":
    naechste(); schritt("VDA_B + EMA, ein Durchlauf")
    t0 = uhr()
    shutil.rmtree(os.path.join(BASE, "fast"), ignore_errors=True)
    os.makedirs(os.path.join(BASE, "fast"), exist_ok=True)
    rc = lauf(os.path.join(BASE, "log", "fast.txt"), [
        P, "-m", "iw3", "-i", SRC, "-o", os.path.join(BASE, "fast"), "-y",
        "--gpu", GPU, "--depth-model", "VDA_B", "--divergence", D, "--convergence", "0.5",
        "--convergence-mode", "sod_v1", "--foreground-scale", "0", "--edge-dilation", "2", "1",
        "--video-codec", "libx265", "--pix-fmt", "yuv420p", "--max-fps", "1000", "--scene-detect",
        "--ema-normalize", "--ema-decay", "0.75", "--ema-buffer", "30"] + SF)
    if rc != 0:
        sys.exit(1)
    f = erstes(os.path.join(BASE, "fast"), lambda p, n: n.endswith(".mp4"))
    if not f:
        ausgabe("### ABBRUCH: keine iw3-Ausgabe")
        sys.exit(1)
    try:
        verschiebe(f, os.path.join(OUT, NAME))
    except OSError as e:
        print(f"mv: {e}", file=sys.stderr)
        sys.exit(1)
    rechte_99_100(os.path.join(OUT, NAME))
    zeit("fast", uhr() - t0, 0, "ein Durchlauf")
    ausgabe(f"### ALLES FERTIG -> {os.path.join(OUT, NAME)}  {jetzt()}")
    sys.stdout.write(open(Z, encoding="utf-8").read()); sys.stdout.flush()
    sys.exit(0)

# ------------------------------------------------------------ 1  Bilder
naechste()
if not os.path.isfile(os.path.join(BASE, "frames.ok")):
    schritt("Bilder extrahieren")
    t0 = uhr()
    os.makedirs(os.path.join(BASE, "frames"), exist_ok=True)
    # Ohne -fps_mode passthrough verdoppelt ffmpeg still einzelne Bilder.
    if lauf("-", ["ffmpeg", "-v", "error", "-i", SRC, "-fps_mode", "passthrough",
                  "-qscale:v", "2", os.path.join(BASE, "frames", "%08d.jpg")]) != 0:
        sys.exit(1)
    n = zaehle(os.path.join(BASE, "frames"))
    schreibe(os.path.join(BASE, "frames.ok"), f"{n}\n")
    zeit("bilder", uhr() - t0, n, f"{n} Bilder")
N = lies(os.path.join(BASE, "frames.ok"))
ausgabe(f"    Bilder: {N}")
N_ZAHL = ganzzahl(N)

# ------------------------------------------------ 2  Hochskalieren (optional)
RGB = os.path.join(BASE, "frames")
if HOCH == "1":
    naechste()
    if not os.path.isfile(os.path.join(BASE, "hoch.ok")):
        schritt("waifu2x noise_scale2x (GPU)")
        t0 = uhr()
        os.makedirs(os.path.join(BASE, "rgb2x"), exist_ok=True)
        if lauf(os.path.join(BASE, "log", "hoch.txt"), [
                P, "-m", "waifu2x.cli", "--style", "photo",
                "--method", "noise_scale2x", "-i", os.path.join(BASE, "frames"),
                "-o", os.path.join(BASE, "rgb2x"),
                "--format", "png", "--tile-size", "160", "-g", GPU]) != 0:
            sys.exit(1)
        n = zaehle(os.path.join(BASE, "rgb2x"))
        if n != N_ZAHL:
            ausgabe(f"### ABBRUCH: waifu2x {n} Bilder, Quelle {N}")
            sys.exit(1)
        schreibe(os.path.join(BASE, "hoch.ok"), f"{n}\n")
        zeit("hochskalieren", uhr() - t0, n, "noise_scale2x, Kachel 160")
    RGB = os.path.join(BASE, "rgb2x")
    ausgabe(f"    RGB 2x: {lies(os.path.join(BASE, 'hoch.ok'))}")

# ---------------------------------------------------------- 3  Feinband
naechste()
if not os.path.isfile(os.path.join(BASE, "pro.ok")):
    schritt(MODELL_FEIN)
    t0 = uhr()
    tiefe = os.path.join(BASE, "pro", "depth")
    weiter = []
    if os.path.isdir(tiefe) and zaehle(tiefe):
        geprueft = tiefenbilder_pruefen(tiefe, os.path.join(BASE, "frames"))
        if geprueft is not None:
            weiter = ["--resume"]
            ausgabe(f"    Wiederanlauf: {geprueft[0]} Tiefenbilder lesbar und behalten, "
                    f"{geprueft[1]} geloescht; iw3 --resume rechnet den Rest  {jetzt()}")
    iw3_fein = [P, os.path.join(BIN, "iw3_accel.py")] if KOMPILIERT else [P, "-m", "iw3"]
    if lauf(os.path.join(BASE, "log", "pro.txt"), iw3_fein + [
            "--export", "--export-depth-only",
            "--depth-model", MODELL_FEIN, "--gpu", GPU,
            "-i", os.path.join(BASE, "frames"), "-o", os.path.join(BASE, "pro"), "--yes"] + weiter) != 0:
        sys.exit(1)
    n = zaehle(tiefe)
    if n != N_ZAHL:
        ausgabe(f"### ABBRUCH: {MODELL_FEIN} {n} Tiefenbilder, Quelle {N}")
        sys.exit(1)
    schreibe(os.path.join(BASE, "pro.ok"), f"{n}\n")
    zeit("feinband", uhr() - t0, n, f"{n} Tiefenbilder, {MODELL_FEIN}"
         + (f" {KOMPILIERT}" if KOMPILIERT else "")
         + (f", Wiederanlauf mit {geprueft[0]} vorhandenen" if weiter else ""))
ausgabe(f"    Tiefe {MODELL_FEIN}: {lies(os.path.join(BASE, 'pro.ok'))}")

def vda_abgleich(roh):
    """iw3 schickt im Videopfad jedes Bild durch ffmpegs fps-Filter, Stufe 1 nicht. Der doppelt ein Bild, wo die Uhr
    der Datei eine halbe Bilddauer gegen die Nennrate verloren hat -- am Ende (23.09.: 1801 gegen 1800) oder mitten
    im Film (30.09., Server, 60-fps-Film: Karte 40080 = Bild 40079); pendelt die Drift dort, doppelt er und laesst
    abwechselnd aus (01.10., .mov-Quelle: 18121 gegen 18120, 5 Doppelungen und 4 Auslassungen bei Bild 9004-9014;
    vda_ende_pruefen.py brach ab: "Versatz 0 in 0/6"). vda_abgleich.py dekodiert die Quelle genau wie iw3, weiss damit,
    welche Karte welches Bild traegt, prueft das am Inhalt (Bewegung gegen DepthPro, Ortsprobe an den Doppelungen)
    und verlinkt vda/ erst dann neu -- wie iw3:/config/chain/vda_abgleich.py auf dem Server. Loest
    vda_ende_pruefen.py ab, das nur das Ende kannte und die Lage aus der Bewegung erraten musste.
    interne Messnotiz"""
    return direkt([P, os.path.join(BIN, "vda_abgleich.py"), SRC, os.path.join(os.path.dirname(roh), "iw3_export.yml"),
                   roh, os.path.join(BASE, "vda"), os.path.join(BASE, "pro", "depth"), os.path.join(BASE, "frames"),
                   str(N_ZAHL)]) == 0


def fehlendes_vda_bild_am_ende(roh):
    """Gegenstueck zu vda_abgleich() fuer ein fehlendes Bild am Ende: iw3 schickt im Videopfad jedes Bild durch
    ffmpegs fps-Filter (nunif/utils/video.py hook_frame -> FixedFPSFilter, "fps=<Rate>"), Stufe 1
    zieht mit -fps_mode passthrough. Am Clipende kann der Filter ein Bild verdoppeln (23.09.) oder
    verschlucken (26.09., Job: 27499 gegen 27500 nach 8,5 h Vorarbeit). Aufgefuellt wird nur,
    wenn die Lage bewiesen ist:
      1. VDAs eigene Szenengrenzen (iw3_export.yml) liegen auf den Schnitten in frames/ (Versatz 0)
         -- jeder eindeutige Schnitt, die letzten 12 Grenzen und die ersten 3;
      2. Bewegung VDA gegen DepthPro: Versatz 0 vor beiden Nachbarn in den 6 bewegtesten Fenstern
         ueber den Clip und den 2 bewegtesten im letzten Viertel (Schwellen wie vda_ende_pruefen.py);
      3. das letzte Quellbild ist kein Schnitt (sonst stuende dort eine fremde Tiefe).
    Dann wird die letzte VDA-Karte als Bild N verlinkt. Rueckgabe True = aufgefuellt."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        ausgabe("    VDA-Ende: numpy/PIL fehlen -- nicht pruefbar")
        return False
    vda_dir, pro_dir, fr = os.path.join(BASE, "vda"), os.path.join(BASE, "pro", "depth"), os.path.join(BASE, "frames")
    n = N_ZAHL
    vda = sorted(f for f in os.listdir(vda_dir) if f.endswith(".png"))
    pro = sorted(f for f in os.listdir(pro_dir) if f.endswith(".png"))
    if len(vda) != n - 1 or len(pro) != n or n < 240 or vda[-1] != f"{n - 1:08d}.png":
        ausgabe(f"    VDA-Ende: nicht pruefbar (VDA {len(vda)}, DepthPro {len(pro)}, Quelle {n})")
        return False

    def grau(i):  # frames/ ist 1-basiert
        with Image.open(os.path.join(fr, f"{i:08d}.jpg")) as b:
            return np.asarray(b.convert("L").resize((240, 135), Image.BILINEAR), dtype=np.float32)

    def sprung(i):  # Bild i -> i+1
        return float(np.abs(grau(i + 1) - grau(i)).mean())

    # 1. Szenengrenzen: VDA-Index b (0-basiert) = erstes Bild der neuen Szene = Schnitt jpg b -> b+1.
    yml = os.path.join(os.path.dirname(roh), "iw3_export.yml")
    grenzen = []
    try:
        m = re.search(r"scene_boundary:\s*'?([0-9,\s]+)", open(yml, encoding="utf-8").read())
        grenzen = sorted({int(x) for x in m.group(1).replace("\n", "").split(",") if x.strip()}) if m else []
    except OSError:
        pass
    grenzen = [b for b in grenzen if 3 <= b <= n - 4]
    klar, schief = [], []
    for b in sorted(set(grenzen[:3] + grenzen[-12:])):
        d = {i: sprung(i) for i in range(b - 2, b + 3)}
        folge = sorted(d.values(), reverse=True)
        i = max(d, key=d.get)
        if folge[0] >= 8 and folge[0] >= 4 * max(folge[1], 0.5):
            (klar if i == b else schief).append(f"{b}{i - b:+d}")
    if schief:
        ausgabe(f"    VDA-Ende: Szenengrenzen verschoben {schief} -- Bild fehlt nicht am Ende")
        return False

    # 2. Bewegung
    groesse = Image.open(os.path.join(vda_dir, vda[0])).size

    def karte(pfad):
        with Image.open(pfad) as bild:
            if bild.size != groesse:
                bild = bild.resize(groesse, Image.BILINEAR)
            return np.asarray(bild, dtype=np.float32).ravel()

    def korr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))

    SCHRITT, LAENGE = 4, 30
    rand = max(1, n // 50)
    letzt = n - LAENGE - SCHRITT - 4
    kandidaten = sorted(set(np.linspace(rand, letzt, 20).astype(int))
                        | set(np.linspace(n * 3 // 4, letzt, 8).astype(int)))
    # Bewegung im BILD messen, nicht in der Tiefe: DepthPro normiert jedes Bild fuer sich, auf
    # Schwarz wird daraus Rauschen, das wie Bewegung aussieht (Job: schwarzes Ende 18881
    # gegen 12946-32233 in echten Szenen, Korrelation dort ~0). pro[s] gehoert zu jpg s+1.
    bewegung = {s: float(np.abs(grau(s + 1 + LAENGE) - grau(s + 1)).mean()) for s in kandidaten}
    rang = [s for s in sorted(kandidaten, key=bewegung.get, reverse=True) if bewegung[s] >= 2]
    fenster = sorted(set(rang[:6] + [s for s in rang if s >= n * 3 // 4][:2]))
    siege, m0, mn = 0, [], []
    for s in fenster:
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
    zeile = (f"Schnitte auf Versatz 0: {len(klar)} ({', '.join(klar[-3:]) or '-'}); Bewegung Versatz 0 in "
             f"{siege}/{len(fenster)} Fenstern vorn (letztes ab {fenster[-1]}), Mittel {a:+.3f} gegen {b:+.3f}")
    if siege != len(fenster) or len(fenster) < 6 or a - b < 0.03:
        ausgabe(f"    VDA: 1 Bild zu wenig, Lage nicht eindeutig -- {zeile}")
        return False

    # 3. Letztes Quellbild darf kein Schnitt sein.
    s_ende = sprung(n - 1)
    if s_ende >= 8:
        ausgabe(f"    VDA: 1 Bild zu wenig, aber Bild {n} ist ein Schnitt (Sprung {s_ende:.1f}) -- {zeile}")
        return False
    ziel = os.path.join(vda_dir, f"{n:08d}.png")
    try:
        os.link(os.path.join(vda_dir, vda[-1]), ziel)
    except OSError:
        shutil.copy2(os.path.join(vda_dir, vda[-1]), ziel)
    ausgabe(f"    VDA: 1 Bild zu wenig am Ende -- {n:08d}.png = Karte von {vda[-1]} "
            f"(Sprung {n - 1}->{n}: {s_ende:.2f}) -- {zeile}")
    return True


def vda_drehung():
    """Stufe 1 (ffmpeg) dreht eine Quelle mit Display-Matrix automatisch, iw3 liest das Video in Stufe 4 ueber PyAV
    ungedreht. 01.10., .mov-Quelle (.mov, rotation -90): frames/ und DepthPro hochkant 2160x3840, VDA-Karten
    quer 686x392 -- Korrelation VDA gegen DepthPro ~0, im Uhrzeigersinn gedreht 0,86-0,98. Der Bandtausch haette
    zwei Tiefen verschiedener Lage gemischt. iw3 dreht mit --rotate-right/--rotate-left vor der Tiefenschaetzung
    (iw3/utils.py preprocess_image, auch im VDA-Export). Rueckgabe: Zusatzargumente fuer iw3; None = Abbruch."""
    roh = einfangen(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "stream_side_data=rotation", "-of", "csv=p=0", SRC]).replace(",", "").split()
    w = roh[0] if roh else "0"
    try:
        grad = int(round(float(w))) % 360
    except ValueError:
        grad = None
    if grad == 0:
        return []
    if grad == 270:  # ffprobe -90: im Uhrzeigersinn drehen, wie ffmpegs Autorotation (transpose=clock)
        ausgabe(f"    Quelle gedreht (Display-Matrix {w} Grad): VDA mit --rotate-right wie Stufe 1")
        return ["--rotate-right"]
    if grad == 90:
        ausgabe(f"    Quelle gedreht (Display-Matrix {w} Grad): VDA mit --rotate-left wie Stufe 1")
        return ["--rotate-left"]
    ausgabe(f"### ABBRUCH: Quelle mit Display-Matrix {w} Grad -- iw3 kann das in Stufe 4 nicht nachbilden")
    return None


def gleiche_lage(roh):
    """VDA-Karten und frames/ muessen gleich liegen (hoch/quer); sonst stimmt die Drehung nicht."""
    try:
        from PIL import Image
    except ImportError:
        return True
    k = erstes(roh, lambda p, n: n.endswith(".png"))
    b = erstes(os.path.join(BASE, "frames"), lambda p, n: n.endswith(".jpg"))
    if not k or not b:
        return True
    with Image.open(k) as a, Image.open(b) as c:
        kw, kh, bw, bh = a.size + c.size
    if kw == kh or bw == bh or (kw > kh) == (bw > bh):
        return True
    ausgabe(f"### ABBRUCH: VDA-Karten {kw}x{kh}, Bilder {bw}x{bh} -- Tiefe liegt quer zum Bild")
    return False


# ------------------------------------------------- 4  VDA_B + EMA (Grobband)
# 🔴 Muss ueber das VIDEO laufen, nicht ueber frames/ (--ema-normalize wirkt nur im
# Videopfad, WP8).
naechste()
if not os.path.isfile(os.path.join(BASE, "vda.ok")):
    schritt("VDA_B + EMA")
    t0 = uhr()
    # Wiederanlauf nach einem Abbruch in dieser Stufe: umnummerieren.py ueberspringt
    # vorhandene Ziele, alte Karten blieben also stehen -- beide Ordner neu anlegen.
    shutil.rmtree(os.path.join(BASE, "vda_roh"), ignore_errors=True)
    shutil.rmtree(os.path.join(BASE, "vda"), ignore_errors=True)
    DREHUNG = vda_drehung()
    if DREHUNG is None:
        sys.exit(1)
    if lauf(os.path.join(BASE, "log", "vda.txt"), [
            P, "-m", "iw3", "--export", "--export-depth-only",
            "--convergence", "0.5", "--foreground-scale", "0", "--edge-dilation", "2", "1",
            "--pix-fmt", "yuv420p", "--max-fps", "1000", "--scene-detect",
            "--ema-normalize", "--ema-decay", "0.75", "--ema-buffer", "30",
            "--depth-model", "VDA_B", "--divergence", D, "--gpu", GPU] + DREHUNG + [
            "-i", SRC, "-o", os.path.join(BASE, "vda_roh"), "--yes"]) != 0:
        sys.exit(1)
    roh = erstes(os.path.join(BASE, "vda_roh"), lambda p, n: n == "depth" and os.path.isdir(p))
    if not gleiche_lage(roh):
        sys.exit(1)
    # VDA nummeriert die Videoausgabe ab 0, DepthPro den Bildordner ab 1.
    if direkt([P, os.path.join(BIN, "umnummerieren.py"), roh, os.path.join(BASE, "vda"), "1"]) != 0:
        sys.exit(1)
    n = zaehle(os.path.join(BASE, "vda"))
    if n > N_ZAHL and vda_abgleich(roh):
        n = zaehle(os.path.join(BASE, "vda"))
    elif n == N_ZAHL - 1 and fehlendes_vda_bild_am_ende(roh):
        n = zaehle(os.path.join(BASE, "vda"))
    if n != N_ZAHL:
        ausgabe(f"### ABBRUCH: VDA {n} Bilder, Quelle {N}")
        sys.exit(1)
    schreibe(os.path.join(BASE, "vda.ok"), f"{n}\n")
    zeit("vda_b_ema", uhr() - t0, n, f"{n} Tiefenbilder")
ausgabe(f"    Tiefe VDA_B+EMA: {lies(os.path.join(BASE, 'vda.ok'))}")

# ------------------------------------------------------- 5  Bandtausch (CPU)
naechste()
if not os.path.isfile(os.path.join(BASE, "band.ok")):
    schritt("Bandtausch r=30 linear (CPU)")
    t0 = uhr()
    if lauf(os.path.join(BASE, "log", "band.txt"), [
            P, os.path.join(BIN, "stufe2_band_schnell.py"),
            os.path.join(BASE, "vda"), os.path.join(BASE, "pro", "depth"), os.path.join(BASE, "band"),
            "--radius", "30", "--form", "linear", "--lese-faeden", LESER]) != 0:
        sys.exit(1)
    n = zaehle(os.path.join(BASE, "band"))
    if n != N_ZAHL:
        ausgabe(f"### ABBRUCH: Bandtausch {n} Bilder, Quelle {N}")
        sys.exit(1)
    schreibe(os.path.join(BASE, "band.ok"), f"{n}\n")
    zeit("bandtausch", uhr() - t0, n, f"{LESER} Lesefaeden")
ausgabe(f"    Bandtausch: {lies(os.path.join(BASE, 'band.ok'))}")

# --------------------------------------------------- 6  Fluss (optional, aus)
TIEFE = os.path.join(BASE, "band")
if FLUSS == "1":
    naechste()
    if not os.path.isfile(os.path.join(BASE, "fluss.ok")):
        schritt("Fluss RAFT fp16 (GPU)")
        t0 = uhr()
        if lauf(os.path.join(BASE, "log", "fluss.txt"), [
                P, os.path.join(BIN, "stufe3_fluss.py"),
                os.path.join(BASE, "band"), os.path.join(BASE, "frames"), os.path.join(BASE, "fluss"),
                "--photo-schwelle", "0.03", "--halb"]) != 0:
            sys.exit(1)
        n = zaehle(os.path.join(BASE, "fluss"))
        if n != N_ZAHL:
            ausgabe(f"### ABBRUCH: Fluss {n} Bilder, Quelle {N}")
            sys.exit(1)
        schreibe(os.path.join(BASE, "fluss.ok"), f"{n}\n")
        zeit("fluss", uhr() - t0, n, "RAFT fp16")
    TIEFE = os.path.join(BASE, "fluss")
    ausgabe(f"    Fluss: {lies(os.path.join(BASE, 'fluss.ok'))}")
else:
    ausgabe("    Fluss: aus (Vorgabe)")

# ------------------------------------------------------------- 7  Export
EXP = os.path.join(BASE, f"exp_{VAR}")
naechste()
if not os.path.isfile(os.path.join(BASE, f"exp_{VAR}.ok")):
    schritt("Export")
    t0 = uhr()
    if lauf(os.path.join(BASE, "log", f"exp_{VAR}.txt"), [
            P, os.path.join(BIN, "build_export_full.py"),
            SRC, TIEFE, EXP,
            "--rgb-dir", RGB, "--mapper", "mul_1+mul_2=0.5", "--pad", "0"]) != 0:
        sys.exit(1)
    # Stimmt die Paarzahl nicht exakt, laeuft das Bild gegen den Ton.
    paare = zaehle(os.path.join(EXP, "rgb"))
    if paare != N_ZAHL:
        ausgabe(f"### ABBRUCH: {paare} Paare, {N} Quellbilder -- Ton wuerde verrutschen")
        sys.exit(1)
    beruehre(os.path.join(BASE, f"exp_{VAR}.ok"))
    zeit("export", uhr() - t0, paare, f"{paare} Paare")

# --------------------------------------------------------------- 8  Warp
# Zwischenstand verlustfrei (x264 crf 0); die Lieferkodierung ist ein zweiter Lauf.
naechste()
if not os.path.isfile(os.path.join(BASE, f"warp_{VAR}.ok")):
    schritt("Warp mlbw_l2 (GPU)")
    t0 = uhr()
    # --kein-faststart: der Zwischenstand wird gleich neu kodiert; faststart schob ihn nur ein
    # zweites Mal durch die Platte (Job, 104 GB: ~38 min). Die Lieferdatei behaelt es.
    iw3_warp = [P, KANTENFIX_PY, "--kein-faststart"] if KANTENFIX == "1" else [P, "-m", "iw3"]
    if lauf(os.path.join(BASE, "log", f"warp_{VAR}.txt"), iw3_warp + [
            "-i", os.path.join(EXP, "iw3_export.yml"), "-o", os.path.join(BASE, f"out_{VAR}"),
            "--method", "mlbw_l2", "-d", D, "-c", "0.5", "--convergence-mode", "sod_v1",
            "--synthetic-view", "right", "--edge-dilation", "2", "1", "--max-fps", "1000",
            "--gpu", GPU,
            "--video-codec", "libx264", "--crf", "0", "--preset", "ultrafast", "--pix-fmt", "yuv420p",
            "--yes"] + SF) != 0:
        sys.exit(1)
    f = erstes(os.path.join(BASE, f"out_{VAR}"), lambda p, n: n.endswith(".mp4"))
    if not f:
        ausgabe("### ABBRUCH: keine Warp-Ausgabe")
        sys.exit(1)
    try:
        verschiebe(f, MASTER)
    except OSError as e:
        print(f"mv: {e}", file=sys.stderr)
    beruehre(os.path.join(BASE, f"warp_{VAR}.ok"))
    groesse = str(os.path.getsize(MASTER)) if os.path.isfile(MASTER) else ""
    zeit("warp", uhr() - t0, N, f"Zwischenstand {groesse} Byte")

# ----------------------------------------------------------- 9  Kodierung
LIEFER = os.path.join(BASE, NAME)
if ENCODER == "hevc_qsv":
    KOD_TEXT = f"Kodierung hevc_qsv gq={QSV_QUALITAET} auf {QSV_GERAET}"
    KOD_BEMERKUNG = f"gq={QSV_QUALITAET}"
    FFPROBE_PRUEF = FFPROBE_QSV
    KOD_CMD = [FFMPEG_QSV, "-v", "error", "-y",
               "-init_hw_device", f"qsv=hw:{QSV_GERAET}", "-filter_hw_device", "hw",
               "-i", MASTER,
               "-vf", "format=nv12,hwupload=extra_hw_frames=64",
               "-c:v", "hevc_qsv", "-global_quality", QSV_QUALITAET,
               "-c:a", "copy", "-movflags", "+faststart",
               LIEFER]
elif ENCODER == "hevc_nvenc":
    KOD_TEXT = f"Kodierung hevc_nvenc cq={NVENC_CQ} {NVENC_PRESET}"
    KOD_BEMERKUNG = f"cq={NVENC_CQ}"
    FFPROBE_PRUEF = FFPROBE_ENC
    KOD_CMD = [FFMPEG_ENC, "-v", "error", "-y",
               "-i", MASTER,
               "-c:v", "hevc_nvenc", "-preset", NVENC_PRESET, "-tune", "hq",
               "-rc", "vbr", "-cq", NVENC_CQ, "-b:v", "0",
               "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32", "-bf", "3",
               "-gpu", GPU,
               "-c:a", "copy", "-movflags", "+faststart",
               LIEFER]
else:  # libx265 -- nur fuer Pruefzwecke ohne Hardwarekodierer
    KOD_TEXT = "Kodierung libx265 crf=17 (nur Pruefzwecke)"
    KOD_BEMERKUNG = "libx265 crf=17"
    FFPROBE_PRUEF = FFPROBE_ENC
    KOD_CMD = [FFMPEG_ENC, "-v", "error", "-y", "-i", MASTER,
               "-c:v", "libx265", "-crf", "17", "-preset", "medium",
               "-c:a", "copy", "-movflags", "+faststart", LIEFER]

naechste()
if not os.path.isfile(os.path.join(BASE, f"kod_{VAR}.ok")):
    schritt(KOD_TEXT)
    t0 = uhr()
    if lauf("-", KOD_CMD) != 0:
        sys.exit(1)
    zeit("kodierung", uhr() - t0, N, KOD_BEMERKUNG)

    # ---- Pruefliste, bevor die Datei ausgeliefert wird ----
    vb = einfangen([FFPROBE_PRUEF, "-v", "error", "-select_streams", "v:0", "-count_frames",
                    "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", LIEFER])
    vd = einfangen([FFPROBE_PRUEF, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=duration", "-of", "csv=p=0", LIEFER])
    ad = einfangen([FFPROBE_PRUEF, "-v", "error", "-select_streams", "a:0",
                    "-show_entries", "stream=duration", "-of", "csv=p=0", LIEFER])
    sd = einfangen([FFPROBE_PRUEF, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=duration", "-of", "csv=p=0", SRC])
    sa = einfangen([FFPROBE_PRUEF, "-v", "error", "-select_streams", "a:0",
                    "-show_entries", "stream=duration", "-of", "csv=p=0", SRC])
    ausgabe(f"    Bilder {vb} (Quelle {N}) | Video {vd} s | Ton {ad} s | Quelle {sd} s/{sa} s")
    # 24.09.: ein 60-fps-Clip (sauber 60/1, 21600 Bilder, 21600 Paare im Export) kam
    # aus dem Warp mit 21598 zurueck -- iw3 liess die letzten zwei Bilder (Standbild-
    # Abspann) weg, Stichproben ueber die Laenge zeigen keinen Versatz in der Mitte.
    # Ein Fehlbetrag am Ende bis 50 ms wird hingenommen; den Ton prueft der
    # Versatz-Test darunter mit derselben Grenze.
    fr = einfangen([FFPROBE_PRUEF, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", LIEFER])
    zaehler, _, nenner = (fr or "0/1").partition("/")
    nenner = awk_zahl(nenner) or 1
    duld = int(0.05 * awk_zahl(zaehler) / nenner)
    vb_zahl = ganzzahl(vb)
    if vb_zahl is None or N_ZAHL is None or not (vb_zahl <= N_ZAHL and N_ZAHL - vb_zahl <= duld):
        ausgabe(f"### ABBRUCH: {vb} Bilder statt {N} (Toleranz {duld})")
        sys.exit(1)
    if vb_zahl != N_ZAHL:
        ausgabe(f"    {N_ZAHL - vb_zahl} Bild(er) kuerzer als die Quelle, innerhalb 50 ms")
    # 24.09.: gemessen wird der Versatz, den die Kette hinzufuegt, nicht der, den
    # die Quelle schon mitbringt. Alle diese 4K60-Dateien haben 70-120 ms mehr Ton als
    # Bild; die alte Pruefung (|Video - Ton| > 0,05) haette jede davon nach
    # Stunden Rechenzeit verworfen, obwohl die Datei exakt der Quelle entspricht.
    d = abs((awk_zahl(vd) - awk_zahl(ad)) - (awk_zahl(sd) - awk_zahl(sa)))
    if d > 0.05:
        ausgabe(f"### ABBRUCH: Bild und Ton {'%.6g' % d} s weiter auseinander als in der Quelle")
        sys.exit(1)
    befehl_protokollieren([FFPROBE_PRUEF, "-v", "error", "-select_streams", "v:0",
                           "-show_entries", "stream=codec_name,profile,pix_fmt,width,height",
                           "-of", "default=nw=1", LIEFER])
    sys.stdout.flush()
    subprocess.call([programm(FFPROBE_PRUEF), "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=codec_name,profile,pix_fmt,width,height",
                     "-of", "default=nw=1", LIEFER], cwd=NUNIF_DIR, env=KIND_UMGEBUNG)
    beruehre(os.path.join(BASE, f"kod_{VAR}.ok"))

# ------------------------------------- Punkt 6: genau eine *_LRF.mp4 in $OUT
ZIEL = os.path.join(OUT, NAME)
if OUT != BASE and os.path.isfile(LIEFER):
    try:
        verschiebe(LIEFER, ZIEL)
    except OSError as e:
        print(f"mv: {e}", file=sys.stderr)
        sys.exit(1)
if not os.path.isfile(ZIEL):
    ausgabe(f"### ABBRUCH: Lieferdatei fehlt: {ZIEL}")
    sys.exit(1)
if VGL:
    try:
        shutil.copyfile(ZIEL, os.path.join(VGL, NAME))
    except OSError:
        pass
    rechte_99_100(os.path.join(VGL, NAME))
rechte_99_100(ZIEL)
if MASTER_LOESCHEN == "1" and os.path.isfile(os.path.join(BASE, f"kod_{VAR}.ok")):
    try:
        os.remove(MASTER)
    except OSError:
        pass
    ausgabe("    Zwischenstand geloescht")

ausgabe(f"### ALLES FERTIG -> {ZIEL}  {jetzt()}")
sys.stdout.write(open(Z, encoding="utf-8").read())
sys.stdout.flush()
sys.exit(0)
