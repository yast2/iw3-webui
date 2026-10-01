#!/bin/sh
# ---------------------------------------------------------------------------
# run_voll.sh -- Produktionskette "G2EMA", eingefroren am 21.09.2026.
#
# Entscheidung vom 21.09. 16:35:
#   "Stick with G2EMA. Keep the flow as an optional checkbox."
#
# Kette:
#   Quelle -> frames/ (JPEG)
#          -> [optional] waifu2x noise_scale2x (GPU)  Vorgabe AUS
#          -> DepthPro / DepthPro_S (voll, GPU)       Feinband, Umrisse, Detail
#          -> VDA_B mit --ema-normalize (GPU)         Grobband, zeitliche Ruhe
#          -> Bandtausch r=30 linear (CPU)            stufe2_band_schnell.py, WP2
#          -> [optional] Fluss RAFT fp16 (GPU)        Vorgabe AUS
#          -> Export (mapper mul_1+mul_2=0.5)
#          -> Warp mlbw_l2 -c 0.5 sod_v1 --synthetic-view right (GPU)
#             schreibt einen verlustfreien Zwischenstand (x264 crf 0)
#          -> Kodierung hevc_qsv -global_quality 17 auf renderD128 (Arc B60)
#
# Belege fuer jede dieser Festlegungen: interne Messnotizen
# (Block "EINGEFROREN").
#
# Jede Stufe hat eine .ok-Markerdatei; ein Wiederanlauf rechnet nur das Fehlende.
# Wanduhr je Stufe wird nach $BASE/zeiten.tsv geschrieben.
#
# ---------------------------------------------------------------------------
# 21.09.2026, Ausliefer-Runde: sieben Schnittstellenpunkte fuer die Weboberflaeche
# (interne Messnotiz, Abschnitt "Was ich von
# run_voll.sh brauche"). 🔴 KEINER davon aendert das Rezept -- die Aufrufe der
# Stufen 1..7b sind Zeichen fuer Zeichen die eingefrorenen. Geaendert wurde nur,
# woher die Pfade kommen und was auf die Standardausgabe geht:
#
#   1  --level <fast|economical|standard>   waehlt das Rezept
#   2  --work <verzeichnis>                 statt des festen BASE
#   3  -i <datei>  -o <verzeichnis>         von aussen gesetzt
#   4  Rueckgabewert != 0 bei Fehlschlag    (hatte es schon)
#   5  Stufenmarken "=== i/n Bezeichnung"   durchnummeriert, i genau einmal
#   6  genau eine *_LRF.mp4 in -o
#   7  trap TERM/INT nimmt die laufende Stufe mit
#
# Der alte Aufruf von Hand bleibt gueltig:  sh run_voll.sh [--fluss] [-d=1.376]
# ---------------------------------------------------------------------------
set -u

# ----------------------------------------------------------- Einstellungen
SRC=${SRC:-/input/beispiel.mp4}
BASE=${BASE:-/output/_voll/_freeze/lauf}
VGL_VORGABE=/output/bench/owl3d-vergleich  # dorthin geht die Lieferdatei zusaetzlich,
                                           # aber nur beim Aufruf von Hand (ohne -o)
# BIN: stufe2_band_schnell.py, stufe3_fluss.py, build_export_full.py,
# umnummerieren.py. Vorgabe ist das Verzeichnis dieses Skripts, damit die Kette
# an einem Stueck umziehen kann.
BIN=${BIN:-$(cd "$(dirname "$0")" && pwd)}
# Die Oberflaeche startet Auftraege als iw3 (uid 99) ohne Heimatordner; torchvision legt
# die RAFT-Gewichte der Fluss-Stufe sonst unter $HOME/.cache ab und scheitert (23.09.).
export TORCH_HOME="${TORCH_HOME:-/config/torch}"

# Divergenz. Am 21.09. am ganzen Film ueber p1..p99 nachgetrimmt; der Wert
# davor (1.376) war auf Minute 1 getrimmt. Herleitung in
# interne Messnotiz.
D=${D:-1.376}

# Rezeptstufe. standard = G2EMA (DepthPro voll), economical = G5EMA
# (DepthPro_S), fast = ein einzelner iw3-Durchlauf ohne Kette.
STUFE=${STUFE:-standard}

# Der Schalter, der bleiben sollte: die Flussstufe (RAFT fp16,
# rund 1 h 05 GPU je Film). VORGABE AUS -- sie kauft auf diesem Material
# 8 % Flimmern fuer eine Stunde Rechenzeit, und man sieht den
# Unterschied nicht ("Both are ounces in difference"). Fuer andere Szenen
# -- schnelle Bewegung, Schnitte, kontrastarmes Material -- bleibt sie da.
#   Anschalten per Umgebung:  FLUSS=1 sh run_voll.sh
#   oder per Kommandozeile:   sh run_voll.sh --fluss   bzw.  --flow
FLUSS=${FLUSS:-0}

# waifu2x 2x vor dem Warp. VORGABE AUS. Hochskaliert wird nur das RGB, nicht
# die Tiefe -- DepthPro rechnet intern auf fest 1536x1536, mehr Tiefendetail
# gaebe es ohnehin nicht (interne Messnotiz).
HOCH=${HOCH:-0}

# 24.09.2026, Kantenfix. PFLICHT seit 24.09. 16:35: immer an, nicht abschaltbar.
# (12:02 Vorgabe AN nach einem Vergleichslauf, davor AUS.) Der nunif-Pin d23721f1 kehrt beim Import
# einer --export-YAML die Kantenweitung um (iw3/utils.py 1720/1908:
# -dilate_edge(-depths, ...)); upstream 40d46847 (17.06.2026, nur dev) rechnet
# dilate_edge(depths, ...). Betrifft nur den Warp dieser Kette (Import), nicht
# Fast und nicht die Tiefenstufen. iw3_kantenfix.py laedt iw3 mit genau diesen
# zwei Zeilen getauscht; /opt/nunif bleibt unberuehrt.
#   Kein Abschalten: KANTENFIX aus der Umgebung wird nicht gelesen, --kein-kantenfix
#   bricht ab, --kantenfix wird fuer alte Aufrufe still angenommen. Fehlt
#   iw3_kantenfix.py, bricht die Kette ab, statt ohne Fix zu warpen.
KANTENFIX=1

# Der verlustfreie Zwischenstand vor der QSV-Kodierung ist gross (rund 27 GB
# je Stunde HD). 1 = nach erfolgreicher, geprueefter Kodierung loeschen.
# Leer heisst "nicht gesetzt": beim Aufruf von Hand wird er dann behalten
# (Vorgabe wie bisher), bei einem Auftrag aus der Warteschlange geloescht --
# dort raeumt die Oberflaeche das Kratzverzeichnis nach Erfolg ohnehin weg.
MASTER_LOESCHEN=${MASTER_LOESCHEN-}

# Parallele Leser auf dem Array. 3 gleichzeitige Leser schaffen 40,4 MB/s,
# 6 nur noch 17,7 MB/s -- ab da sucht die Platte nur noch.
# Beleg: interne Messnotiz.
LESER=${LESER:-3}

GPU=${GPU:-0}
STEREO=${STEREO:-full_sbs}

FFMPEG_QSV=${FFMPEG_QSV:-/usr/lib/jellyfin-ffmpeg/ffmpeg}
FFPROBE_QSV=${FFPROBE_QSV:-/usr/lib/jellyfin-ffmpeg/ffprobe}
# renderD128 ist die Arc Pro B60 (8086:E211, Treiber xe), renderD129 die iGPU.
# Die Zuordnung ist die umgekehrte der naheliegenden -- vor dem Aendern
# /sys/class/drm/renderD12*/device/uevent lesen.
QSV_GERAET=${QSV_GERAET:-/dev/dri/renderD128}
QSV_QUALITAET=${QSV_QUALITAET:-17}

OUT=""
VGL=${VGL:-}

fehlt() { echo "Argument $1 braucht einen Wert"; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --level)        [ $# -ge 2 ] || fehlt "$1"; STUFE=$2; shift 2 ;;
    --level=*)      STUFE=${1#--level=}; shift ;;
    -i|--input)     [ $# -ge 2 ] || fehlt "$1"; SRC=$2; shift 2 ;;
    --src=*)        SRC=${1#--src=}; shift ;;
    -o|--output)    [ $# -ge 2 ] || fehlt "$1"; OUT=$2; shift 2 ;;
    --work)         [ $# -ge 2 ] || fehlt "$1"; BASE=$2; shift 2 ;;
    --base=*)       BASE=${1#--base=}; shift ;;
    --gpu)          [ $# -ge 2 ] || fehlt "$1"; GPU=$2; shift 2 ;;
    --stereo-format) [ $# -ge 2 ] || fehlt "$1"; STEREO=$2; shift 2 ;;
    --flow|--fluss) FLUSS=1; shift ;;
    --kein-fluss)   FLUSS=0; shift ;;
    --upscale|--hochskalieren) HOCH=1; shift ;;
    --kantenfix)    shift ;;   # Pflicht, s. o. -- bleibt nur fuer alte Aufrufe
    --kein-kantenfix) echo "### ABBRUCH: der Kantenfix ist Pflicht (seit 24.09.2026) -- abschalten geht nicht"; exit 2 ;;
    --master-weg)   MASTER_LOESCHEN=1; shift ;;
    -d=*)           D=${1#-d=}; shift ;;
    -h|--help)      sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unbekanntes Argument: $1"; exit 2 ;;
  esac
done

# ------------------------------------------------------------ Rezeptstufe
case "$STUFE" in
  standard)    MODELL_FEIN=DepthPro;   STAMM_VAR=G2EMA ;;
  economical)  MODELL_FEIN=DepthPro_S; STAMM_VAR=G5EMA ;;
  fast)        MODELL_FEIN=-;          STAMM_VAR=SCHNELL ;;
  *) echo "unbekannte Stufe: $STUFE (fast|economical|standard)"; exit 2 ;;
esac
if [ "$STUFE" = fast ] && { [ "$FLUSS" = 1 ] || [ "$HOCH" = 1 ]; }; then
  echo "### ABBRUCH: --flow und --upscale sind Kettenstufen; 'fast' ist keine Kette"
  exit 2
fi
# Fast importiert nichts, der Fix greift dort nicht: still aus, damit die Variante
# (und der Dateiname) nicht etwas behauptet, das nie gerechnet wurde.
[ "$STUFE" = fast ] && KANTENFIX=0
if [ "$KANTENFIX" = 1 ]; then
  [ -f "$BIN/iw3_kantenfix.py" ] || { echo "### ABBRUCH: $BIN/iw3_kantenfix.py fehlt"; exit 1; }
fi

# ------------------------------------------------- Stereoformat -> iw3-Schalter
case "$STEREO" in
  full_sbs)   SF="" ;;
  half_sbs)   SF="--half-sbs" ;;
  tb)         SF="--tb" ;;
  half_tb)    SF="--half-tb" ;;
  vr180)      SF="--vr180" ;;
  cross_eyed) SF="--cross-eyed" ;;
  rgbd)       SF="--rgbd" ;;
  half_rgbd)  SF="--half-rgbd" ;;
  anaglyph)   SF="--anaglyph dubois" ;;
  *) echo "unbekanntes Stereoformat: $STEREO"; exit 2 ;;
esac

VAR=$STAMM_VAR
[ "$FLUSS" = 1 ] && VAR=${VAR}Fluss
[ "$HOCH"  = 1 ] && VAR=${VAR}4K
[ "$KANTENFIX" = 1 ] && VAR=${VAR}Kante

STAMM=$(basename "$SRC" | sed 's/\.[^.]*$//')
NAME=${NAME:-${STAMM}_${VAR}_LRF.mp4}
MASTER=$BASE/master_$VAR.mp4
Z=$BASE/zeiten.tsv

# Punkt 3/6: ohne -o bleibt alles wie beim Aufruf von Hand -- Lieferdatei in
# $BASE, zusaetzlich eine Kopie nach $VGL. Mit -o geht genau eine Datei dorthin
# und sonst nichts; die Vergleichskopie entfaellt, weil ein Auftrag aus der
# Warteschlange nichts in fremden Verzeichnissen zu suchen hat.
if [ -z "$OUT" ]; then
  OUT=$BASE
  VGL=${VGL:-$VGL_VORGABE}
  MASTER_LOESCHEN=${MASTER_LOESCHEN:-0}
else
  MASTER_LOESCHEN=${MASTER_LOESCHEN:-1}
fi

[ -f "$SRC" ] || { echo "### ABBRUCH: Quelle nicht gefunden: $SRC"; exit 1; }
mkdir -p "$BASE" "$BASE/log" "$OUT" || exit 1
[ -n "$VGL" ] && mkdir -p "$VGL"
cd /opt/nunif || exit 1
[ -f "$Z" ] || printf "stufe\tvariante\tbeginn\twall_s\ts_pro_bild\tbemerkung\n" > "$Z"

# ---------------------------------------------------- Punkt 5: Stufenmarken
# Die Oberflaeche liest "=== i/n Bezeichnung"; i muss genau einmal vorkommen und
# n wird mitgelesen, nicht angenommen. Der Zaehler laeuft deshalb auch ueber
# Stufen weiter, die ein Wiederanlauf ueberspringt -- sonst wuerde nach einem
# Abbruch dieselbe Nummer ein zweites Mal auftauchen.
if [ "$STUFE" = fast ]; then
  STUFEN=1
else
  STUFEN=7
  [ "$FLUSS" = 1 ] && STUFEN=$((STUFEN + 1))
  [ "$HOCH"  = 1 ] && STUFEN=$((STUFEN + 1))
fi
I=0
naechste() { I=$((I + 1)); }
schritt()  { echo "=== $I/$STUFEN $1  $(date '+%F %H:%M:%S')"; }
zeit()     { # stufe wall_s n bemerkung
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$VAR" "$(date '+%F %H:%M:%S')" "$2" \
     "$(echo "$2" "$3" | awk '{if($2>0)printf "%.4f",$1/$2; else printf "-"}')" "$4" >> "$Z"
  echo "    $1: ${2}s"
}

# ------------------------------------------- Punkt 7: Abbruch nimmt die Stufe mit
# Ohne das laeuft nach einem "Cancel" in der Oberflaeche die Stufe weiter und die
# Warteschlange startet den naechsten Auftrag auf dieselbe GPU.
KIND=
abbruch() {
  echo ""
  echo "### ABBRUCH: Signal empfangen $(date '+%F %H:%M:%S')"
  if [ -n "$KIND" ]; then
    kill -TERM "$KIND" 2>/dev/null
    pkill -TERM -P "$KIND" 2>/dev/null
    n=0
    while [ $n -lt 10 ] && kill -0 "$KIND" 2>/dev/null; do sleep 1; n=$((n + 1)); done
    kill -KILL "$KIND" 2>/dev/null
    pkill -KILL -P "$KIND" 2>/dev/null
  fi
  exit 143
}
trap abbruch TERM INT HUP

# lauf <logdatei|-> befehl...
# Startet im Hintergrund, merkt sich die PID fuer den trap und wartet.
lauf() {
  _log=$1; shift
  if [ "$_log" = "-" ]; then "$@" & else "$@" > "$_log" 2>&1 & fi
  KIND=$!
  wait "$KIND"
  _rc=$?
  KIND=
  return $_rc
}

echo "### run_voll.sh  Stufe=$STUFE  Variante=$VAR  $(date '+%F %H:%M:%S')"
echo "    Quelle      $SRC"
echo "    Arbeit      $BASE"
echo "    Ausgabe     $OUT"
echo "    Divergenz   $D"
echo "    GPU         $GPU"
echo "    Stereo      $STEREO"
echo "    Fluss       $([ "$FLUSS" = 1 ] && echo AN || echo aus)"
echo "    4K          $([ "$HOCH" = 1 ] && echo AN || echo aus)"
echo "    Kantenfix   $([ "$KANTENFIX" = 1 ] && echo 'AN (nunif 40d46847, nur Warp)' || echo 'entfaellt (fast importiert nichts)')"
echo "    Lieferdatei $NAME"

# ------------------------------------------ 0  Feste Bildrate, nur wenn noetig
# 24.09.: Handyaufnahme (Android, gemeldet 120 fps, echt 30,3) -- ffmpeg zog 6326 echte
# Bilder, iw3 fuellte fuer VDA auf die gemeldeten 120 auf (25038), und der Export
# haette 120 fps ins yml geschrieben. Dieselbe Falle, nur kleiner, steckt in 15
# 60-fps-Dateien, meist 4K60-Studiodateien (2-17 Bilder daneben). Solche Quellen einmal
# auf die naechste Normrate bringen (Bilder doppeln/auslassen, Ton kopiert); danach
# sehen alle Stufen dieselbe Quelle. bildrate_pruefen.py schweigt im Normalfall.
if [ ! -f "$BASE/cfr.ok" ]; then
  ziel=$(python3 "$BIN/bildrate_pruefen.py" "$SRC") || exit 1
  if [ -n "$ziel" ]; then
    [ -f "$BASE/frames.ok" ] && { echo "### ABBRUCH: $BASE stammt von vor der Bildratenpruefung -- Arbeitsordner leeren"; exit 1; }
    echo "    Bildrate: gemeldet passt nicht zur Bildzahl -> fest $ziel fps (x264 crf 10)"
    t0=$(date +%s)
    cfr() { lauf - ffmpeg -v error -y -i "$SRC" -map 0:v:0 -map '0:a:0?' \
        -fps_mode cfr -r "$ziel" -c:v libx264 -crf 10 -preset veryfast \
        -pix_fmt yuv420p "$@" -movflags +faststart "$BASE/quelle_cfr.mp4"; }
    cfr -c:a copy || cfr -c:a aac -b:a 256k || exit 1
    zeit bildrate $(( $(date +%s) - t0 )) 0 "fest $ziel fps"
  fi
  echo "$ziel" > "$BASE/cfr.ok"
fi
if [ -n "$(cat "$BASE/cfr.ok")" ]; then
  SRC=$BASE/quelle_cfr.mp4
  echo "    Quelle fest $(cat "$BASE/cfr.ok") fps: $SRC"
fi

# ===========================================================================
# Stufe "fast" -- ein einzelner iw3-Durchlauf, keine Kette.
# Die Oberflaeche ruft diesen Weg nie ueber das Skript auf (sie baut denselben
# Aufruf selbst aus params_json); er steht hier, damit ein Aufruf von Hand
# vollstaendig ist. Parameter wie in der Stufentabelle der Oberflaeche.
# ===========================================================================
if [ "$STUFE" = fast ]; then
  naechste; schritt "VDA_B + EMA, ein Durchlauf"
  t0=$(date +%s)
  rm -rf "$BASE/fast"; mkdir -p "$BASE/fast"
  # shellcheck disable=SC2086
  lauf "$BASE/log/fast.txt" python3 -m iw3 -i "$SRC" -o "$BASE/fast" -y \
      --gpu "$GPU" --depth-model VDA_B --divergence "$D" --convergence 0.5 \
      --convergence-mode sod_v1 --foreground-scale 0 --edge-dilation 2 1 \
      --video-codec libx265 --pix-fmt yuv420p --max-fps 1000 --scene-detect \
      --ema-normalize --ema-decay 0.75 --ema-buffer 30 $SF || exit 1
  f=$(find "$BASE/fast" -name '*.mp4' | head -1)
  [ -n "$f" ] || { echo "### ABBRUCH: keine iw3-Ausgabe"; exit 1; }
  mv "$f" "$OUT/$NAME" || exit 1
  chown 99:100 "$OUT/$NAME" 2>/dev/null
  zeit fast $(( $(date +%s) - t0 )) 0 "ein Durchlauf"
  # Keine Stufenmarke: die Oberflaeche darf dieselbe Nummer nicht zweimal sehen.
  echo "### ALLES FERTIG -> $OUT/$NAME  $(date '+%F %H:%M:%S')"
  cat "$Z"
  exit 0
fi

# ------------------------------------------------------------ 1  Bilder
naechste
if [ ! -f "$BASE/frames.ok" ]; then
  schritt "Bilder extrahieren"
  t0=$(date +%s)
  mkdir -p "$BASE/frames"
  # Ohne -fps_mode passthrough verdoppelt ffmpeg still einzelne Bilder und
  # alles danach steht gegen die Tiefe versetzt.
  lauf - ffmpeg -v error -i "$SRC" -fps_mode passthrough -qscale:v 2 \
      "$BASE/frames/%08d.jpg" || exit 1
  n=$(ls "$BASE/frames" | wc -l); echo "$n" > "$BASE/frames.ok"
  zeit bilder $(( $(date +%s) - t0 )) "$n" "$n Bilder"
fi
N=$(cat "$BASE/frames.ok")
echo "    Bilder: $N"

# ------------------------------------------------ 2  Hochskalieren (optional)
# Nur das RGB, nicht die Tiefe. --noise-level blieb auf der Vorgabe 0, wie im
# gemessenen Lauf (interne Messnotiz); die
# Kachelgroesse 160 ist die gemessene schnellste
# (interne Messnotiz).
RGB=$BASE/frames
if [ "$HOCH" = 1 ]; then
  naechste
  if [ ! -f "$BASE/hoch.ok" ]; then
    schritt "waifu2x noise_scale2x (GPU)"
    t0=$(date +%s)
    mkdir -p "$BASE/rgb2x"
    lauf "$BASE/log/hoch.txt" python3 -m waifu2x.cli --style photo \
        --method noise_scale2x -i "$BASE/frames" -o "$BASE/rgb2x" \
        --format png --tile-size 160 -g "$GPU" || exit 1
    n=$(ls "$BASE/rgb2x" | wc -l)
    [ "$n" -eq "$N" ] || { echo "### ABBRUCH: waifu2x $n Bilder, Quelle $N"; exit 1; }
    echo "$n" > "$BASE/hoch.ok"
    zeit hochskalieren $(( $(date +%s) - t0 )) "$n" "noise_scale2x, Kachel 160"
  fi
  RGB=$BASE/rgb2x
  echo "    RGB 2x: $(cat "$BASE/hoch.ok")"
fi

# ---------------------------------------------------------- 3  Feinband
# Traegt Umrisse und Detail; laeuft ueber den Bildordner (export_images), wo
# --ema-normalize wirkungslos waere -- er wird hier auch nicht gebraucht,
# DepthPro hat keinen zeitlichen Zustand. Laeuft immer auf dem HD-Bildordner,
# auch bei --upscale: DepthPro rechnet intern auf fest 1536x1536.
naechste
if [ ! -f "$BASE/pro.ok" ]; then
  schritt "$MODELL_FEIN"
  t0=$(date +%s)
  lauf "$BASE/log/pro.txt" python3 -m iw3 --export --export-depth-only \
      --depth-model "$MODELL_FEIN" --gpu "$GPU" \
      -i "$BASE/frames" -o "$BASE/pro" --yes || exit 1
  n=$(ls "$BASE/pro/depth" | wc -l); echo "$n" > "$BASE/pro.ok"
  zeit feinband $(( $(date +%s) - t0 )) "$n" "$n Tiefenbilder, $MODELL_FEIN"
fi
echo "    Tiefe $MODELL_FEIN: $(cat "$BASE/pro.ok")"

# ------------------------------------------------- 4  VDA_B + EMA (Grobband)
# 🔴 Muss ueber das VIDEO laufen, nicht ueber frames/: --ema-normalize ist im
# Bildexportpfad (export_images) hart abgeschaltet und wirkt nur im Videopfad
# (WP8, interne Messnotiz). Die EMA ist eine bildweise
# affine Umskalierung, kostet 1,3 % Laufzeit und nimmt VDA_B 29 % Flimmern.
# --scene-detect setzt den zeitlichen Zustand an Schnitten zurueck; auf
# am Testfilm ohne Wirkung (ein einziger Schnitt, unter der Schwelle),
# auf 92 % des Bestands sehr wohl.
naechste
if [ ! -f "$BASE/vda.ok" ]; then
  schritt "VDA_B + EMA"
  t0=$(date +%s)
  # Wiederanlauf nach einem Abbruch in dieser Stufe: umnummerieren.py ueberspringt
  # vorhandene Ziele, alte Karten blieben also stehen -- beide Ordner neu anlegen.
  [ -n "$BASE" ] && rm -rf "$BASE/vda_roh" "$BASE/vda"
  lauf "$BASE/log/vda.txt" python3 -m iw3 --export --export-depth-only \
      --convergence 0.5 --foreground-scale 0 --edge-dilation 2 1 \
      --pix-fmt yuv420p --max-fps 1000 --scene-detect \
      --ema-normalize --ema-decay 0.75 --ema-buffer 30 \
      --depth-model VDA_B --divergence "$D" --gpu "$GPU" \
      -i "$SRC" -o "$BASE/vda_roh" --yes || exit 1
  roh=$(find "$BASE/vda_roh" -type d -name depth | head -1)
  # VDA nummeriert die Videoausgabe ab 0, DepthPro den Bildordner ab 1.
  python3 "$BIN/umnummerieren.py" "$roh" "$BASE/vda" 1 || exit 1
  n=$(ls "$BASE/vda" | wc -l)
  # iw3 schickt das Video durch ffmpegs fps-Filter, Stufe 1 nicht. Der doppelt ein Bild, wo die
  # Uhr der Datei eine halbe Bilddauer gegen die Nennrate verloren hat -- am Ende (23.09.: 1801
  # gegen 1800) oder mitten im Film (30.09., 60-fps-Film: Karte 40080 = Bild 40079, 77422
  # gegen 77421). vda_abgleich.py dekodiert die Quelle genau wie iw3, weiss damit, welche Karte
  # welches Bild traegt, prueft das am Inhalt (Bewegung gegen DepthPro, Ortsprobe an der
  # Doppelung) und verlinkt vda/ erst dann neu. Loest vda_ende_pruefen.py ab, das nur das Ende
  # kannte und die Lage aus der Bewegung erraten musste.
  # interne Messnotiz
  if [ "$n" -gt "$N" ] && python3 "$BIN/vda_abgleich.py" "$SRC" "$(dirname "$roh")/iw3_export.yml" \
       "$roh" "$BASE/vda" "$BASE/pro/depth" "$BASE/frames" "$N"; then
    n=$(ls "$BASE/vda" | wc -l)
  fi
  [ "$n" -eq "$N" ] || { echo "### ABBRUCH: VDA $n Bilder, Quelle $N"; exit 1; }
  echo "$n" > "$BASE/vda.ok"
  zeit vda_b_ema $(( $(date +%s) - t0 )) "$n" "$n Tiefenbilder"
fi
echo "    Tiefe VDA_B+EMA: $(cat "$BASE/vda.ok")"

# ------------------------------------------------------- 5  Bandtausch (CPU)
# Grobband von VDA_B, Feinband von DepthPro, Abgleich linear (Restfehler
# 0,015 gegen 0,264 bei "kehrwert"). Die schnelle Fassung ist rechenweg-,
# nicht ergebnisgleich zur alten: 0,056-0,079 s/Bild statt 0,541 -- 83 % der
# alten Laufzeit lagen im PNG-Schreiben. Lauf gegen Lauf bitgleich.
naechste
if [ ! -f "$BASE/band.ok" ]; then
  schritt "Bandtausch r=30 linear (CPU)"
  t0=$(date +%s)
  lauf "$BASE/log/band.txt" python3 "$BIN/stufe2_band_schnell.py" \
      "$BASE/vda" "$BASE/pro/depth" "$BASE/band" \
      --radius 30 --form linear --lese-faeden "$LESER" || exit 1
  n=$(ls "$BASE/band" | wc -l)
  [ "$n" -eq "$N" ] || { echo "### ABBRUCH: Bandtausch $n Bilder, Quelle $N"; exit 1; }
  echo "$n" > "$BASE/band.ok"
  zeit bandtausch $(( $(date +%s) - t0 )) "$n" "$LESER Lesefaeden"
fi
echo "    Bandtausch: $(cat "$BASE/band.ok")"

# --------------------------------------------------- 6  Fluss (optional, aus)
TIEFE=$BASE/band
if [ "$FLUSS" = "1" ]; then
  naechste
  if [ ! -f "$BASE/fluss.ok" ]; then
    schritt "Fluss RAFT fp16 (GPU)"
    t0=$(date +%s)
    lauf "$BASE/log/fluss.txt" python3 "$BIN/stufe3_fluss.py" \
        "$BASE/band" "$BASE/frames" "$BASE/fluss" \
        --photo-schwelle 0.03 --halb || exit 1
    n=$(ls "$BASE/fluss" | wc -l)
    [ "$n" -eq "$N" ] || { echo "### ABBRUCH: Fluss $n Bilder, Quelle $N"; exit 1; }
    echo "$n" > "$BASE/fluss.ok"
    zeit fluss $(( $(date +%s) - t0 )) "$n" "RAFT fp16"
  fi
  TIEFE=$BASE/fluss
  echo "    Fluss: $(cat "$BASE/fluss.ok")"
else
  echo "    Fluss: aus (Vorgabe)"
fi

# ------------------------------------------------------------- 7  Export
naechste
if [ ! -f "$BASE/exp_$VAR.ok" ]; then
  schritt "Export"
  t0=$(date +%s)
  lauf "$BASE/log/exp_$VAR.txt" python3 "$BIN/build_export_full.py" \
      "$SRC" "$TIEFE" "$BASE/exp_$VAR" \
      --rgb-dir "$RGB" --mapper "mul_1+mul_2=0.5" --pad 0 || exit 1
  # Der Ton wird ab 0 gemuxt. Stimmt die Paarzahl nicht exakt, laeuft das Bild
  # gegen den Ton -- genau die 0,1 s vom 16.09., die drei Polsterbilder kosteten.
  paare=$(ls "$BASE/exp_$VAR/rgb" | wc -l)
  [ "$paare" -eq "$N" ] || { echo "### ABBRUCH: $paare Paare, $N Quellbilder -- Ton wuerde verrutschen"; exit 1; }
  touch "$BASE/exp_$VAR.ok"
  zeit export $(( $(date +%s) - t0 )) "$paare" "$paare Paare"
fi

# --------------------------------------------------------------- 8  Warp
# Zwischenstand verlustfrei (x264 crf 0), weil PyAV in diesem Container kein
# QSV kennt (av.codecs_available: kein *_qsv) und die Lieferkodierung deshalb
# ein zweiter Lauf mit jellyfin-ffmpeg ist. crf 0 in yuv420p heisst: der
# QSV-Kodierer bekommt genau die Pixel, die er auch direkt bekommen haette.
naechste
if [ ! -f "$BASE/warp_$VAR.ok" ]; then
  schritt "Warp mlbw_l2 (GPU)"
  t0=$(date +%s)
  IW3_WARP="-m iw3"
  [ "$KANTENFIX" = 1 ] && IW3_WARP="$BIN/iw3_kantenfix.py"
  # shellcheck disable=SC2086
  lauf "$BASE/log/warp_$VAR.txt" python3 $IW3_WARP \
      -i "$BASE/exp_$VAR/iw3_export.yml" -o "$BASE/out_$VAR" \
      --method mlbw_l2 -d "$D" -c 0.5 --convergence-mode sod_v1 \
      --synthetic-view right --edge-dilation 2 1 --max-fps 1000 --gpu "$GPU" \
      --video-codec libx264 --crf 0 --preset ultrafast --pix-fmt yuv420p \
      --yes $SF || exit 1
  f=$(find "$BASE/out_$VAR" -name '*.mp4' | head -1)
  [ -n "$f" ] || { echo "### ABBRUCH: keine Warp-Ausgabe"; exit 1; }
  mv "$f" "$MASTER"
  touch "$BASE/warp_$VAR.ok"
  zeit warp $(( $(date +%s) - t0 )) "$N" "Zwischenstand $(stat -c%s "$MASTER") Byte"
fi

# ----------------------------------------------------------- 9  Kodierung
naechste
if [ ! -f "$BASE/kod_$VAR.ok" ]; then
  schritt "Kodierung hevc_qsv gq=$QSV_QUALITAET auf $QSV_GERAET"
  t0=$(date +%s)
  lauf - "$FFMPEG_QSV" -v error -y \
      -init_hw_device "qsv=hw:$QSV_GERAET" -filter_hw_device hw \
      -i "$MASTER" \
      -vf "format=nv12,hwupload=extra_hw_frames=64" \
      -c:v hevc_qsv -global_quality "$QSV_QUALITAET" \
      -c:a copy -movflags +faststart \
      "$BASE/$NAME" || exit 1
  zeit kodierung $(( $(date +%s) - t0 )) "$N" "gq=$QSV_QUALITAET"

  # ---- Pruefliste, bevor die Datei ausgeliefert wird ----
  vb=$("$FFPROBE_QSV" -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of csv=p=0 "$BASE/$NAME")
  vd=$("$FFPROBE_QSV" -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$BASE/$NAME")
  ad=$("$FFPROBE_QSV" -v error -select_streams a:0 -show_entries stream=duration -of csv=p=0 "$BASE/$NAME")
  sd=$("$FFPROBE_QSV" -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$SRC")
  sa=$("$FFPROBE_QSV" -v error -select_streams a:0 -show_entries stream=duration -of csv=p=0 "$SRC")
  echo "    Bilder $vb (Quelle $N) | Video $vd s | Ton $ad s | Quelle $sd s/$sa s"
  # 24.09.: ein 60-fps-Clip (sauber 60/1, 21600 Bilder, 21600 Paare im Export) kam
  # aus dem Warp mit 21598 zurueck -- iw3 liess die letzten zwei Bilder (Standbild-
  # Abspann) weg, Stichproben ueber die Laenge zeigen keinen Versatz in der Mitte.
  # Ein Fehlbetrag am Ende bis 50 ms wird hingenommen; den Ton prueft der
  # Versatz-Test darunter mit derselben Grenze.
  fr=$("$FFPROBE_QSV" -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$BASE/$NAME")
  duld=$(echo "${fr:-0/1}" | awk '{split($1,a,"/"); if(a[2]+0==0)a[2]=1; printf "%d", 0.05*a[1]/a[2]}')
  [ "$vb" -le "$N" ] && [ $((N - vb)) -le "$duld" ] || { echo "### ABBRUCH: $vb Bilder statt $N (Toleranz $duld)"; exit 1; }
  [ "$vb" -eq "$N" ] || echo "    $((N - vb)) Bild(er) kuerzer als die Quelle, innerhalb 50 ms"
  # 24.09.: gemessen wird der Versatz, den die Kette hinzufuegt, nicht der, den
  # die Quelle schon mitbringt. Alle diese 4K60-Dateien haben 70-120 ms mehr Ton als
  # Bild; die alte Pruefung (|Video - Ton| > 0,05) haette jede davon nach
  # Stunden Rechenzeit verworfen, obwohl die Datei exakt der Quelle entspricht.
  echo "${vd:-0} ${ad:-0} ${sd:-0} ${sa:-0}" | awk '{d=($1-$2)-($3-$4); if(d<0)d=-d; if(d>0.05){print "### ABBRUCH: Bild und Ton "d" s weiter auseinander als in der Quelle"; exit 1}}' || exit 1
  "$FFPROBE_QSV" -v error -select_streams v:0 \
      -show_entries stream=codec_name,profile,pix_fmt,width,height -of default=nw=1 "$BASE/$NAME"
  touch "$BASE/kod_$VAR.ok"
fi

# ------------------------------------- Punkt 6: genau eine *_LRF.mp4 in $OUT
# Der Zwischenstand (master_*.mp4) bleibt im Kratzverzeichnis; in $OUT landet
# genau diese eine Datei. Beim Wiederanlauf nach erfolgter Kodierung liegt sie
# schon dort -- dann ist nichts zu verschieben.
if [ "$OUT" != "$BASE" ] && [ -f "$BASE/$NAME" ]; then
  mv -f "$BASE/$NAME" "$OUT/$NAME" || exit 1
fi
[ -f "$OUT/$NAME" ] || { echo "### ABBRUCH: Lieferdatei fehlt: $OUT/$NAME"; exit 1; }
if [ -n "$VGL" ]; then
  cp -f "$OUT/$NAME" "$VGL/$NAME" 2>/dev/null
  chown 99:100 "$VGL/$NAME" 2>/dev/null
fi
chown 99:100 "$OUT/$NAME" 2>/dev/null
if [ "$MASTER_LOESCHEN" = "1" ] && [ -f "$BASE/kod_$VAR.ok" ]; then
  rm -f "$MASTER"; echo "    Zwischenstand geloescht"
fi

# Keine Stufenmarke: die Oberflaeche darf dieselbe Nummer nicht zweimal sehen.
echo "### ALLES FERTIG -> $OUT/$NAME  $(date '+%F %H:%M:%S')"
cat "$Z"
exit 0
