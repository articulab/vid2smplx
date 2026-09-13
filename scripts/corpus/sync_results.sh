#!/bin/bash
# Pull finished interpersonality results from cleps into a tree that mirrors the
# input dataset layout:
#
#   staging/interpersonality/S01/BP/P001_S01_BP.mp4        <- input
#   staging/interpersonality_out/S01/BP/P001_S01_BP.npz    <- params
#                                       P001_S01_BP_excerpt.mp4
#                                       P001_S01_BP_summary.json
#
# Only clips marked DONE on the cluster are pulled. Re-runnable: rsync skips
# unchanged files, so this can be run repeatedly as the array progresses.
# Read-only on the cluster — nothing is deleted there.
set -u

REMOTE=${REMOTE:-ymachta@cleps}
RDIR=${RDIR:-/scratch/ymachta/interpersonality_out}
DST=${DST:-$HOME/staging/interpersonality_out}

mkdir -p "$DST"

echo "[sync] listing finished clips on $REMOTE ..."
CLIPS=$(ssh -o BatchMode=yes "$REMOTE" "ls -d $RDIR/*/DONE 2>/dev/null | xargs -r -n1 dirname | xargs -r -n1 basename" 2>/dev/null)
[ -n "$CLIPS" ] || { echo "[sync] no finished clips"; exit 0; }
echo "[sync] $(echo "$CLIPS" | wc -w) clips marked DONE"

n_new=0
for C in $CLIPS; do
    # P001_S01_BP -> session S01, condition BP
    SESS=$(echo "$C" | grep -oE 'S[0-9]+')
    COND=$(echo "$C" | sed -E 's/.*_(BP|CS|TP|FT1|FT2)$/\1/')
    if [ -z "$SESS" ] || [ "$COND" = "$C" ]; then
        echo "  [skip] cannot parse session/condition from '$C'"
        continue
    fi
    OUT="$DST/$SESS/$COND"
    mkdir -p "$OUT"

    # params live one level deeper (process_video nests <out>/<name>/)
    got=0
    rsync -a --partial "$REMOTE:$RDIR/$C/$C/smplx_params.npz" "$OUT/$C.npz" 2>/dev/null && got=1
    rsync -a --partial "$REMOTE:$RDIR/$C/${C}_excerpt.mp4" "$OUT/" 2>/dev/null
    rsync -a --partial "$REMOTE:$RDIR/$C/$C/summary.json" "$OUT/${C}_summary.json" 2>/dev/null
    # a 0-byte excerpt is a failed render, not a result — don't keep it
    [ -f "$OUT/${C}_excerpt.mp4" ] && [ ! -s "$OUT/${C}_excerpt.mp4" ] && rm -f "$OUT/${C}_excerpt.mp4"
    [ "$got" = 1 ] && n_new=$((n_new + 1))
done

echo
echo "[sync] tree under $DST:"
find "$DST" -name '*.npz' | sed "s|$DST/||" | sort | head -20
echo
echo "[sync] npz=$(find "$DST" -name '*.npz' | wc -l)  excerpts=$(find "$DST" -name '*_excerpt.mp4' -size +1k | wc -l)  total=$(du -sh "$DST" 2>/dev/null | cut -f1)"
