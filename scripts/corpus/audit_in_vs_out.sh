#!/bin/bash
# Audit: every source video should have exactly one .npz, and vice versa.
#
# Reports, per session/condition: source count, output count, and which
# participants are missing. Distinguishes "not processed" from "not synced".
set -u

SRC=${SRC:-$HOME/staging/interpersonality}
OUT=${OUT:-$HOME/staging/interpersonality_out}
REMOTE=${REMOTE:-ymachta@cleps}
RDIR=${RDIR:-/scratch/ymachta/interpersonality_out}

echo "SRC: $SRC"
echo "OUT: $OUT"
echo

# What the cluster believes is finished (authoritative)
DONE_LIST=$(ssh -o BatchMode=yes "$REMOTE" \
  "ls -d $RDIR/*/DONE 2>/dev/null | xargs -r -n1 dirname | xargs -r -n1 basename" 2>/dev/null)

printf "%-6s %-5s %6s %6s %6s   %s\n" SESS COND SRC NPZ EXCERPT MISSING
total_src=0; total_npz=0; total_exc=0; missing_all=""

for sd in "$SRC"/S*/; do
    sess=$(basename "$sd")
    for cd in "$sd"*/; do
        [ -d "$cd" ] || continue
        cond=$(basename "$cd")
        names=$(find "$cd" -name '*.mp4' -printf '%f\n' 2>/dev/null | sed 's/\.mp4$//' | sort)
        n_src=$(echo "$names" | grep -c . )
        [ "$n_src" -eq 0 ] && continue

        n_npz=0; n_exc=0; miss=""
        for nm in $names; do
            if [ -f "$OUT/$sess/$cond/$nm.npz" ]; then
                n_npz=$((n_npz+1))
                [ -s "$OUT/$sess/$cond/${nm}_excerpt.mp4" ] && n_exc=$((n_exc+1))
            else
                # distinguish "cluster finished it, just not synced" from "never done"
                if echo "$DONE_LIST" | grep -qx "$nm"; then
                    miss="$miss $nm(unsynced)"
                else
                    miss="$miss $nm(NOT-DONE)"
                fi
                missing_all="$missing_all $nm"
            fi
        done
        flag=""
        [ "$n_npz" -ne "$n_src" ] && flag="  <-- gap"
        printf "%-6s %-5s %6d %6d %6d  %s%s\n" "$sess" "$cond" "$n_src" "$n_npz" "$n_exc" "$miss" "$flag"
        total_src=$((total_src+n_src)); total_npz=$((total_npz+n_npz)); total_exc=$((total_exc+n_exc))
    done
done

echo
echo "TOTAL  source=$total_src  npz=$total_npz  excerpts=$total_exc"
if [ "$total_src" -eq 0 ]; then
    echo "ERROR — found no source videos under $SRC; check the path (vacuous pass avoided)"
    exit 1
elif [ "$total_src" -eq "$total_npz" ]; then
    echo "OK — every source video has a .npz"
else
    echo "GAP — $((total_src-total_npz)) source videos have no .npz:$missing_all"
fi

# orphans: outputs with no corresponding source
orphans=""
for f in $(find "$OUT" -name '*.npz' 2>/dev/null); do
    rel=${f#$OUT/}; sess=${rel%%/*}; rest=${rel#*/}; cond=${rest%%/*}; nm=$(basename "$f" .npz)
    [ -f "$SRC/$sess/$cond/$nm.mp4" ] || orphans="$orphans $nm"
done
[ -n "$orphans" ] && echo "ORPHAN outputs (no source video):$orphans" || echo "OK — no orphan outputs"
