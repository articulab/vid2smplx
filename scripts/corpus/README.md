# Historical corpus scripts

One-off tools from the 2026-08 interpersonality corpus run (101 clips). **None of them are
part of the pipeline** — `vid2smplx run` never calls any of these, and you do not need them
to process your own videos. They are kept because they are the only record of how that
corpus was produced and repaired, and its outputs are still on disk.

| Script | What it did, once |
|---|---|
| `slurm_redo_excerpts.sh` | re-rendered the excerpt MP4s after a render fix |
| `sync_results.sh` | rsynced finished clips off the cluster |
| `audit_in_vs_out.sh` | counted inputs against outputs to find clips the array had missed |
| `make_corpus_readme.py` | generated the corpus's own README from the finished outputs |
| `backfill_fps.py` | added the `fps` key to params written before fps was stored |
| `repair_body_pose.py` | re-derived `body_pose` for clips written by a buggy merge |
| `check_frame_alignment.py` | checked params frame counts against the source videos |

They may need editing for paths before they run again.
