import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

# Measured on a Quadro RTX 8000 (46 GB), 369-frame clip, torch 2.3.0+cu121, ViTPose fp16 off.
# Source: docs/benchmarks.md stage table.
stages = ["GVHMR\nbody", "HaMeR\nhands", "EMICA\nface", "Gaze\n(opt-in)", "IK\nhands", "Render"]
secs   = [65.3, 105.3, 39.4, 18.1, 12.6, 1.6]

sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(9, 4.2))
colors = ["#4c72b0"] * len(stages)
colors[1] = "#c44e52"                      # HaMeR: the one that dominates
sns.barplot(x=stages, y=secs, hue=stages, palette=colors, legend=False, ax=ax)
for i, v in enumerate(secs):
    ax.text(i, v + 2, f"{v:.0f}s", ha="center", va="bottom", fontsize=13)
ax.set_ylabel("seconds")
ax.set_xlabel("")
ax.set_ylim(0, max(secs) * 1.18)
ax.set_title("Wall time per stage — 369-frame clip, Quadro RTX 8000", fontsize=15, pad=12)
sns.despine(left=True)
fig.tight_layout()
fig.savefig("docs/img/stage_times.png", dpi=140)
print("wrote docs/img/stage_times.png")
