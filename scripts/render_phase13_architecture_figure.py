"""Render the Phase 13 pipeline comparison as a simple educational diagram.

This script draws boxes and arrows only. It does not import PyTorch, load a
checkpoint, inspect images, or execute either model.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "figures" / "phase13_pipeline_architectures.png"


def box(axis, x, y, width, height, text, color):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color, edgecolor="#263238", linewidth=1.4,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)


def arrow(axis, start, end, color="#455a64", style="-"):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                  linewidth=1.5, color=color, linestyle=style))


fig, axes = plt.subplots(1, 3, figsize=(16, 7))
titles = ["A. YOLO only", "B. Classifier → YOLO", "C. Independent classifier + YOLO"]
for axis, title in zip(axes, titles):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title(title, fontsize=13, weight="bold", pad=14)

# Pipeline A: one detector path.
box(axes[0], .28, .78, .44, .10, "Camera image", "#e3f2fd")
box(axes[0], .28, .56, .44, .12, "YOLO detector", "#fff3e0")
box(axes[0], .20, .31, .60, .14, "Damage boxes + type\n+ confidence", "#fff8e1")
box(axes[0], .25, .08, .50, .12, "Operational decision", "#e8f5e9")
arrow(axes[0], (.50, .78), (.50, .68)); arrow(axes[0], (.50, .56), (.50, .45)); arrow(axes[0], (.50, .31), (.50, .20))
axes[0].text(.5, .01, "Simple and localized; image-level and\nopen-state coverage still unvalidated.", ha="center", fontsize=9)

# Pipeline B: visually show the dangerous hard gate.
box(axes[1], .28, .80, .44, .10, "Camera image", "#e3f2fd")
box(axes[1], .24, .60, .52, .12, "MobileNet classifier", "#e8eaf6")
box(axes[1], .08, .37, .34, .11, "Intact → STOP", "#ffebee")
box(axes[1], .58, .37, .34, .11, "Damaged → YOLO", "#fff3e0")
box(axes[1], .58, .13, .34, .12, "Boxes + type", "#fff8e1")
arrow(axes[1], (.50, .80), (.50, .72)); arrow(axes[1], (.42, .60), (.25, .48), color="#c62828"); arrow(axes[1], (.58, .60), (.75, .48)); arrow(axes[1], (.75, .37), (.75, .25))
axes[1].text(.25, .29, "Unsafe hard gate:\nFN suppresses YOLO", ha="center", color="#b71c1c", fontsize=9, weight="bold")
axes[1].text(.5, .01, "Not recommended: open-box recall is 28.57%.", ha="center", fontsize=9)

# Pipeline C: neither branch controls the other.
box(axes[2], .28, .82, .44, .10, "Camera image", "#e3f2fd")
box(axes[2], .05, .58, .40, .13, "MobileNet\nimage-level evidence", "#e8eaf6")
box(axes[2], .55, .58, .40, .13, "YOLO\nlocalized evidence", "#fff3e0")
box(axes[2], .22, .33, .56, .13, "Combined evidence record\n(no rule invented yet)", "#f3e5f5")
box(axes[2], .25, .10, .50, .12, "Decision / REVIEW conflicts", "#e8f5e9")
arrow(axes[2], (.45, .82), (.25, .71)); arrow(axes[2], (.55, .82), (.75, .71)); arrow(axes[2], (.25, .58), (.42, .46)); arrow(axes[2], (.75, .58), (.58, .46)); arrow(axes[2], (.50, .33), (.50, .22))
axes[2].text(.5, .01, "Recommended prototype: evidence is independent;\nfuture open/closed evidence can be added.", ha="center", fontsize=9)

fig.suptitle("Phase 13 candidate parcel-inspection pipelines", fontsize=16, weight="bold")
fig.tight_layout(rect=(0, .02, 1, .94))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved design-only figure: {OUTPUT}")
