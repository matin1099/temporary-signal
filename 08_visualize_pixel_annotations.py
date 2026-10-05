import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg


# ============================================================
# PATHS
# ============================================================

IMAGE_FILE = Path(
    r"data\spectrograms_color\chunk_0000.png"
)

ANNOTATION_FILE = Path(
    r"data\annotations\pixel_annotations_test.json"
)

OUTPUT_FILE = Path(
    r"data\spectrograms_color\chunk_0000_bbox_QA.png"
)


# ============================================================
# LOAD IMAGE
# ============================================================

image = mpimg.imread(IMAGE_FILE)


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# IMAGE SIZE
# ============================================================

height, width = image.shape[:2]

print("=" * 70)
print("PIXEL ANNOTATION VISUAL QA")
print("=" * 70)

print("Image:")
print(f"  Width  : {width}")
print(f"  Height : {height}")

print()
print("Annotations:")
print(f"  Count: {len(data['annotations'])}")


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(
    image,
    origin="upper"
)


# ============================================================
# DRAW BOUNDING BOXES
# ============================================================

for ann in data["annotations"]:

    bbox = ann["bbox"]

    x_min = bbox["x_min"]
    y_min = bbox["y_min"]

    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    width_box = x_max - x_min
    height_box = y_max - y_min

    rectangle = patches.Rectangle(
        (x_min, y_min),
        width_box,
        height_box,
        linewidth=3,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rectangle)

    # Label
    ax.text(
        x_min + 10,
        y_min + 25,
        ann["class_name"],
        fontsize=14,
        fontweight="bold",
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.7,
            pad=3
        )
    )


# ============================================================
# REMOVE AXES
# ============================================================

ax.set_xlim(0, width)
ax.set_ylim(height, 0)

ax.axis("off")


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print()
print("=" * 70)
print("QA IMAGE SAVED")
print("=" * 70)

print(OUTPUT_FILE)