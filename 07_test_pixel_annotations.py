import json
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ANNOTATION_FILE = Path(
    r"data\annotations\master_annotations.json"
)

OUTPUT_FILE = Path(
    r"data\annotations\pixel_annotations_test.json"
)

# ============================================================
# IMAGE / SPECTROGRAM PARAMETERS
# ============================================================

IMAGE_WIDTH = 1500
IMAGE_HEIGHT = 1500

FREQ_BINS = 4096
TIME_BINS = 978

SAMPLE_RATE = 1.0

# Because fftshift=True and sample rate = 1.0
FREQ_MIN = -SAMPLE_RATE / 2
FREQ_MAX = SAMPLE_RATE / 2


# ============================================================
# LOAD MASTER ANNOTATIONS
# ============================================================

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("PIXEL ANNOTATION TEST")
print("=" * 70)

# Only test chunk 0
chunk = data["chunks"][0]

print("Chunk:", chunk["chunk_id"])
print("Image:", chunk["image_file"])


# ============================================================
# FREQUENCY -> IMAGE Y
# ============================================================

def frequency_to_y(freq):
    """
    Convert frequency to image Y coordinate.

    fftshifted spectrogram:

        high frequency -> top
        low frequency  -> bottom

    Image coordinates:

        y = 0           -> top
        y = HEIGHT      -> bottom
    """

    normalized = (
        (freq - FREQ_MIN)
        / (FREQ_MAX - FREQ_MIN)
    )

    y = IMAGE_HEIGHT * (1 - normalized)

    return y


# ============================================================
# TIME -> IMAGE X
# ============================================================

def time_to_x(time_value, time_start, time_end):
    """
    Convert time coordinate to image X coordinate.
    """

    if time_end == time_start:
        return 0.0

    normalized = (
        (time_value - time_start)
        / (time_end - time_start)
    )

    x = normalized * IMAGE_WIDTH

    return x


# ============================================================
# CHUNK TIME RANGE
# ============================================================

chunk_time_start = chunk["time_start"]
chunk_time_end = chunk["time_end"]


# ============================================================
# PROCESS ANNOTATIONS
# ============================================================

pixel_annotations = []

for ann in chunk["annotations"]:

    class_id = ann["class_id"]
    class_name = ann["class_name"]

    freq_low = ann["freq_low"]
    freq_high = ann["freq_high"]

    ann_time_start = ann["time_start"]
    ann_time_end = ann["time_end"]

    # --------------------------------------------------------
    # TIME -> X
    # --------------------------------------------------------

    x_min = time_to_x(
        ann_time_start,
        chunk_time_start,
        chunk_time_end
    )

    x_max = time_to_x(
        ann_time_end,
        chunk_time_start,
        chunk_time_end
    )

    # --------------------------------------------------------
    # FREQUENCY -> Y
    # --------------------------------------------------------

    # High frequency is smaller Y
    y_min = frequency_to_y(freq_high)

    # Low frequency is larger Y
    y_max = frequency_to_y(freq_low)

    # --------------------------------------------------------
    # CLIP TO IMAGE
    # --------------------------------------------------------

    x_min = max(0.0, min(IMAGE_WIDTH, x_min))
    x_max = max(0.0, min(IMAGE_WIDTH, x_max))

    y_min = max(0.0, min(IMAGE_HEIGHT, y_min))
    y_max = max(0.0, min(IMAGE_HEIGHT, y_max))

    # --------------------------------------------------------
    # NORMALIZED COORDINATES
    # --------------------------------------------------------

    normalized = {
        "x_min": x_min / IMAGE_WIDTH,
        "y_min": y_min / IMAGE_HEIGHT,
        "x_max": x_max / IMAGE_WIDTH,
        "y_max": y_max / IMAGE_HEIGHT,
    }

    pixel_ann = {
        "class_id": class_id,
        "class_name": class_name,

        "bbox": {
            "x_min": round(x_min, 3),
            "y_min": round(y_min, 3),
            "x_max": round(x_max, 3),
            "y_max": round(y_max, 3),
        },

        "bbox_normalized": {
            "x_min": round(normalized["x_min"], 6),
            "y_min": round(normalized["y_min"], 6),
            "x_max": round(normalized["x_max"], 6),
            "y_max": round(normalized["y_max"], 6),
        },

        "frequency": {
            "low": freq_low,
            "high": freq_high,
        },

        "time": {
            "start": ann_time_start,
            "end": ann_time_end,
        },
    }

    pixel_annotations.append(pixel_ann)


# ============================================================
# RESULT
# ============================================================

result = {
    "image_id": chunk["image_id"],
    "image_file": chunk["image_file"],
    "chunk_id": chunk["chunk_id"],

    "image": {
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
    },

    "spectrogram": {
        "frequency_bins": FREQ_BINS,
        "time_bins": TIME_BINS,
        "frequency_range": [
            FREQ_MIN,
            FREQ_MAX
        ],
    },

    "annotations": pixel_annotations,
}


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("IMAGE:")
print(f"  Width  : {IMAGE_WIDTH}")
print(f"  Height : {IMAGE_HEIGHT}")

print()
print("SPECTROGRAM:")
print(f"  Frequency bins : {FREQ_BINS}")
print(f"  Time bins      : {TIME_BINS}")
print(f"  Frequency range: {FREQ_MIN} -> {FREQ_MAX}")

print()
print("=" * 70)
print("PIXEL BOUNDING BOXES")
print("=" * 70)

for ann in pixel_annotations:

    b = ann["bbox"]
    n = ann["bbox_normalized"]

    print()
    print(f"Class: {ann['class_name']}")
    print(f"Class ID: {ann['class_id']}")

    print(
        f"  BBox pixels: "
        f"({b['x_min']}, {b['y_min']}) "
        f"-> "
        f"({b['x_max']}, {b['y_max']})"
    )

    print(
        f"  BBox normalized: "
        f"({n['x_min']}, {n['y_min']}) "
        f"-> "
        f"({n['x_max']}, {n['y_max']})"
    )


# ============================================================
# SAVE
# ============================================================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False
    )

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)