import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ANNOTATION_FILE = Path(
    r"data\annotations\master_annotations.json"
)

OUTPUT_FILE = Path(
    r"data\annotations\pixel_annotations.json"
)


# ============================================================
# IMAGE / SPECTROGRAM PARAMETERS
# ============================================================

IMAGE_WIDTH = 1500
IMAGE_HEIGHT = 1500

SAMPLE_RATE = 1.0

FREQ_BINS = 4096
TIME_BINS = 978

# Because fftshift=True
FREQ_MIN = -SAMPLE_RATE / 2
FREQ_MAX = SAMPLE_RATE / 2


# ============================================================
# FREQUENCY -> IMAGE Y
# ============================================================

def frequency_to_y(freq):
    """
    Convert physical frequency to image Y coordinate.

    After fftshift:

        high frequency -> top
        low frequency  -> bottom
    """

    normalized = (
        (freq - FREQ_MIN)
        / (FREQ_MAX - FREQ_MIN)
    )

    y = IMAGE_HEIGHT * (1.0 - normalized)

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
# LOAD MASTER ANNOTATIONS
# ============================================================

print("=" * 70)
print("BUILDING FINAL PIXEL ANNOTATIONS")
print("=" * 70)

print()
print("Loading master annotations...")

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    master = json.load(f)

chunks = master["chunks"]

print(f"Total chunks: {len(chunks)}")


# ============================================================
# OUTPUT DATASET
# ============================================================

output = {
    "dataset_name": master["dataset_name"],
    "source_format": master["source_format"],
    "source_file": master["source_file"],

    "image": {
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "format": "PNG",
        "type": "RGB color",
    },

    "spectrogram": {
        "frequency_bins": FREQ_BINS,
        "time_bins": TIME_BINS,

        "frequency_range": [
            FREQ_MIN,
            FREQ_MAX
        ],

        "sample_rate": SAMPLE_RATE,

        "stft": {
            "nperseg": 4096,
            "noverlap": 3072,
            "return_onesided": False,
            "fftshift": True,
        },
    },

    "class_map": master["class_map"],

    "chunks": []
}


# ============================================================
# PROCESS ALL CHUNKS
# ============================================================

total_annotations = 0

for index, chunk in enumerate(chunks):

    chunk_id = chunk["chunk_id"]

    print(
        f"[{index + 1:03d}/{len(chunks):03d}] "
        f"Processing chunk {chunk_id:04d}"
    )

    chunk_time_start = chunk["time_start"]
    chunk_time_end = chunk["time_end"]

    new_annotations = []

    for ann in chunk["annotations"]:

        class_id = ann["class_id"]
        class_name = ann["class_name"]

        freq_low = ann["freq_low"]
        freq_high = ann["freq_high"]

        ann_time_start = ann["time_start"]
        ann_time_end = ann["time_end"]

        # ----------------------------------------------------
        # TIME -> X
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # FREQUENCY -> Y
        # ----------------------------------------------------

        y_min = frequency_to_y(freq_high)
        y_max = frequency_to_y(freq_low)

        # ----------------------------------------------------
        # CLIP TO IMAGE
        # ----------------------------------------------------

        x_min = max(
            0.0,
            min(IMAGE_WIDTH, x_min)
        )

        x_max = max(
            0.0,
            min(IMAGE_WIDTH, x_max)
        )

        y_min = max(
            0.0,
            min(IMAGE_HEIGHT, y_min)
        )

        y_max = max(
            0.0,
            min(IMAGE_HEIGHT, y_max)
        )

        # ----------------------------------------------------
        # NORMALIZED COORDINATES
        # ----------------------------------------------------

        bbox_normalized = {
            "x_min": x_min / IMAGE_WIDTH,
            "y_min": y_min / IMAGE_HEIGHT,
            "x_max": x_max / IMAGE_WIDTH,
            "y_max": y_max / IMAGE_HEIGHT,
        }

        # ----------------------------------------------------
        # SAVE ANNOTATION
        # ----------------------------------------------------

        new_annotation = {
            "class_id": class_id,
            "class_name": class_name,

            "bbox": {
                "x_min": round(x_min, 3),
                "y_min": round(y_min, 3),
                "x_max": round(x_max, 3),
                "y_max": round(y_max, 3),
            },

            "bbox_normalized": {
                "x_min": round(
                    bbox_normalized["x_min"],
                    6
                ),
                "y_min": round(
                    bbox_normalized["y_min"],
                    6
                ),
                "x_max": round(
                    bbox_normalized["x_max"],
                    6
                ),
                "y_max": round(
                    bbox_normalized["y_max"],
                    6
                ),
            },

            "physical": {
                "time_start": ann_time_start,
                "time_end": ann_time_end,

                "freq_low": freq_low,
                "freq_high": freq_high,
            },

            "source": {
                "sample_start": ann["sample_start"],
                "sample_end": ann["sample_end"],
            },
        }

        new_annotations.append(new_annotation)

        total_annotations += 1

    # --------------------------------------------------------
    # SAVE CHUNK
    # --------------------------------------------------------

    new_chunk = {
        "image_id": chunk["image_id"],
        "image_file": chunk["image_file"],
        "chunk_id": chunk_id,

        "sample_start": chunk["sample_start"],
        "sample_end": chunk["sample_end"],
        "sample_count": chunk["sample_count"],

        "time_start": chunk_time_start,
        "time_end": chunk_time_end,

        "spectrogram_shape": [
            FREQ_BINS,
            TIME_BINS
        ],

        "annotations": new_annotations,
    }

    output["chunks"].append(new_chunk)


# ============================================================
# SAVE
# ============================================================

print()
print("=" * 70)
print("SAVING")
print("=" * 70)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )

print()
print(f"Saved: {OUTPUT_FILE}")


# ============================================================
# QA
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET QA")
print("=" * 70)

expected_chunks = len(chunks)
actual_chunks = len(output["chunks"])

expected_annotations = expected_chunks * 4
actual_annotations = total_annotations

missing_images = 0
wrong_bbox = 0
wrong_class = 0

for chunk in output["chunks"]:

    image_path = (
        Path(r"data\spectrograms_color")
        / chunk["image_file"]
    )

    if not image_path.exists():
        missing_images += 1

    for ann in chunk["annotations"]:

        # Check class
        if ann["class_name"] not in master["class_map"]:
            wrong_class += 1

        # Check bbox
        bbox = ann["bbox"]

        if not (
            0 <= bbox["x_min"] <= IMAGE_WIDTH
            and
            0 <= bbox["x_max"] <= IMAGE_WIDTH
            and
            0 <= bbox["y_min"] <= IMAGE_HEIGHT
            and
            0 <= bbox["y_max"] <= IMAGE_HEIGHT
            and
            bbox["x_min"] <= bbox["x_max"]
            and
            bbox["y_min"] <= bbox["y_max"]
        ):
            wrong_bbox += 1


# ============================================================
# QA RESULT
# ============================================================

print()
print(f"Expected chunks      : {expected_chunks}")
print(f"Actual chunks        : {actual_chunks}")

print(
    f"Expected annotations : {expected_annotations}"
)

print(
    f"Actual annotations   : {actual_annotations}"
)

print(
    f"Missing images       : {missing_images}"
)

print(
    f"Wrong bounding boxes : {wrong_bbox}"
)

print(
    f"Wrong classes        : {wrong_class}"
)


# ============================================================
# FINAL STATUS
# ============================================================

success = (
    expected_chunks == actual_chunks
    and
    expected_annotations == actual_annotations
    and
    missing_images == 0
    and
    wrong_bbox == 0
    and
    wrong_class == 0
)

print()
print("=" * 70)

if success:
    print("STATUS: SUCCESS")
else:
    print("STATUS: FAILED")

print("=" * 70)