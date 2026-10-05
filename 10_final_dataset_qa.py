import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MASTER_FILE = Path(
    r"data\annotations\master_annotations.json"
)

PIXEL_FILE = Path(
    r"data\annotations\pixel_annotations.json"
)

IMAGE_DIR = Path(
    r"data\spectrograms_color"
)


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 70)
print("FINAL MASTER DATASET QA")
print("=" * 70)

print()
print("Loading files...")

with open(MASTER_FILE, "r", encoding="utf-8") as f:
    master = json.load(f)

with open(PIXEL_FILE, "r", encoding="utf-8") as f:
    pixel = json.load(f)

print("Files loaded successfully.")


# ============================================================
# BASIC INFORMATION
# ============================================================

master_chunks = master["chunks"]
pixel_chunks = pixel["chunks"]

class_map = master["class_map"]

print()
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Dataset name : {master['dataset_name']}")
print(f"Source file  : {master['source_file']}")
print(f"Chunks       : {len(master_chunks)}")
print(f"Classes      : {len(class_map)}")

print()
print("Class map:")

for name, class_id in class_map.items():
    print(f"  {class_id} -> {name}")


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_CHUNKS = 100
EXPECTED_ANNOTATIONS_PER_CHUNK = 4
EXPECTED_TOTAL_ANNOTATIONS = 400

EXPECTED_CLASS_IDS = {0, 1, 2, 3}


# ============================================================
# COUNTERS
# ============================================================

errors = []

missing_images = 0
missing_pixel_chunks = 0
wrong_chunk_ids = 0
wrong_annotation_count = 0
wrong_classes = 0
wrong_bbox = 0
wrong_normalized = 0
wrong_image_reference = 0

total_annotations = 0


# ============================================================
# CHECK NUMBER OF CHUNKS
# ============================================================

if len(master_chunks) != EXPECTED_CHUNKS:
    errors.append(
        f"Master chunk count is {len(master_chunks)}, "
        f"expected {EXPECTED_CHUNKS}"
    )

if len(pixel_chunks) != EXPECTED_CHUNKS:
    errors.append(
        f"Pixel chunk count is {len(pixel_chunks)}, "
        f"expected {EXPECTED_CHUNKS}"
    )


# ============================================================
# CREATE PIXEL CHUNK LOOKUP
# ============================================================

pixel_lookup = {
    chunk["chunk_id"]: chunk
    for chunk in pixel_chunks
}


# ============================================================
# CHECK EACH CHUNK
# ============================================================

for master_chunk in master_chunks:

    chunk_id = master_chunk["chunk_id"]
    image_file = master_chunk["image_file"]

    # --------------------------------------------------------
    # IMAGE EXISTENCE
    # --------------------------------------------------------

    image_path = IMAGE_DIR / image_file

    if not image_path.exists():

        missing_images += 1

        errors.append(
            f"Chunk {chunk_id}: "
            f"missing image {image_file}"
        )

    # --------------------------------------------------------
    # PIXEL CHUNK EXISTENCE
    # --------------------------------------------------------

    if chunk_id not in pixel_lookup:

        missing_pixel_chunks += 1

        errors.append(
            f"Chunk {chunk_id}: "
            f"missing pixel annotation chunk"
        )

        continue

    pixel_chunk = pixel_lookup[chunk_id]

    # --------------------------------------------------------
    # CHUNK ID
    # --------------------------------------------------------

    if pixel_chunk["chunk_id"] != chunk_id:

        wrong_chunk_ids += 1

        errors.append(
            f"Chunk ID mismatch: "
            f"{chunk_id}"
        )

    # --------------------------------------------------------
    # IMAGE REFERENCE
    # --------------------------------------------------------

    if pixel_chunk["image_file"] != image_file:

        wrong_image_reference += 1

        errors.append(
            f"Chunk {chunk_id}: image reference mismatch"
        )

    # --------------------------------------------------------
    # ANNOTATION COUNT
    # --------------------------------------------------------

    master_annotations = master_chunk["annotations"]
    pixel_annotations = pixel_chunk["annotations"]

    if len(master_annotations) != EXPECTED_ANNOTATIONS_PER_CHUNK:

        wrong_annotation_count += 1

        errors.append(
            f"Chunk {chunk_id}: "
            f"master annotation count = "
            f"{len(master_annotations)}"
        )

    if len(pixel_annotations) != EXPECTED_ANNOTATIONS_PER_CHUNK:

        wrong_annotation_count += 1

        errors.append(
            f"Chunk {chunk_id}: "
            f"pixel annotation count = "
            f"{len(pixel_annotations)}"
        )

    # --------------------------------------------------------
    # CHECK PIXEL ANNOTATIONS
    # --------------------------------------------------------

    for ann in pixel_annotations:

        total_annotations += 1

        class_id = ann["class_id"]
        class_name = ann["class_name"]

        # ----------------------------------------------------
        # CLASS
        # ----------------------------------------------------

        if class_id not in EXPECTED_CLASS_IDS:

            wrong_classes += 1

            errors.append(
                f"Chunk {chunk_id}: "
                f"invalid class ID {class_id}"
            )

        if class_name not in class_map:

            wrong_classes += 1

            errors.append(
                f"Chunk {chunk_id}: "
                f"invalid class name {class_name}"
            )

        elif class_map[class_name] != class_id:

            wrong_classes += 1

            errors.append(
                f"Chunk {chunk_id}: "
                f"class mismatch "
                f"{class_name} -> {class_id}"
            )

        # ----------------------------------------------------
        # BBOX
        # ----------------------------------------------------

        bbox = ann["bbox"]

        x_min = bbox["x_min"]
        y_min = bbox["y_min"]
        x_max = bbox["x_max"]
        y_max = bbox["y_max"]

        valid_bbox = (
            0 <= x_min <= 1500
            and
            0 <= x_max <= 1500
            and
            0 <= y_min <= 1500
            and
            0 <= y_max <= 1500
            and
            x_min <= x_max
            and
            y_min <= y_max
        )

        if not valid_bbox:

            wrong_bbox += 1

            errors.append(
                f"Chunk {chunk_id}: "
                f"invalid bbox {bbox}"
            )

        # ----------------------------------------------------
        # NORMALIZED BBOX
        # ----------------------------------------------------

        normalized = ann["bbox_normalized"]

        values = [
            normalized["x_min"],
            normalized["y_min"],
            normalized["x_max"],
            normalized["y_max"],
        ]

        valid_normalized = (
            all(0 <= v <= 1 for v in values)
            and
            normalized["x_min"] <= normalized["x_max"]
            and
            normalized["y_min"] <= normalized["y_max"]
        )

        if not valid_normalized:

            wrong_normalized += 1

            errors.append(
                f"Chunk {chunk_id}: "
                f"invalid normalized bbox "
                f"{normalized}"
            )


# ============================================================
# TOTAL ANNOTATION CHECK
# ============================================================

if total_annotations != EXPECTED_TOTAL_ANNOTATIONS:

    errors.append(
        f"Total annotations = {total_annotations}, "
        f"expected {EXPECTED_TOTAL_ANNOTATIONS}"
    )


# ============================================================
# CHECK IMAGE COUNT
# ============================================================

image_files = list(
    IMAGE_DIR.glob("chunk_*.png")
)

print()
print("=" * 70)
print("IMAGE CHECK")
print("=" * 70)

print(f"PNG images found : {len(image_files)}")

if len(image_files) != EXPECTED_CHUNKS:

    errors.append(
        f"Image count = {len(image_files)}, "
        f"expected {EXPECTED_CHUNKS}"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("FINAL QA REPORT")
print("=" * 70)

print()
print(f"Expected chunks          : {EXPECTED_CHUNKS}")
print(f"Master chunks             : {len(master_chunks)}")
print(f"Pixel chunks              : {len(pixel_chunks)}")

print()
print(
    f"Expected annotations      : "
    f"{EXPECTED_TOTAL_ANNOTATIONS}"
)

print(
    f"Actual annotations        : "
    f"{total_annotations}"
)

print()
print(f"Missing images            : {missing_images}")
print(
    f"Missing pixel chunks      : "
    f"{missing_pixel_chunks}"
)

print(
    f"Wrong chunk IDs           : "
    f"{wrong_chunk_ids}"
)

print(
    f"Wrong annotation counts   : "
    f"{wrong_annotation_count}"
)

print(
    f"Wrong image references    : "
    f"{wrong_image_reference}"
)

print(
    f"Wrong classes             : "
    f"{wrong_classes}"
)

print(
    f"Wrong bounding boxes      : "
    f"{wrong_bbox}"
)

print(
    f"Wrong normalized boxes    : "
    f"{wrong_normalized}"
)


# ============================================================
# ERROR DETAILS
# ============================================================

if errors:

    print()
    print("=" * 70)
    print("ERROR DETAILS")
    print("=" * 70)

    for error in errors:
        print("ERROR:", error)


# ============================================================
# STATUS
# ============================================================

print()
print("=" * 70)

if not errors:

    print("STATUS: MASTER DATASET QA PASSED")

else:

    print(
        f"STATUS: FAILED "
        f"({len(errors)} errors)"
    )

print("=" * 70)