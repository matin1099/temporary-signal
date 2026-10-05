import os
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


# ============================================================
# 1. PATHS
# ============================================================

DATA_FILE = r"data\west-wideband-modrec-ex1-tmpl2-20.04.sigmf-data"
META_FILE = r"data\west-wideband-modrec-ex1-tmpl2-20.04.sigmf-meta"

# IMPORTANT:
# Old grayscale images are kept untouched.
# New color images will be saved here.
SPECTROGRAM_DIR = r"data\spectrograms_color"

ANNOTATION_DIR = r"data\annotations"
METADATA_DIR = r"data\metadata"


# ============================================================
# 2. DATASET SETTINGS
# ============================================================

SAMPLE_RATE = 1.0

TOTAL_SAMPLES = 100_000_000

CHUNK_SIZE = 1_000_000

TOTAL_CHUNKS = TOTAL_SAMPLES // CHUNK_SIZE


# ============================================================
# 3. STFT SETTINGS
# ============================================================

NPERSEG = 4096

NOVERLAP = 3072

RETURN_ONESIDED = False


# ============================================================
# 4. SPECTROGRAM DISPLAY SETTINGS
# ============================================================

DISPLAY_VMAX = -37.81
DISPLAY_VMIN = -97.81

# Color map for the new spectrogram images
COLORMAP = "turbo"


# ============================================================
# 5. CLASS MAPPING
# ============================================================

CLASS_MAP = {
    "PSK4": 0,
    "QAM64": 1,
    "PSK8": 2,
    "OFDM": 3,
}


# ============================================================
# 6. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
os.makedirs(ANNOTATION_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


# ============================================================
# 7. READ SIGMF METADATA
# ============================================================

print("=" * 70)
print("READING SIGMF METADATA")
print("=" * 70)

with open(META_FILE, "r", encoding="utf-8") as f:
    meta = json.load(f)

annotations = meta.get("annotations", [])

print(f"Found annotations: {len(annotations)}")


# ============================================================
# 8. PARSE MASTER ANNOTATIONS
# ============================================================

master_signal_annotations = []

for ann in annotations:

    # Signal class is stored in core:description
    label = ann.get(
        "core:description",
        "UNKNOWN"
    )

    if label not in CLASS_MAP:
        print(
            f"WARNING: Unknown label: {label}"
        )
        continue

    sample_start = int(
        ann.get(
            "core:sample_start",
            0
        )
    )

    sample_count = int(
        ann.get(
            "core:sample_count",
            0
        )
    )

    sample_end = (
        sample_start +
        sample_count
    )

    freq_low = float(
        ann["core:freq_lower_edge"]
    )

    freq_high = float(
        ann["core:freq_upper_edge"]
    )

    master_signal_annotations.append({

        "class_id":
            CLASS_MAP[label],

        "class_name":
            label,

        "sample_start":
            sample_start,

        "sample_end":
            sample_end,

        "freq_low":
            freq_low,

        "freq_high":
            freq_high,
    })


print("\nSignal classes:")

for ann in master_signal_annotations:

    print(
        f"  {ann['class_id']} | "
        f"{ann['class_name']:5s} | "
        f"{ann['freq_low']:.6f} "
        f"to "
        f"{ann['freq_high']:.6f}"
    )


# ============================================================
# 9. OPEN IQ DATA WITH MEMMAP
# ============================================================

print("\n" + "=" * 70)
print("OPENING IQ DATA")
print("=" * 70)

iq_data = np.memmap(
    DATA_FILE,
    dtype=np.complex64,
    mode="r",
    shape=(TOTAL_SAMPLES,)
)

print(
    f"Total samples : {TOTAL_SAMPLES:,}"
)

print(
    f"Chunk size    : {CHUNK_SIZE:,}"
)

print(
    f"Total chunks  : {TOTAL_CHUNKS}"
)


# ============================================================
# 10. PROCESS ALL CHUNKS
# ============================================================

all_chunks = []


for chunk_id in range(TOTAL_CHUNKS):

    print("\n")
    print("=" * 70)

    print(
        f"PROCESSING CHUNK "
        f"{chunk_id:04d} / "
        f"{TOTAL_CHUNKS - 1:04d}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Sample range
    # --------------------------------------------------------

    chunk_start = (
        chunk_id *
        CHUNK_SIZE
    )

    chunk_end = min(
        chunk_start +
        CHUNK_SIZE,
        TOTAL_SAMPLES
    )

    chunk_count = (
        chunk_end -
        chunk_start
    )

    print(
        f"Samples: "
        f"{chunk_start:,} -> "
        f"{chunk_end:,}"
    )


    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    image_name = (
        f"chunk_{chunk_id:04d}.png"
    )

    image_path = os.path.join(
        SPECTROGRAM_DIR,
        image_name
    )


    # --------------------------------------------------------
    # Read IQ
    # --------------------------------------------------------

    print("Reading IQ...")

    iq = np.asarray(
        iq_data[
            chunk_start:chunk_end
        ]
    )

    print(
        f"Loaded: "
        f"{len(iq):,} samples"
    )


    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    print("Computing STFT...")

    frequencies, times, Zxx = stft(
        iq,
        fs=SAMPLE_RATE,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        return_onesided=RETURN_ONESIDED,
    )


    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    power = (
        np.abs(Zxx) ** 2
    )

    power_db = (
        10 *
        np.log10(
            power + 1e-12
        )
    )


    # --------------------------------------------------------
    # FFT shift
    # --------------------------------------------------------

    frequencies = np.fft.fftshift(
        frequencies
    )

    power_db = np.fft.fftshift(
        power_db,
        axes=0
    )


    print(
        f"Spectrogram: "
        f"{power_db.shape}"
    )


    # ========================================================
    # 11. SAVE COLOR SPECTROGRAM
    # ========================================================

    if os.path.exists(image_path):

        print(
            f"Image already exists: "
            f"{image_name}"
        )

    else:

        print(
            "Saving COLOR spectrogram..."
        )

        fig = plt.figure(
            figsize=(10, 10),
            dpi=150
        )

        ax = fig.add_axes(
            [0, 0, 1, 1]
        )

        ax.imshow(
            power_db,
            origin="lower",
            aspect="auto",

            # IMPORTANT:
            # Color representation
            cmap=COLORMAP,

            # Fixed intensity range
            vmin=DISPLAY_VMIN,
            vmax=DISPLAY_VMAX,
        )

        # Remove axes
        ax.axis("off")

        fig.savefig(
            image_path,
            format="png",
            dpi=150,
            bbox_inches=None,
            pad_inches=0
        )

        plt.close(fig)

        print(
            f"Saved: {image_name}"
        )


    # ========================================================
    # 12. CREATE CHUNK ANNOTATIONS
    # ========================================================

    chunk_annotations = []


    for ann in master_signal_annotations:

        # ----------------------------------------------------
        # Intersection with current chunk
        # ----------------------------------------------------

        intersection_start = max(
            chunk_start,
            ann["sample_start"]
        )

        intersection_end = min(
            chunk_end,
            ann["sample_end"]
        )


        # No overlap
        if (
            intersection_start >=
            intersection_end
        ):
            continue


        # ----------------------------------------------------
        # Relative sample coordinates
        # ----------------------------------------------------

        relative_start = (
            intersection_start -
            chunk_start
        )

        relative_end = (
            intersection_end -
            chunk_start
        )


        # ----------------------------------------------------
        # Relative time
        # ----------------------------------------------------

        time_start = (
            relative_start /
            SAMPLE_RATE
        )

        time_end = (
            relative_end /
            SAMPLE_RATE
        )


        # ----------------------------------------------------
        # Annotation
        # ----------------------------------------------------

        chunk_annotations.append({

            "class_id":
                ann["class_id"],

            "class_name":
                ann["class_name"],

            "sample_start":
                int(relative_start),

            "sample_end":
                int(relative_end),

            "time_start":
                float(time_start),

            "time_end":
                float(time_end),

            "freq_low":
                float(ann["freq_low"]),

            "freq_high":
                float(ann["freq_high"]),
        })


    # ========================================================
    # 13. CREATE CHUNK RECORD
    # ========================================================

    chunk_record = {

        "image_id":
            f"chunk_{chunk_id:04d}",

        "image_file":
            image_name,

        "source_file":
            os.path.basename(
                DATA_FILE
            ),

        "chunk_id":
            chunk_id,

        "sample_start":
            int(chunk_start),

        "sample_end":
            int(chunk_end),

        "sample_count":
            int(chunk_count),

        "time_start":
            float(
                chunk_start /
                SAMPLE_RATE
            ),

        "time_end":
            float(
                chunk_end /
                SAMPLE_RATE
            ),

        "sample_rate":
            SAMPLE_RATE,


        "spectrogram": {

            "frequency_bins":
                int(
                    len(frequencies)
                ),

            "time_bins":
                int(
                    len(times)
                ),

            "shape":
                [
                    int(
                        power_db.shape[0]
                    ),

                    int(
                        power_db.shape[1]
                    )
                ],
        },


        "stft": {

            "nperseg":
                NPERSEG,

            "noverlap":
                NOVERLAP,

            "return_onesided":
                RETURN_ONESIDED,

            "fftshift":
                True,
        },


        "display": {

            "vmin_db":
                DISPLAY_VMIN,

            "vmax_db":
                DISPLAY_VMAX,

            "colormap":
                COLORMAP,

            "image_format":
                "PNG",

            "image_type":
                "RGB color",
        },


        "annotations":
            chunk_annotations,
    }


    all_chunks.append(
        chunk_record
    )


    print(
        f"Annotations: "
        f"{len(chunk_annotations)}"
    )


# ============================================================
# 14. BUILD MASTER ANNOTATION FILE
# ============================================================

print("\n")

print("=" * 70)
print("SAVING MASTER ANNOTATIONS")
print("=" * 70)


master_output = {

    "dataset_name":
        "West Wideband ModRec",

    "format":
        "Master algorithm-independent annotation",

    "source_format":
        "SigMF",

    "source_file":
        os.path.basename(
            DATA_FILE
        ),

    "total_samples":
        TOTAL_SAMPLES,

    "sample_rate":
        SAMPLE_RATE,

    "chunk_size":
        CHUNK_SIZE,

    "number_of_chunks":
        TOTAL_CHUNKS,

    "class_map":
        CLASS_MAP,

    "stft": {

        "nperseg":
            NPERSEG,

        "noverlap":
            NOVERLAP,

        "return_onesided":
            RETURN_ONESIDED,

        "fftshift":
            True,
    },

    "spectrogram": {

        "colormap":
            COLORMAP,

        "vmin_db":
            DISPLAY_VMIN,

        "vmax_db":
            DISPLAY_VMAX,

        "format":
            "PNG",

        "type":
            "RGB color",
    },

    "chunks":
        all_chunks,
}


MASTER_JSON = os.path.join(
    ANNOTATION_DIR,
    "master_annotations.json"
)


with open(
    MASTER_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        master_output,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"Saved: {MASTER_JSON}"
)


# ============================================================
# 15. DATASET METADATA
# ============================================================

dataset_metadata = {

    "dataset_name":
        "West Wideband ModRec",

    "source_file":
        os.path.basename(
            DATA_FILE
        ),

    "source_format":
        "SigMF",

    "datatype":
        "cf32_le",

    "total_samples":
        TOTAL_SAMPLES,

    "sample_rate":
        SAMPLE_RATE,

    "chunk_size":
        CHUNK_SIZE,

    "number_of_chunks":
        TOTAL_CHUNKS,

    "processed_chunks":
        TOTAL_CHUNKS,

    "classes":
        CLASS_MAP,


    "stft": {

        "nperseg":
            NPERSEG,

        "noverlap":
            NOVERLAP,

        "return_onesided":
            RETURN_ONESIDED,

        "fftshift":
            True,
    },


    "spectrogram": {

        "frequency_bins":
            NPERSEG,

        "time_bins":
            978,

        "colormap":
            COLORMAP,

        "image_format":
            "PNG",

        "image_type":
            "RGB color",
    },


    "display_range_db": {

        "vmin":
            DISPLAY_VMIN,

        "vmax":
            DISPLAY_VMAX,
    },
}


METADATA_JSON = os.path.join(
    METADATA_DIR,
    "dataset_metadata.json"
)


with open(
    METADATA_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dataset_metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"Saved: {METADATA_JSON}"
)


# ============================================================
# 16. AUTOMATIC QA
# ============================================================

print("\n")

print("=" * 70)
print("DATASET QA")
print("=" * 70)


expected_chunks = TOTAL_CHUNKS

actual_chunks = len(
    all_chunks
)


print(
    f"Expected chunks : "
    f"{expected_chunks}"
)

print(
    f"JSON chunks     : "
    f"{actual_chunks}"
)


missing_images = []

wrong_shape = []

wrong_annotation_count = []


for chunk in all_chunks:

    image_file = os.path.join(
        SPECTROGRAM_DIR,
        chunk["image_file"]
    )


    # --------------------------------------------------------
    # Check image exists
    # --------------------------------------------------------

    if not os.path.exists(
        image_file
    ):

        missing_images.append(
            chunk["image_file"]
        )


    # --------------------------------------------------------
    # Check STFT shape
    # --------------------------------------------------------

    expected_shape = [
        NPERSEG,
        chunk["spectrogram"]["time_bins"]
    ]

    if (
        chunk["spectrogram"]["shape"]
        != expected_shape
    ):

        wrong_shape.append(
            chunk["image_id"]
        )


    # --------------------------------------------------------
    # Check annotation count
    # --------------------------------------------------------

    if len(
        chunk["annotations"]
    ) != len(
        CLASS_MAP
    ):

        wrong_annotation_count.append(
            chunk["image_id"]
        )


print(
    f"Missing images  : "
    f"{len(missing_images)}"
)

print(
    f"Wrong shapes    : "
    f"{len(wrong_shape)}"
)

print(
    f"Wrong labels    : "
    f"{len(wrong_annotation_count)}"
)


# ============================================================
# 17. FINAL RESULT
# ============================================================

print("\n")

print("=" * 70)
print("DATASET BUILD FINISHED")
print("=" * 70)


if (
    actual_chunks == expected_chunks
    and len(missing_images) == 0
    and len(wrong_shape) == 0
    and len(wrong_annotation_count) == 0
):

    print(
        "STATUS: SUCCESS"
    )

else:

    print(
        "STATUS: CHECK REQUIRED"
    )


print(
    f"\nSpectrogram directory:"
    f"\n{SPECTROGRAM_DIR}"
)

print(
    f"\nMaster annotations:"
    f"\n{MASTER_JSON}"
)

print(
    f"\nDataset metadata:"
    f"\n{METADATA_JSON}"
)