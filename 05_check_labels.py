import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


# =========================
# Paths
# =========================

data_path = r"data\west-wideband-modrec-ex1-tmpl2-20.04.sigmf-data"
meta_path = r"data\west-wideband-modrec-ex1-tmpl2-20.04.sigmf-meta"

output_path = r"data\spectrograms\chunk_0000_QA_labels.png"


# =========================
# Settings
# =========================

sample_rate = 1.0
chunk_start = 0
chunk_size = 1_000_000

nperseg = 4096
noverlap = 3072


# =========================
# Class information
# =========================

class_names = {
    "PSK4": 0,
    "QAM64": 1,
    "PSK8": 2,
    "OFDM": 3,
}

class_colors = {
    "PSK4": "red",
    "QAM64": "blue",
    "PSK8": "yellow",
    "OFDM": "cyan",
}


# =========================
# Read metadata
# =========================

print("Reading SigMF metadata...")

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

annotations = meta["annotations"]

print(f"Found {len(annotations)} annotations")


# =========================
# Read IQ chunk
# =========================

print("Reading IQ chunk...")

iq = np.fromfile(
    data_path,
    dtype=np.complex64,
    count=chunk_size,
    offset=chunk_start * 8,
)

print(f"Loaded {len(iq):,} samples")


# =========================
# STFT
# =========================

print("Computing STFT...")

f, t, Zxx = stft(
    iq,
    fs=sample_rate,
    nperseg=nperseg,
    noverlap=noverlap,
    return_onesided=False,
)

power = np.abs(Zxx) ** 2

power_db = 10 * np.log10(power + 1e-12)


# =========================
# Shift frequency
# =========================

f = np.fft.fftshift(f)

power_db = np.fft.fftshift(
    power_db,
    axes=0,
)


# =========================
# Display range
# =========================

vmax = np.percentile(power_db, 99)
vmin = vmax - 60


# =========================
# Plot
# =========================

fig, ax = plt.subplots(figsize=(14, 8))

ax.imshow(
    power_db,
    aspect="auto",
    origin="lower",
    extent=[
        t[0],
        t[-1],
        f[0],
        f[-1],
    ],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)


# =========================
# Draw labels
# =========================

for ann in annotations:

    label = ann.get("core:description", "UNKNOWN")

    if label not in class_names:
        print(f"Skipping unknown label: {label}")
        continue

    freq_low = ann["core:freq_lower_edge"]
    freq_high = ann["core:freq_upper_edge"]

    sample_start = ann["core:sample_start"]
    sample_count = ann["core:sample_count"]

    sample_end = sample_start + sample_count

    # Convert samples to time
    time_start = sample_start / sample_rate
    time_end = sample_end / sample_rate

    print()
    print(f"Class      : {label}")
    print(f"Class ID   : {class_names[label]}")
    print(f"Frequency  : {freq_low:.6f} -> {freq_high:.6f}")
    print(f"Samples    : {sample_start:,} -> {sample_end:,}")
    print(f"Time       : {time_start:.2f} -> {time_end:.2f}")

    color = class_colors[label]

    rect = plt.Rectangle(
        (time_start, freq_low),
        time_end - time_start,
        freq_high - freq_low,
        fill=False,
        edgecolor=color,
        linewidth=2,
    )

    ax.add_patch(rect)

    ax.text(
        time_start,
        freq_high,
        label,
        color=color,
        fontsize=12,
        fontweight="bold",
        verticalalignment="bottom",
    )


# =========================
# Axis labels
# =========================

ax.set_xlabel("Time")
ax.set_ylabel("Normalized Frequency")

ax.set_title(
    "Chunk 0000 - SigMF Label Verification"
)

plt.tight_layout()


# =========================
# Save
# =========================

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

print()
print("Saved QA image:")
print(output_path)

plt.show()