import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from matplotlib.patches import Rectangle


# ==========================================
# 1. Input
# ==========================================

data_path = r"../dataset/train/west-wideband-modrec-ex1-tmpl2-20.04.sigmf-data"

num_samples = 1_000_000_000


# ==========================================
# 2. Read IQ
# ==========================================

print("Reading IQ data...")

iq = np.fromfile(
    data_path,
    dtype=np.complex64,
    count=num_samples
)

print(f"Loaded {len(iq):,} IQ samples")


# ==========================================
# 3. STFT
# ==========================================

fs = 1.0

nperseg = 4096
noverlap = 3072

print("Computing STFT...")

f, t, Zxx = stft(
    iq,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    return_onesided=False
)


# ==========================================
# 4. Power in dB
# ==========================================

power = np.abs(Zxx) ** 2

power_db = 10 * np.log10(
    power + 1e-12
)


# ==========================================
# 5. Shift frequency
# ==========================================

f_shifted = np.fft.fftshift(f)

power_db_shifted = np.fft.fftshift(
    power_db,
    axes=0
)


# ==========================================
# 6. Plot
# ==========================================

fig, ax = plt.subplots(figsize=(14, 7))

ax.imshow(
    power_db_shifted,
    aspect="auto",
    origin="lower",
    extent=[
        t[0],
        t[-1],
        f_shifted[0],
        f_shifted[-1]
    ]
)

ax.set_xlabel("Time")
ax.set_ylabel("Normalized Frequency")

ax.set_title(
    "IQ Spectrogram with SigMF Annotations"
)


# ==========================================
# 7. SigMF annotations
# ==========================================

annotations = [
    {
        "name": "PSK4",
        "f_low": 0.2568932634,
        "f_high": 0.3048932634
    },
    {
        "name": "QAM64",
        "f_low": -0.2400610411,
        "f_high": -0.1920610411
    },
    {
        "name": "PSK8",
        "f_low": 0.3681675179,
        "f_high": 0.4641675179
    },
    {
        "name": "OFDM",
        "f_low": 0.0415955920,
        "f_high": 0.2335955920
    }
]


# ==========================================
# 8. Draw annotation boxes
# ==========================================

time_start = 0
time_end = t[-1]

for ann in annotations:

    f_low = ann["f_low"]
    f_high = ann["f_high"]

    width = time_end - time_start
    height = f_high - f_low

    rect = Rectangle(
        (time_start, f_low),
        width,
        height,
        fill=False,
        linewidth=2
    )

    ax.add_patch(rect)

    ax.text(
        time_start + width * 0.01,
        f_high,
        ann["name"],
        fontsize=11,
        verticalalignment="bottom",
        fontweight="bold"
    )


plt.colorbar(
    ax.images[0],
    ax=ax,
    label="Power (dB)"
)

plt.tight_layout()

output_path = r"../dataset/spectrograms/sample_0000.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

print(f"Saved spectrogram: {output_path}")

plt.show()